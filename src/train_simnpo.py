"""
train_simnpo.py — SimNPO (NeurIPS 2025) for any RWKU entity.
Reference-free, length-normalized negative preference optimization.
Loss: L = -(2/beta) * log_sigmoid(beta * (mean_token_NLL - delta))

Usage:
  python3 src/train_simnpo.py --subject "Taylor Swift" --model_size 1b
"""
import argparse, sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    MODEL_1B, MODEL_8B, RESULTS_DIR,
    SIMNPO_BETA, SIMNPO_DELTA, SIMNPO_LR, SIMNPO_STEPS,
    LORA_R_1B, LORA_R_8B,
    ALPHA_1B, ALPHA_8B,
    ALL_RWKU_ENTITIES,
)
from data_loader import load_forget_rows, load_retain_rows

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",    required=True)
    p.add_argument("--model_size", default="1b", choices=["1b", "8b"])
    p.add_argument("--beta",       type=float, default=SIMNPO_BETA)
    p.add_argument("--delta",      type=float, default=SIMNPO_DELTA)
    p.add_argument("--steps",      type=int,   default=SIMNPO_STEPS)
    p.add_argument("--retain_weight", type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def simnpo_loss(logits, labels, beta, delta, mask):
    """Compute SimNPO loss on a batch of forget examples."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask   = mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Length-normalized NLL
    seq_nll = -(token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)

    # SimNPO objective
    loss = -(2.0 / beta) * F.logsigmoid(beta * (seq_nll - delta)).mean()
    return loss


class SimNPOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, delta=2.0, retain_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta          = beta
        self.delta         = delta
        self.retain_weight = retain_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row_type = inputs.pop("row_type")
        labels   = inputs["input_ids"].clone()
        mask     = inputs.get("attention_mask", torch.ones_like(labels))

        outputs = model(**inputs)
        logits  = outputs.logits

        forget_mask = (row_type == 0)
        retain_mask = (row_type == 1)

        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        if forget_mask.any():
            loss = loss + simnpo_loss(
                logits[forget_mask], labels[forget_mask],
                self.beta, self.delta, mask[forget_mask],
            )

        if retain_mask.any():
            shift_logits = logits[retain_mask][..., :-1, :].contiguous()
            shift_labels = labels[retain_mask][..., 1:].contiguous()
            pad_id = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            pad_id = pad_id.pad_token_id if pad_id is not None else -100
            retain_loss  = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=pad_id,
            )
            loss = loss + self.retain_weight * retain_loss

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    is_8b  = args.model_size == "8b"
    base   = MODEL_8B if is_8b else MODEL_1B
    lora_r = LORA_R_8B if is_8b else LORA_R_1B

    subject_slug = slug(args.subject)
    out_dir = RESULTS_DIR / f"simnpo_{args.model_size}_{subject_slug}"

    print(f"\n{'='*60}")
    print(f"SimNPO  subject={args.subject}  model={args.model_size}")
    print(f"beta={args.beta}  delta={args.delta}  steps={args.steps}")
    print('='*60)

    # ── Data ──────────────────────────────────────────────────────────
    forget_rows = load_forget_rows(args.subject)
    ood_subjects = [s for s in ALL_RWKU_ENTITIES if s != args.subject]
    random.seed(args.seed)
    random.shuffle(ood_subjects)
    retain_rows = load_retain_rows(ood_subjects[:15], n_per_level=33)

    examples = (
        [{"text": r["query"] + " " + r["answer"], "row_type": 0} for r in forget_rows] +
        [{"text": r["query"] + " " + r["answer"], "row_type": 1} for r in retain_rows]
    )
    random.shuffle(examples)

    # ── Model ─────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    def tokenize(ex):
        enc = tok(ex["text"], truncation=True, max_length=512, padding="max_length")
        enc["row_type"] = ex["row_type"]
        return enc

    hf_ds = Dataset.from_list(examples).map(tokenize, remove_columns=["text"])

    trainer = SimNPOTrainer(
        model=model,
        processing_class=tok,
        beta=args.beta,
        delta=args.delta,
        retain_weight=args.retain_weight,
        train_dataset=hf_ds,
        args=TrainingArguments(
            output_dir=str(out_dir),
            max_steps=args.steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=args.steps and SIMNPO_LR or SIMNPO_LR,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=True,
            logging_steps=20,
            save_steps=args.steps,
            save_total_limit=1,
            report_to="none",
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
