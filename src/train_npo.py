"""train_npo.py — Negative Preference Optimization (Zhang et al., arXiv:2404.05868).

Loss:  L_NPO = -(2/beta) * logsigmoid(-beta * log(pi/pi_ref))   on forget set
       + retain_weight * L_CE(retain)     on retain set

NPO is DPO with only the "rejected" (negative) side: we push pi_theta DOWN
on the forget answers, anchored by a frozen reference. Proven slower
utility collapse than plain GA.

Usage:
  python3 src/train_npo.py --subject "Stephen King" --model_size 1b
"""
import argparse, sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    MODEL_1B, MODEL_8B, RESULTS_DIR,
    LORA_R_1B, LORA_R_8B, ALL_RWKU_ENTITIES,
)
from data_loader import load_forget_rows, load_retain_rows
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",    required=True)
    p.add_argument("--model_size", default="1b", choices=["1b", "8b"])
    p.add_argument("--steps",      type=int,   default=200)
    p.add_argument("--beta",       type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--retain_weight", type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def seq_logprob(logits, labels, mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask   = mask[..., 1:].contiguous().float()
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * shift_mask).sum(-1)


class NPOTrainer(Trainer):
    def __init__(self, *a, ref_model=None, beta=0.1, retain_weight=1.0, **kw):
        super().__init__(*a, **kw)
        self.ref_model     = ref_model
        self.beta          = beta
        self.retain_weight = retain_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        row_type = inputs.pop("row_type")
        ids  = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        labels = ids.clone()
        pad_id = self.processing_class.pad_token_id
        labels[labels == pad_id] = -100

        outputs = model(input_ids=ids, attention_mask=mask)
        logits  = outputs.logits

        forget_mask = (row_type == 0)
        retain_mask = (row_type == 1)
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if forget_mask.any():
            f_ids  = ids[forget_mask]
            f_mask = mask[forget_mask]
            f_logits = logits[forget_mask]
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=f_ids, attention_mask=f_mask).logits
            logp_cur = seq_logprob(f_logits,  f_ids, f_mask)
            logp_ref = seq_logprob(ref_logits, f_ids, f_mask)
            ratio = logp_cur - logp_ref
            npo_loss = -(2.0 / self.beta) * F.logsigmoid(-self.beta * ratio).mean()
            loss = loss + npo_loss

        if retain_mask.any():
            r_logits = logits[retain_mask][..., :-1, :].contiguous()
            r_labels = labels[retain_mask][..., 1:].contiguous()
            retain_loss = F.cross_entropy(
                r_logits.reshape(-1, r_logits.size(-1)),
                r_labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.retain_weight * retain_loss

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    is_8b  = args.model_size == "8b"
    base   = MODEL_8B if is_8b else MODEL_1B
    lora_r = LORA_R_8B if is_8b else LORA_R_1B

    subject_slug = slug(args.subject)
    out_dir = RESULTS_DIR / f"npo_{args.model_size}_{subject_slug}"

    print(f"\n{'='*60}")
    print(f"NPO  subject={args.subject}  model={args.model_size}  beta={args.beta}  steps={args.steps}")
    print('='*60)

    forget_rows = load_forget_rows(args.subject)
    ood = [s for s in ALL_RWKU_ENTITIES if s != args.subject]
    random.seed(args.seed); random.shuffle(ood)
    retain_rows = load_retain_rows(ood[:15], n_per_level=33)

    examples = (
        [{"text": r["query"] + " " + r["answer"], "row_type": 0} for r in forget_rows] +
        [{"text": r["query"] + " " + r["answer"], "row_type": 1} for r in retain_rows]
    )
    random.shuffle(examples)
    print(f"Forget={len(forget_rows)}  Retain={len(retain_rows)}  Total={len(examples)}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))

    def tokenize(ex):
        enc = tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        enc["row_type"] = ex["row_type"]
        return enc

    hf_ds = Dataset.from_list(examples).map(tokenize, remove_columns=["text"])

    trainer = NPOTrainer(
        model=model,
        processing_class=tok,
        ref_model=ref_model,
        beta=args.beta,
        retain_weight=args.retain_weight,
        train_dataset=hf_ds,
        args=TrainingArguments(
            output_dir=str(out_dir),
            max_steps=args.steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=True,
            logging_steps=20,
            save_steps=args.steps,
            save_total_limit=1,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
