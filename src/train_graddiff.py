"""train_graddiff.py — Gradient Difference unlearning.

Loss:  L = -L_CE(forget) + lambda * L_CE(retain)
(negative forget loss pushes probability AWAY from the ground truth answer,
retain term anchors utility on OOD subjects).

Usage:
  python3 src/train_graddiff.py --subject "Stephen King" --model_size 1b
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
    p.add_argument("--lam",        type=float, default=1.0)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


class GradDiffTrainer(Trainer):
    def __init__(self, *a, lam=1.0, **kw):
        super().__init__(*a, **kw)
        self.lam = lam

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        row_type = inputs.pop("row_type")
        labels = inputs["input_ids"].clone()
        pad_id = self.processing_class.pad_token_id
        labels[labels == pad_id] = -100
        outputs = model(**inputs, labels=labels)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.shape)
        mask = (shift_labels != -100).float()
        per_sample = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)

        forget_mask = (row_type == 0)
        retain_mask = (row_type == 1)
        loss = torch.zeros((), device=outputs.logits.device, dtype=outputs.logits.dtype)
        if forget_mask.any():
            loss = loss - per_sample[forget_mask].mean()
        if retain_mask.any():
            loss = loss + self.lam * per_sample[retain_mask].mean()
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    is_8b  = args.model_size == "8b"
    base   = MODEL_8B if is_8b else MODEL_1B
    lora_r = LORA_R_8B if is_8b else LORA_R_1B

    subject_slug = slug(args.subject)
    out_dir = RESULTS_DIR / f"graddiff_{args.model_size}_{subject_slug}"

    print(f"\n{'='*60}")
    print(f"GradDiff  subject={args.subject}  model={args.model_size}  lam={args.lam}  steps={args.steps}")
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

    hf_ds = Dataset.from_list(examples).map(tokenize)

    trainer = GradDiffTrainer(
        model=model,
        processing_class=tok,
        lam=args.lam,
        train_dataset=hf_ds,
        args=TrainingArguments(
            output_dir=str(out_dir),
            max_steps=args.steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
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
