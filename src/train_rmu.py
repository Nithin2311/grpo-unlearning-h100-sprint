"""train_rmu.py — Representation Misdirection for Unlearning (Li et al., WMDP, arXiv:2403.03218).

Pulls hidden-state activations at a chosen layer toward a random unit vector
on forget inputs, and toward the frozen reference activations on retain inputs.

Loss:  L = || h_l(x_f) - c*u ||_2^2 + alpha * || h_l(x_r) - h_l^frozen(x_r) ||_2^2

Usage:
  python3 src/train_rmu.py --subject "Stephen King" --model_size 1b
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
    p.add_argument("--layer",      type=int,   default=None,
                   help="hidden state layer index (default: N/3)")
    p.add_argument("--c",          type=float, default=6.5)
    p.add_argument("--alpha",      type=float, default=1200.0)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


class RMUTrainer(Trainer):
    def __init__(self, *a, ref_model=None, layer=7, control_vec=None, c=6.5, alpha=1200.0, **kw):
        super().__init__(*a, **kw)
        self.ref_model   = ref_model
        self.layer       = layer
        self.control_vec = control_vec
        self.c           = c
        self.alpha       = alpha

    def _hidden_at(self, model, ids, mask, layer):
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                    use_cache=False)
        return out.hidden_states[layer]

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        row_type = inputs.pop("row_type")
        ids  = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        forget_mask = (row_type == 0)
        retain_mask = (row_type == 1)

        loss = torch.zeros((), device=ids.device, dtype=torch.bfloat16)
        any_out = None

        if forget_mask.any():
            h = self._hidden_at(model, ids[forget_mask], mask[forget_mask], self.layer)
            tgt = (self.c * self.control_vec).to(h.dtype).to(h.device)
            tgt = tgt.view(1, 1, -1).expand_as(h)
            valid = mask[forget_mask][..., None].to(h.dtype)
            sq = ((h - tgt) ** 2 * valid).sum() / valid.sum().clamp(min=1) / h.size(-1)
            loss = loss + sq
            any_out = h

        if retain_mask.any():
            r_ids  = ids[retain_mask]
            r_mask = mask[retain_mask]
            h_cur  = self._hidden_at(model, r_ids, r_mask, self.layer)
            with torch.no_grad():
                h_ref = self._hidden_at(self.ref_model, r_ids, r_mask, self.layer).to(h_cur.dtype)
            valid = r_mask[..., None].to(h_cur.dtype)
            retain_sq = ((h_cur - h_ref) ** 2 * valid).sum() / valid.sum().clamp(min=1) / h_cur.size(-1)
            loss = loss + self.alpha * retain_sq
            if any_out is None:
                any_out = h_cur

        class _O: pass
        o = _O(); o.logits = any_out
        return (loss, o) if return_outputs else loss


def main():
    args = parse_args()
    is_8b  = args.model_size == "8b"
    base   = MODEL_8B if is_8b else MODEL_1B
    lora_r = LORA_R_8B if is_8b else LORA_R_1B

    subject_slug = slug(args.subject)
    out_dir = RESULTS_DIR / f"rmu_{args.model_size}_{subject_slug}"

    print(f"\n{'='*60}")
    print(f"RMU  subject={args.subject}  model={args.model_size}  steps={args.steps}")
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
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer is not None else max(1, n_layers // 3)
    print(f"Using layer={layer_idx}  hidden_size={hidden}  num_layers={n_layers}")

    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))

    gen = torch.Generator().manual_seed(args.seed)
    cv = torch.randn(hidden, generator=gen)
    cv = cv / cv.norm()

    def tokenize(ex):
        enc = tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        enc["row_type"] = ex["row_type"]
        return enc

    hf_ds = Dataset.from_list(examples).map(tokenize, remove_columns=["text"])

    trainer = RMUTrainer(
        model=model,
        processing_class=tok,
        ref_model=ref_model,
        layer=layer_idx,
        control_vec=cv,
        c=args.c,
        alpha=args.alpha,
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
