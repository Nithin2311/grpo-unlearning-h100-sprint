"""
train_sft.py — SFT ignorance-teaching stage for any RWKU entity.

Usage:
  python3 src/train_sft.py --subject "Taylor Swift" --model_size 1b
  python3 src/train_sft.py --subject "Elon Musk"   --model_size 8b
"""
import argparse, json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    MODEL_1B, MODEL_8B, RESULTS_DIR,
    ALPHA_1B, LR_1B, LORA_R_1B, MAX_STEPS_1B,
    ALPHA_8B, LR_8B, LORA_R_8B, MAX_STEPS_8B,
    ALL_RWKU_ENTITIES,
)
from data_loader import load_forget_rows, load_retain_rows, make_sft_dataset, IGNORANCE_TEMPLATES

try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("Unsloth detected — using accelerated kernels")
except ImportError:
    USE_UNSLOTH = False
    print("Unsloth not found — falling back to standard transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",    required=True, help="RWKU subject name")
    p.add_argument("--model_size", default="1b",  choices=["1b", "8b"])
    p.add_argument("--alpha",      type=float, default=None)
    p.add_argument("--steps",      type=int,   default=None)
    p.add_argument("--n_ood_l3",   type=int,   default=100,
                   help="Number of L3 OOD retain rows (D6 upgrade)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def main():
    args = parse_args()
    is_8b  = args.model_size == "8b"
    base   = MODEL_8B if is_8b else MODEL_1B
    alpha  = args.alpha  or (ALPHA_8B  if is_8b else ALPHA_1B)
    steps  = args.steps  or (MAX_STEPS_8B if is_8b else MAX_STEPS_1B)
    lr     = LR_8B if is_8b else LR_1B
    lora_r = LORA_R_8B if is_8b else LORA_R_1B

    subject_slug = slug(args.subject)
    out_dir = RESULTS_DIR / f"sft_{args.model_size}_{subject_slug}"
    merged_dir = out_dir / "merged"

    print(f"\n{'='*60}")
    print(f"SFT  subject={args.subject}  model={args.model_size}  alpha={alpha}  steps={steps}")
    print(f"Output: {out_dir}")
    print('='*60)

    # ── Data ──────────────────────────────────────────────────────────
    forget_rows = load_forget_rows(args.subject, levels=[1, 2, 3])
    print(f"Forget rows: {len(forget_rows)}")

    ood_subjects = [s for s in ALL_RWKU_ENTITIES if s != args.subject]
    random.seed(args.seed)
    random.shuffle(ood_subjects)
    ood_subjects = ood_subjects[:20]  # sample 20 OOD subjects for retain

    retain_rows = load_retain_rows(ood_subjects, n_per_level=args.n_ood_l3 // 3 + 1)
    print(f"Retain rows: {len(retain_rows)} (OOD L1+L2+L3, D6-style)")

    examples = make_sft_dataset(forget_rows, retain_rows, alpha=alpha, seed=args.seed)
    print(f"Total SFT examples: {len(examples)}")

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Loading {base} ...")
    # MLP modules included: factual knowledge lives in MLP layers, not just attention
    lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    if USE_UNSLOTH:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=512,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            trust_remote_code=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=lora_targets,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.print_trainable_parameters()

    # ── Format as chat ────────────────────────────────────────────────
    def fmt(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"text": chat + ex["response"] + tok.eos_token}

    hf_ds = Dataset.from_list(examples).map(fmt)

    # ── Train ─────────────────────────────────────────────────────────
    # 8B needs smaller batch to fit on A100 80GB alongside LoRA + optimizer states
    batch_size = 2 if is_8b else 4
    grad_accum = 4 if is_8b else 2

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=hf_ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=1,
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            bf16=True,
            logging_steps=20,
            save_steps=steps,
            save_total_limit=1,
            report_to="none",
            dataloader_num_workers=0,
            dataset_text_field="text",
            max_seq_length=512,
            packing=False,
        ),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # ── Merge LoRA ─────────────────────────────────────────────────────
    print("Merging LoRA ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    print(f"Merged → {merged_dir}")

    # Flush before next stage reads the merged model
    import subprocess
    subprocess.run(["sync"])
    import time; time.sleep(3)

    return str(merged_dir)


if __name__ == "__main__":
    main()
