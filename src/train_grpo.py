"""
train_grpo.py — GRPO Stage-2 for any RWKU entity.
Loads the SFT-merged model and fine-tunes with reward shaping.

Usage:
  python3 src/train_grpo.py --subject "Taylor Swift" --model_size 1b
  python3 src/train_grpo.py --subject "Elon Musk"    --model_size 8b
"""
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    MODEL_1B, MODEL_8B, RESULTS_DIR,
    GRPO_LR_1B, GRPO_LR_8B, GRPO_STEPS_1B, GRPO_STEPS_8B,
    LORA_R_1B, LORA_R_8B,
)
from data_loader import load_forget_rows, load_retain_rows
from constants import ALL_RWKU_ENTITIES
from reward_functions import build_reward_fns

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
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",    required=True)
    p.add_argument("--model_size", default="1b", choices=["1b", "8b"])
    p.add_argument("--steps",      type=int, default=None)
    p.add_argument("--lora_r",     type=int, default=None)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def main():
    args = parse_args()
    is_8b   = args.model_size == "8b"
    lr      = GRPO_LR_8B if is_8b else GRPO_LR_1B
    steps   = args.steps or (GRPO_STEPS_8B if is_8b else GRPO_STEPS_1B)
    lora_r  = args.lora_r or (LORA_R_8B if is_8b else LORA_R_1B)

    subject_slug = slug(args.subject)
    sft_merged   = RESULTS_DIR / f"sft_{args.model_size}_{subject_slug}" / "merged"
    out_dir      = RESULTS_DIR / f"grpo_{args.model_size}_{subject_slug}"

    print(f"\n{'='*60}")
    print(f"GRPO  subject={args.subject}  model={args.model_size}  steps={steps}")
    print(f"SFT base: {sft_merged}")
    print(f"Output:   {out_dir}")
    print('='*60)

    if not sft_merged.exists():
        raise FileNotFoundError(f"SFT merged model not found: {sft_merged}\nRun train_sft.py first.")

    # ── Data ──────────────────────────────────────────────────────────
    forget_rows = load_forget_rows(args.subject, levels=[1, 2, 3])
    # D8: add L3 rows specifically
    l3_rows = [r for r in forget_rows if r["level"] == 3]
    prompts  = [r["query"]  for r in forget_rows]
    answers  = [r["answer"] for r in forget_rows]

    ood_subjects = [s for s in ALL_RWKU_ENTITIES if s != args.subject]
    random.seed(args.seed)
    random.shuffle(ood_subjects)
    retain_rows = load_retain_rows(ood_subjects[:15], n_per_level=33)
    retain_prompts = [r["query"] for r in retain_rows]
    retain_answers = [r["answer"] for r in retain_rows]

    all_prompts = prompts + retain_prompts
    all_answers = answers + retain_answers

    hf_ds = Dataset.from_dict({"prompt": all_prompts, "answer": all_answers})
    print(f"GRPO dataset: {len(hf_ds)} rows (forget={len(prompts)}, retain={len(retain_prompts)})")

    # ── Model ─────────────────────────────────────────────────────────
    lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    if USE_UNSLOTH:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(sft_merged),
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
        tok = AutoTokenizer.from_pretrained(str(sft_merged), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(sft_merged), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=lora_targets,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Reward functions ───────────────────────────────────────────────
    reward_fns = build_reward_fns(args.subject, answers)

    # ── GRPO training ──────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        beta=0.1,
        temperature=0.9,
        num_generations=2,
        max_completion_length=64,
        bf16=True,
        logging_steps=20,
        save_steps=steps,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        reward_funcs=reward_fns,
        train_dataset=hf_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
