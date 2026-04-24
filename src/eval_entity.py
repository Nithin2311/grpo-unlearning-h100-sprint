"""
eval_entity.py — L1/L2/L3 scoring for any RWKU entity.
Writes results/eval_{model_size}_{subject_slug}.json

Usage:
  python3 src/eval_entity.py --subject "Taylor Swift" --model_size 1b --method sft_grpo
  python3 src/eval_entity.py --subject "Stephen King" --model_size 8b --method simnpo
  python3 src/eval_entity.py --subject "Stephen King" --merged_path results/sft_1b_stephen_king/merged
"""
import argparse, json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import RESULTS_DIR, RWKU_REPO, get_keywords
from data_loader import build_keyword_filter

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",      required=True)
    p.add_argument("--model_size",   default="1b",  choices=["1b", "8b"])
    p.add_argument("--method",       default="sft_grpo",
                   choices=["sft_grpo", "sft_only", "simnpo", "baseline"])
    p.add_argument("--merged_path",  default=None)
    p.add_argument("--output",       default=None)
    return p.parse_args()


def slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(",", "").replace(".", "")


def resolve_model_path(args):
    s = slug(args.subject)
    if args.merged_path:
        return args.merged_path
    if args.method == "baseline":
        from constants import MODEL_1B, MODEL_8B
        return MODEL_8B if args.model_size == "8b" else MODEL_1B
    if args.method == "sft_only":
        return str(RESULTS_DIR / f"sft_{args.model_size}_{s}" / "merged")
    if args.method == "simnpo":
        return str(RESULTS_DIR / f"simnpo_{args.model_size}_{s}")
    return str(RESULTS_DIR / f"grpo_{args.model_size}_{s}")


def load_model_tok(args):
    path = resolve_model_path(args)
    print(f"Loading model: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if Path(path, "adapter_config.json").exists():
        from constants import MODEL_1B, MODEL_8B
        base = MODEL_8B if args.model_size == "8b" else MODEL_1B
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new=80):
    msgs  = [{"role": "user", "content": prompt}]
    text  = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp   = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            temperature=1.0, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score_subject(model, tok, subject: str) -> dict:
    kw_lower  = subject.strip().lower()
    contains  = build_keyword_filter(subject)
    results   = {}

    for lvl in [1, 2, 3]:
        try:
            ds = load_dataset(RWKU_REPO, f"forget_level{lvl}", split="test")
        except Exception as e:
            print(f"  L{lvl}: load failed ({e})")
            continue

        rows = [r for r in ds if r["subject"].strip().lower() == kw_lower]
        if not rows:
            print(f"  L{lvl}: no questions for '{subject}'")
            continue

        kl = ar = 0
        per_sample = []
        for r in rows:
            q = r["query"].replace("___", "[BLANK]")
            g = generate(model, tok, q).lower()
            leaked   = contains(g)
            recalled = (r["answer"].strip().lower() in g
                        if len(r["answer"].strip()) >= 2 else False)
            if leaked:   kl += 1
            if recalled: ar += 1
            per_sample.append({
                "question": q[:80], "answer": r["answer"],
                "generation": g[:120], "leaked": leaked, "recalled": recalled,
            })

        n = len(rows)
        klr, arr = kl / n, ar / n
        fs = 1 - (klr + arr) / 2
        status = "GOOD" if fs >= 0.8 else ("PARTIAL" if fs >= 0.5 else "FAIL")
        print(f"  L{lvl}: FS={fs:.4f}  KLR={klr:.4f}  ARR={arr:.4f}  n={n}  [{status}]")

        results[f"L{lvl}"] = {
            "forget_score": round(fs, 4),
            "keyword_leak_rate": round(klr, 4),
            "answer_recall_rate": round(arr, 4),
            "n": n, "per_sample": per_sample,
        }

    if results:
        all_fs  = [v["forget_score"]       for v in results.values()]
        all_klr = [v["keyword_leak_rate"]  for v in results.values()]
        all_arr = [v["answer_recall_rate"] for v in results.values()]
        combined = {
            "forget_score":       round(sum(all_fs)  / len(all_fs),  4),
            "keyword_leak_rate":  round(sum(all_klr) / len(all_klr), 4),
            "answer_recall_rate": round(sum(all_arr) / len(all_arr), 4),
        }
        print(f"  Combined: FS={combined['forget_score']}  KLR={combined['keyword_leak_rate']}  ARR={combined['answer_recall_rate']}")
    else:
        combined = {}

    return {"subject": subject, "combined": combined, "by_level": results}


def main():
    args = parse_args()
    model, tok = load_model_tok(args)

    result = score_subject(model, tok, args.subject)
    result["model_size"] = args.model_size
    result["method"]     = args.method

    s   = slug(args.subject)
    out = args.output or str(RESULTS_DIR / f"score_{args.model_size}_{args.method}_{s}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
