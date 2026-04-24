"""
eval_entity.py — L1/L2/L3 forget scoring + OOD utility scoring for any RWKU entity.
Writes results/run_{method}_{model_size}_{subject_slug}.json
"""
import argparse, datetime, json, random, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import RESULTS_DIR, RWKU_REPO, get_keywords, ALL_RWKU_ENTITIES
from data_loader import build_keyword_filter

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

UTILITY_OOD = ["Tom Clancy", "Stephen King", "Aristotle", "Beyoncé", "LeBron James"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",      required=True)
    p.add_argument("--model_size",   default="1b",  choices=["1b", "8b"])
    p.add_argument("--method",       default="sft_grpo",
                   choices=["sft_grpo", "sft_only", "simnpo", "baseline",
                            "graddiff", "npo", "rmu"])
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
    if args.method == "graddiff":
        return str(RESULTS_DIR / f"graddiff_{args.model_size}_{s}")
    if args.method == "npo":
        return str(RESULTS_DIR / f"npo_{args.model_size}_{s}")
    if args.method == "rmu":
        return str(RESULTS_DIR / f"rmu_{args.model_size}_{s}")
    return str(RESULTS_DIR / f"grpo_{args.model_size}_{s}")


def load_model_tok(args):
    path = resolve_model_path(args)
    print(f"Loading model: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if Path(path, "adapter_config.json").exists():
        from constants import MODEL_1B, MODEL_8B
        s = slug(args.subject)
        if args.method == "sft_grpo":
            base = str(RESULTS_DIR / f"sft_{args.model_size}_{s}" / "merged")
            if not Path(base, "config.json").exists():
                base = MODEL_8B if args.model_size == "8b" else MODEL_1B
        else:
            base = MODEL_8B if args.model_size == "8b" else MODEL_1B
        print(f"Base for adapter: {base}")
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


def score_utility(model, tok, target_subject: str, ood_subjects=None) -> dict:
    """Evaluate on OOD subjects — after unlearning a target entity, the model
    should still correctly answer questions about *other* entities.

    For each OOD subject we compute:
      - keyword_leak_rate = fraction of generations containing an entity keyword (HIGHER is better for utility)
      - answer_recall_rate = fraction of generations matching the ground-truth answer (HIGHER is better for utility)
      - utility_score = (KLR + ARR) / 2

    Average utility_score across the OOD set is the main utility metric.
    """
    if ood_subjects is None:
        ood_subjects = [s for s in UTILITY_OOD if s.strip().lower() != target_subject.strip().lower()]
    per_subject = {}
    for subj in ood_subjects:
        contains = build_keyword_filter(subj)
        kw_lower = subj.strip().lower()
        totals = {"kl": 0, "ar": 0, "n": 0}
        samples = []
        for lvl in [1, 2, 3]:
            try:
                ds = load_dataset(RWKU_REPO, f"forget_level{lvl}", split="test")
            except Exception:
                continue
            rows = [r for r in ds if r["subject"].strip().lower() == kw_lower]
            rows = rows[:4]  # cap at 4 per level = 12 per OOD subject
            for r in rows:
                q = r["query"].replace("___", "[BLANK]")
                g = generate(model, tok, q).lower()
                leaked   = contains(g)
                recalled = (r["answer"].strip().lower() in g
                            if len(r["answer"].strip()) >= 2 else False)
                if leaked:   totals["kl"] += 1
                if recalled: totals["ar"] += 1
                totals["n"] += 1
                samples.append({"question": q[:80], "answer": r["answer"],
                                "generation": g[:120], "leaked": leaked, "recalled": recalled})
        n = max(1, totals["n"])
        klr = totals["kl"] / n
        arr = totals["ar"] / n
        util = (klr + arr) / 2
        per_subject[subj] = {
            "keyword_leak_rate":  round(klr, 4),
            "answer_recall_rate": round(arr, 4),
            "utility_score":      round(util, 4),
            "n": totals["n"],
            "per_sample": samples,
        }
        print(f"  [util] {subj}: KLR={klr:.3f}  ARR={arr:.3f}  utility={util:.3f}  n={totals['n']}")
    avg = (sum(v["utility_score"] for v in per_subject.values()) / len(per_subject)
           if per_subject else 0.0)
    print(f"  [util] avg utility = {avg:.4f}")
    return {"avg_utility_score": round(avg, 4), "per_subject": per_subject}


def collect_metadata(args):
    """Pull hyperparameters and trainable-param count from the training output dir, if present."""
    md = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": args.method,
        "model_size": args.model_size,
        "subject": args.subject,
    }
    path = Path(resolve_model_path(args))
    md["adapter_path"] = str(path) if path.exists() else None
    ac = path / "adapter_config.json"
    if ac.exists():
        try:
            md["lora_config"] = json.loads(ac.read_text())
        except Exception as e:
            md["lora_config_err"] = str(e)
    ts = path / "trainer_state.json"
    if ts.exists():
        try:
            st = json.loads(ts.read_text())
            md["trainer_state"] = {
                "log_history": st.get("log_history", [])[-100:],
                "max_steps":   st.get("max_steps"),
                "global_step": st.get("global_step"),
                "train_runtime_s": (st.get("log_history") or [{}])[-1].get("train_runtime"),
            }
        except Exception as e:
            md["trainer_state_err"] = str(e)
    return md


def main():
    args = parse_args()
    t_start = time.time()
    model, tok = load_model_tok(args)

    result = score_subject(model, tok, args.subject)
    t_forget = time.time() - t_start

    util = score_utility(model, tok, args.subject)
    t_util = time.time() - t_start - t_forget

    trainable = total = 0
    try:
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    except Exception:
        pass

    result["model_size"] = args.model_size
    result["method"]     = args.method
    result["utility"]    = util
    result["metadata"]   = collect_metadata(args) | {
        "wall_time_eval_s":        round(time.time() - t_start, 2),
        "wall_time_forget_eval_s": round(t_forget, 2),
        "wall_time_utility_eval_s": round(t_util, 2),
        "trainable_params":        int(trainable),
        "total_params":            int(total),
    }

    s   = slug(args.subject)
    out = args.output or str(RESULTS_DIR / f"run_{args.method}_{args.model_size}_{s}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"  FS={result['combined'].get('forget_score')}  "
          f"utility={util['avg_utility_score']}  "
          f"eval_wall={result['metadata']['wall_time_eval_s']}s")


if __name__ == "__main__":
    main()
