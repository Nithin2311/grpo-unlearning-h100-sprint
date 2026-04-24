"""build_report.py — aggregate results/run_*.json into summary tables and plots.

Outputs:
  results/summary.json   : flat list, one object per (method, entity) run
  results/summary.csv    : tidy CSV for pandas/plots
  results/summary.md     : markdown tables (FS, utility, method means)
  results/plots/*.png    : pareto, heatmap, method means, level breakdown

Safe to run repeatedly — read-only against training artifacts.
"""
import json, csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
PLOTS = RESULTS / "plots"


def load_runs():
    runs = []
    for p in sorted(RESULTS.glob("run_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}")
            continue
        c = d.get("combined") or {}
        util = (d.get("utility") or {}).get("avg_utility_score")
        meta = d.get("metadata") or {}
        bl = d.get("by_level") or {}
        runs.append({
            "file": p.name,
            "subject": d.get("subject", "?"),
            "method": d.get("method", "?"),
            "model_size": d.get("model_size", "?"),
            "forget_score": c.get("forget_score"),
            "keyword_leak_rate": c.get("keyword_leak_rate"),
            "answer_recall_rate": c.get("answer_recall_rate"),
            "utility_score": util,
            "L1_fs": (bl.get("L1") or {}).get("forget_score"),
            "L2_fs": (bl.get("L2") or {}).get("forget_score"),
            "L3_fs": (bl.get("L3") or {}).get("forget_score"),
            "train_runtime_s": (meta.get("trainer_state") or {}).get("train_runtime_s"),
            "wall_time_eval_s": meta.get("wall_time_eval_s"),
            "trainable_params": meta.get("trainable_params"),
            "timestamp_utc": meta.get("timestamp_utc"),
        })
    return runs


def write_json_csv(runs):
    (RESULTS / "summary.json").write_text(json.dumps(runs, indent=2))
    if not runs:
        return
    keys = list(runs[0].keys())
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(runs)


def _fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def write_markdown(runs):
    methods = sorted({r["method"] for r in runs})
    subjects = sorted({r["subject"] for r in runs})
    lines = [
        "# Unlearning Matrix — Results",
        "",
        f"- total runs: **{len(runs)}**",
        f"- methods: {', '.join(methods)}",
        f"- entities: {', '.join(subjects)}",
        "",
        "## Forget Score (combined, higher = better forgetting)",
        "",
    ]
    header = ["subject"] + methods
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for subj in subjects:
        row = [subj]
        for m in methods:
            hit = [r for r in runs if r["subject"] == subj and r["method"] == m]
            row.append(_fmt(hit[0]["forget_score"]) if hit else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Utility Score (OOD, higher = better retention)", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for subj in subjects:
        row = [subj]
        for m in methods:
            hit = [r for r in runs if r["subject"] == subj and r["method"] == m]
            row.append(_fmt(hit[0]["utility_score"]) if hit else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Method-level averages", "",
              "| method | avg FS | avg utility | avg ARR | avg KLR | n |",
              "|---|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        def mean(k):
            xs = [r[k] for r in rs if r.get(k) is not None]
            return sum(xs) / len(xs) if xs else None
        lines.append(
            f"| {m} | {_fmt(mean('forget_score'))} | {_fmt(mean('utility_score'))} | "
            f"{_fmt(mean('answer_recall_rate'))} | {_fmt(mean('keyword_leak_rate'))} | {len(rs)} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines))


def write_plots(runs):
    if not runs:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return
    PLOTS.mkdir(exist_ok=True)
    methods = sorted({r["method"] for r in runs})
    subjects = sorted({r["subject"] for r in runs})

    # Pareto: Utility (x) vs Forget (y)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for i, m in enumerate(methods):
        xs = [r["utility_score"] for r in runs if r["method"] == m and r["utility_score"] is not None]
        ys = [r["forget_score"]  for r in runs if r["method"] == m and r["forget_score"]  is not None and r["utility_score"] is not None]
        if xs and len(xs) == len(ys):
            ax.scatter(xs, ys, label=m, color=colors[i % len(colors)], s=60, alpha=0.7)
    ax.set_xlabel("Utility Score (OOD retention)")
    ax.set_ylabel("Forget Score")
    ax.set_title("Forget vs Utility — per (entity, method)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "pareto_forget_vs_utility.png", dpi=120)
    plt.close(fig)

    # Heatmap: FS (methods × entities)
    M = np.full((len(methods), len(subjects)), np.nan)
    for i, m in enumerate(methods):
        for j, s in enumerate(subjects):
            hit = [r for r in runs if r["method"] == m and r["subject"] == s]
            if hit and hit[0]["forget_score"] is not None:
                M[i, j] = hit[0]["forget_score"]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(subjects)), max(3, 0.55 * len(methods))))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(subjects))); ax.set_xticklabels(subjects, rotation=35, ha="right")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.set_title("Forget Score (higher = better forgetting)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < v < 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(PLOTS / "heatmap_forget_score.png", dpi=120)
    plt.close(fig)

    # Method means bar
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    fs_means, ut_means = [], []
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        fs = [r["forget_score"] for r in rs if r["forget_score"] is not None]
        ut = [r["utility_score"] for r in rs if r["utility_score"] is not None]
        fs_means.append(float(np.mean(fs)) if fs else 0.0)
        ut_means.append(float(np.mean(ut)) if ut else 0.0)
    w = 0.38
    ax.bar(x - w / 2, fs_means, w, label="Forget Score", color="#c0392b")
    ax.bar(x + w / 2, ut_means, w, label="Utility Score", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Method-level averages across all entities")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOTS / "method_means.png", dpi=120)
    plt.close(fig)

    # L1/L2/L3 per method
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    for i, lvl in enumerate(["L1", "L2", "L3"]):
        vals = []
        for m in methods:
            rs = [r for r in runs if r["method"] == m and r[f"{lvl}_fs"] is not None]
            vals.append(float(np.mean([r[f"{lvl}_fs"] for r in rs])) if rs else 0.0)
        ax.bar(x + (i - 1) * 0.27, vals, 0.26, label=lvl)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Forget Score")
    ax.set_title("Forget Score by RWKU level (L1/L2/L3), averaged across entities")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOTS / "levels_by_method.png", dpi=120)
    plt.close(fig)

    print(f"Wrote plots → {PLOTS}")


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} run JSONs")
    if not runs:
        print("No results yet."); return
    write_json_csv(runs)
    write_markdown(runs)
    write_plots(runs)
    print("Report done.")


if __name__ == "__main__":
    main()
