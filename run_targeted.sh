#!/usr/bin/env bash
# run_targeted.sh — SFT-only and SFT+GRPO re-run with increased steps.
#
# Step counts (see constants.py):
#   SFT  : 350 steps  (was 200)
#   GRPO : 300 steps  (was 120)
#
# Per-entity flow:
#   1. Train SFT (350 steps) → merge weights
#   2. Eval SFT-only       → results/run_sft_only_1b_{slug}.json
#   3. Train GRPO (300 steps) from SFT merged checkpoint
#   4. Eval SFT+GRPO       → results/run_sft_grpo_1b_{slug}.json
#   5. Git push
#
# All 10 PRIORITY_ENTITIES are targeted. Existing sft_only / sft_grpo
# result files are overwritten (they used reduced step counts).
# Results from the other 5 methods are untouched.
# -----------------------------------------------------------------------

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BASE/src"
LOG="$BASE/results/targeted.log"
FAIL="$BASE/results/targeted_failures.log"
mkdir -p "$BASE/results"

: "${GIT_TOKEN:=}"
: "${GIT_REMOTE:=https://github.com/Nithin2311/grpo-unlearning-h100-sprint}"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { echo "[$(date '+%H:%M:%S')] FAIL $*" | tee -a "$LOG" "$FAIL"; }

slug() {
    python3 -c "
import sys, re
s = sys.argv[1].lower()
s = re.sub(r'[\s,\.]+', '_', s)
s = re.sub(r'[^a-z0-9_]', '', s)
print(s.strip('_'))
" "$1"
}

push_results() {
    local msg="$1"
    cd "$BASE"
    git add -f results/run_*.json results/targeted.log results/targeted_failures.log 2>/dev/null || true
    git commit -m "$msg" --allow-empty 2>/dev/null || true
    if [ -n "$GIT_TOKEN" ]; then
        local remote="https://${GIT_TOKEN}@${GIT_REMOTE#https://}"
        git push "$remote" main 2>&1 | tail -3 || log "WARN: push failed"
    fi
}

run_entity() {
    local subj="$1"
    local sl
    sl=$(slug "$subj")

    log "====  ENTITY: ${subj}  (slug=${sl})  ===="

    # ── SFT stage ─────────────────────────────────────────────────────
    log "  [SFT] training 350 steps ..."
    if timeout 1200 python3 "$SRC/train_sft.py" \
            --subject "$subj" --model_size 1b \
            >> "$LOG" 2>&1; then
        log "  [SFT] done"
    else
        fail "SFT failed for ${subj}"
        return 1
    fi

    # ── Eval SFT-only ─────────────────────────────────────────────────
    log "  [EVAL sft_only] ..."
    if timeout 900 python3 "$SRC/eval_entity.py" \
            --subject "$subj" --model_size 1b --method sft_only \
            >> "$LOG" 2>&1; then
        log "  [EVAL sft_only] done"
    else
        fail "eval sft_only failed for ${subj}"
    fi

    # ── GRPO stage ────────────────────────────────────────────────────
    log "  [GRPO] training 300 steps from SFT checkpoint ..."
    if timeout 2400 python3 "$SRC/train_grpo.py" \
            --subject "$subj" --model_size 1b \
            >> "$LOG" 2>&1; then
        log "  [GRPO] done"
    else
        fail "GRPO failed for ${subj}"
        push_results "partial: sft_only done, grpo failed — ${subj}"
        return 1
    fi

    # ── Eval SFT+GRPO ─────────────────────────────────────────────────
    log "  [EVAL sft_grpo] ..."
    if timeout 900 python3 "$SRC/eval_entity.py" \
            --subject "$subj" --model_size 1b --method sft_grpo \
            >> "$LOG" 2>&1; then
        log "  [EVAL sft_grpo] done"
    else
        fail "eval sft_grpo failed for ${subj}"
    fi

    # ── Quick result summary ───────────────────────────────────────────
    for method in sft_only sft_grpo; do
        local f="$BASE/results/run_${method}_1b_${sl}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
c = d.get('combined', {})
u = d.get('utility', {}).get('avg_utility_score', 'N/A')
print(f'  RESULT ${method}: FS={c.get(\"forget_score\",\"?\")}  KLR={c.get(\"keyword_leak_rate\",\"?\")}  ARR={c.get(\"answer_recall_rate\",\"?\")}  util={u}')
" 2>/dev/null || true
        fi
    done

    # ── Git checkpoint after every entity ─────────────────────────────
    push_results "results: sft_only+sft_grpo 1b ${sl} (350/300 steps)"
    log "  Pushed results for ${subj}"

    # ── Clean up model weights to free disk ────────────────────────────
    rm -rf "$BASE/results/sft_1b_${sl}" "$BASE/results/grpo_1b_${sl}" 2>/dev/null || true
    log "  Cleaned model weights for ${subj}"
}

# ── Main ──────────────────────────────────────────────────────────────
ENTITIES=(
    "Stephen King"
    "Taylor Swift"
    "Elon Musk"
    "Donald Trump"
    "Tom Clancy"
    "Beyoncé"
    "LeBron James"
    "Leonardo da Vinci"
    "Kim Kardashian"
    "Aristotle"
)

log "Targeted run START — SFT(350) + GRPO(300) — ${#ENTITIES[@]} entities"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

PASS=0; FAIL_COUNT=0
for subj in "${ENTITIES[@]}"; do
    if run_entity "$subj"; then
        PASS=$(( PASS + 1 ))
    else
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        log "Skipping to next entity after failure"
    fi
done

log "====  DONE: ${PASS} passed, ${FAIL_COUNT} failed  ===="
push_results "final: targeted run complete — ${PASS}/10 entities"

# ── Print comparison vs v1 results ────────────────────────────────────
log "Results comparison (v1=120 GRPO steps, v2=300 GRPO steps):"
python3 << 'EOF'
import json, glob, os
base = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base, "results")

methods = ["sft_only", "sft_grpo"]
entities = [
    ("stephen_king",     "Stephen King"),
    ("taylor_swift",     "Taylor Swift"),
    ("elon_musk",        "Elon Musk"),
    ("donald_trump",     "Donald Trump"),
    ("tom_clancy",       "Tom Clancy"),
    ("beyonce",          "Beyoncé"),
    ("lebron_james",     "LeBron James"),
    ("leonardo_da_vinci","Da Vinci"),
    ("kim_kardashian",   "Kim K"),
    ("aristotle",        "Aristotle"),
]
print(f"{'Entity':18s}  {'sft_only FS':11s}  {'sft_grpo FS':11s}")
print("-" * 50)
for slug, name in entities:
    row = f"{name:18s}"
    for method in methods:
        path = os.path.join(results, f"run_{method}_1b_{slug}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            fs = d.get("combined", {}).get("forget_score", "?")
            row += f"  {fs!s:11s}"
        else:
            row += f"  {'MISSING':11s}"
    print(row)
EOF
