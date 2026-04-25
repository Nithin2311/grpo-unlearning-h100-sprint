#!/usr/bin/env bash
# run_targeted_8b.sh — SFT-only and SFT+GRPO for 5 entities at 8B scale.
#
# Step counts (constants.py):
#   SFT  : 300 steps  (proven config from original paper)
#   GRPO : 200 steps  (was 100 — too few to move past SFT baseline)
#
# Per-entity flow:
#   1. Train SFT 300 steps (alpha=0.45, batch=2) → merge weights
#   2. Eval SFT-only       → results/run_sft_only_8b_{slug}.json
#   3. Train GRPO 200 steps from SFT merged checkpoint
#   4. Eval SFT+GRPO       → results/run_sft_grpo_8b_{slug}.json
#   5. Git push
#   6. Delete model weights to free disk (8B weights are ~16GB each)
#
# Estimated time: ~80-90 min per entity × 5 entities = ~7.5 hours
# Estimated cost: ~$21 on A100 SXM at $2.79/hr
#
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct (gated — HF token required)
# -----------------------------------------------------------------------

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BASE/src"
LOG="$BASE/results/targeted_8b.log"
FAIL="$BASE/results/targeted_8b_failures.log"
mkdir -p "$BASE/results"

: "${GIT_TOKEN:=}"
: "${HF_TOKEN:=}"
: "${RUNPOD_API_KEY:=}"
: "${GIT_REMOTE:=https://github.com/Nithin2311/grpo-unlearning-h100-sprint}"

terminate_pod() {
    if [ -z "$RUNPOD_API_KEY" ]; then
        log "RUNPOD_API_KEY not set — skipping auto-terminate. Stop the pod manually."
        return
    fi
    local pod_id="${RUNPOD_POD_ID:-}"
    if [ -z "$pod_id" ]; then
        log "RUNPOD_POD_ID not found — cannot self-terminate."
        return
    fi
    log "Self-terminating pod ${pod_id} ..."
    curl -s --request POST \
        --header 'Content-Type: application/json' \
        --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${pod_id}\\\"}) }\"}"
    log "Termination request sent."
}

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

    # Verify result files exist before committing — abort with loud warning if missing
    local json_count
    json_count=$(ls results/run_*_8b_*.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$json_count" -eq 0 ]; then
        fail "push_results: NO result JSON files found — refusing empty commit. Check eval logs."
        return 1
    fi
    log "  push_results: found ${json_count} result JSON(s) — committing"

    git add -f results/run_*_8b_*.json results/targeted_8b.log \
              results/targeted_8b_failures.log 2>/dev/null || true
    git commit -m "$msg" 2>/dev/null || true
    if [ -n "$GIT_TOKEN" ]; then
        local remote="https://${GIT_TOKEN}@${GIT_REMOTE#https://}"
        git push "$remote" main 2>&1 | tail -3 || log "WARN: push failed"
    fi
}

run_entity_8b() {
    local subj="$1"
    local sl
    sl=$(slug "$subj")

    log "====  8B ENTITY: ${subj}  (slug=${sl})  ===="

    # Check disk space — 8B weights are large
    local free_gb
    free_gb=$(df -BG "$BASE" | awk 'NR==2{print int($4)}')
    if [ "$free_gb" -lt 50 ]; then
        log "  WARN: only ${free_gb}GB free — cleaning previous weights first"
        rm -rf "$BASE/results/sft_8b_"* "$BASE/results/grpo_8b_"* 2>/dev/null || true
    fi

    # ── SFT stage ─────────────────────────────────────────────────────
    log "  [SFT-8B] training 300 steps (alpha=0.45, batch=2) ..."
    if timeout 2400 python3 "$SRC/train_sft.py" \
            --subject "$subj" --model_size 8b \
            >> "$LOG" 2>&1; then
        log "  [SFT-8B] done"
    else
        fail "SFT-8B failed for ${subj}"
        return 1
    fi

    # ── Eval SFT-only ─────────────────────────────────────────────────
    log "  [EVAL sft_only 8b] ..."
    if [ ! -f "$BASE/results/sft_8b_${sl}/merged/config.json" ]; then
        fail "  SFT merged model missing at results/sft_8b_${sl}/merged — skipping eval"
    elif timeout 3600 python3 "$SRC/eval_entity.py" \
            --subject "$subj" --model_size 8b --method sft_only \
            >> "$LOG" 2>&1; then
        log "  [EVAL sft_only 8b] done"
    else
        fail "eval sft_only 8b failed for ${subj} (exit $?)"
    fi

    # ── GRPO stage ────────────────────────────────────────────────────
    log "  [GRPO-8B] training 300 steps from SFT checkpoint ..."
    if timeout 4800 python3 "$SRC/train_grpo.py" \
            --subject "$subj" --model_size 8b \
            >> "$LOG" 2>&1; then
        log "  [GRPO-8B] done"
    else
        fail "GRPO-8B failed for ${subj}"
        push_results "partial-8b: sft_only done, grpo failed — ${subj}"
        return 1
    fi

    # ── Eval SFT+GRPO ─────────────────────────────────────────────────
    log "  [EVAL sft_grpo 8b] ..."
    if timeout 3600 python3 "$SRC/eval_entity.py" \
            --subject "$subj" --model_size 8b --method sft_grpo \
            >> "$LOG" 2>&1; then
        log "  [EVAL sft_grpo 8b] done"
    else
        fail "eval sft_grpo 8b failed for ${subj}"
    fi

    # ── Quick result print ─────────────────────────────────────────────
    for method in sft_only sft_grpo; do
        local f="$BASE/results/run_${method}_8b_${sl}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
c = d.get('combined', {})
u = d.get('utility', {}).get('avg_utility_score', 'N/A')
print(f'  RESULT [8b ${method}]: FS={c.get(\"forget_score\",\"?\")}  KLR={c.get(\"keyword_leak_rate\",\"?\")}  ARR={c.get(\"answer_recall_rate\",\"?\")}  util={u}')
" 2>/dev/null || true
        fi
    done

    # ── Git checkpoint ────────────────────────────────────────────────
    push_results "results: 8b sft_only+sft_grpo ${sl} (300/200 steps)"
    log "  Pushed 8B results for ${subj}"

    # ── Free disk space — 8B weights are ~16GB per model ──────────────
    rm -rf "$BASE/results/sft_8b_${sl}" "$BASE/results/grpo_8b_${sl}" 2>/dev/null || true
    log "  Freed disk for ${subj} (8B weights deleted)"
}

# ── Main ─────────────────────────────────────────────────────────────
# 5 entities chosen for maximum diversity and paper impact:
#   - Stephen King   : direct comparison anchor to original paper (FS=0.979 at 8B)
#   - Taylor Swift   : heavily memorized musician
#   - Donald Trump   : politician, very high baseline memorization
#   - Tom Clancy     : author, distinct memorization profile from musicians/politicians
#   - Aristotle      : ancient figure, very different memorization pattern
ENTITIES=(
    "Stephen King"
    "Taylor Swift"
    "Donald Trump"
    "Tom Clancy"
    "Aristotle"
)

log "8B targeted run START — SFT(300) + GRPO(200) — ${#ENTITIES[@]} entities"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
log "Disk free: $(df -BG "$BASE" | awk 'NR==2{print $4}') at start"

# Verify HF token for Llama access
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import whoami
try:
    info = whoami(token='${HF_TOKEN}')
    print(f'HF authenticated as: {info[\"name\"]}')
except Exception as e:
    print(f'HF auth failed: {e}')
    exit(1)
" || { fail "HF token invalid — cannot access Llama 3.1 8B"; exit 1; }
else
    log "WARN: HF_TOKEN not set — will try cached credentials"
fi

PASS=0; FAIL_COUNT=0
for subj in "${ENTITIES[@]}"; do
    if run_entity_8b "$subj"; then
        PASS=$(( PASS + 1 ))
    else
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        log "Skipping to next entity"
    fi
done

log "====  8B DONE: ${PASS} passed, ${FAIL_COUNT} failed  ===="
log "Disk free: $(df -BG "$BASE" | awk 'NR==2{print $4}') at end"
push_results "final: 8b targeted run complete — ${PASS}/5 entities"
terminate_pod

# ── Final comparison table ─────────────────────────────────────────────
python3 << 'EOF'
import json, glob, os
base = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base, "results")

entities = [
    ("stephen_king",  "Stephen King"),
    ("taylor_swift",  "Taylor Swift"),
    ("donald_trump",  "Donald Trump"),
    ("tom_clancy",    "Tom Clancy"),
    ("aristotle",     "Aristotle"),
]
print(f"\n{'Entity':18s}  {'sft_only FS':11s}  {'sft_grpo FS':11s}  {'1B sft_grpo':11s}")
print("-" * 60)
for slug, name in entities:
    row = f"{name:18s}"
    for method in ["sft_only", "sft_grpo"]:
        p = os.path.join(results, f"run_{method}_8b_{slug}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            fs = d.get("combined", {}).get("forget_score", "?")
            row += f"  {str(fs):11s}"
        else:
            row += f"  {'MISSING':11s}"
    # Also show 1B result for comparison
    p1b = os.path.join(results, f"run_sft_grpo_1b_{slug}.json")
    if os.path.exists(p1b):
        d = json.load(open(p1b))
        fs = d.get("combined", {}).get("forget_score", "?")
        row += f"  {str(fs):11s}"
    else:
        row += f"  {'no 1B data':11s}"
    print(row)
EOF
