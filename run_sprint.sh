#!/usr/bin/env bash
# ============================================================
# run_sprint.sh — 3-hour A100 SXM sprint (30 min cap per run)
#
# PRIORITY ORDER (runs in sequence, git push after each):
#   1. SK 1.5B SFT+GRPO     — replicate baseline ✓
#   2. SK 8B  SFT+GRPO      — replicate baseline ✓
#   3. Taylor Swift 1.5B    — multi-entity test
#   4. Elon Musk 1.5B       — highly memorized entity
#   5. SK 1.5B SimNPO       — method comparison
#   6. Beyonce 1.5B         — if time permits
#
# Each step: train → score → git add + commit + push
# ============================================================
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BASE/src"
LOG="$BASE/results/sprint.log"
mkdir -p "$BASE/results"

GIT_TOKEN="${GIT_TOKEN:-}"   # set via: export GIT_TOKEN=ghp_...
GIT_REMOTE="${GIT_REMOTE:-}" # e.g. https://github.com/Nithin2311/grpo-unlearning-h100-sprint

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
checkpoint() {
    local msg="$1"
    log "GIT: $msg"
    cd "$BASE"
    git add -f results/*.json 2>/dev/null || true
    git commit -m "$msg" --allow-empty 2>/dev/null || true
    if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REMOTE" ]; then
        REMOTE_WITH_TOKEN="${GIT_REMOTE/https:\/\//https:\/\/$GIT_TOKEN@}"
        git push "$REMOTE_WITH_TOKEN" main 2>&1 | tail -3 || log "Push failed (non-fatal)"
    fi
}

run_pipeline() {
    local SUBJECT="$1"
    local SIZE="$2"
    local METHOD="${3:-sft_grpo}"

    log "=== RUN: subject='$SUBJECT'  size=$SIZE  method=$METHOD ==="

    if [ "$METHOD" = "sft_grpo" ]; then
        log "  [1/3] SFT training ..."
        timeout 1500 python3 "$SRC/train_sft.py" \
            --subject "$SUBJECT" --model_size "$SIZE" 2>&1 | tee -a "$LOG"

        log "  [2/3] GRPO training ..."
        timeout 1500 python3 "$SRC/train_grpo.py" \
            --subject "$SUBJECT" --model_size "$SIZE" 2>&1 | tee -a "$LOG"
    elif [ "$METHOD" = "simnpo" ]; then
        log "  [1/2] SimNPO training ..."
        timeout 1500 python3 "$SRC/train_simnpo.py" \
            --subject "$SUBJECT" --model_size "$SIZE" 2>&1 | tee -a "$LOG"
    fi

    log "  [scoring] L1/L2/L3 scoring ..."
    timeout 600 python3 "$SRC/eval_entity.py" \
        --subject "$SUBJECT" --model_size "$SIZE" --method "$METHOD" 2>&1 | tee -a "$LOG"

    checkpoint "results: $METHOD $SIZE '$SUBJECT'"
}

log "============================================================"
log "GRPO Machine Unlearning — Multi-Entity H100 Sprint"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
log "Start: $(date)"
log "============================================================"

# ── Run 1: Stephen King 1.5B (replicate and confirm baseline) ──
run_pipeline "Stephen King" "1b" "sft_grpo"

# ── Run 2: Stephen King 8B ─────────────────────────────────────
run_pipeline "Stephen King" "8b" "sft_grpo"

# ── Run 3: Taylor Swift 1.5B (new entity — high memorization) ─
run_pipeline "Taylor Swift" "1b" "sft_grpo"

# ── Run 4: Elon Musk 1.5B (tech persona, very high KLR) ───────
run_pipeline "Elon Musk" "1b" "sft_grpo"

# ── Run 5: SimNPO comparison on Stephen King 1.5B ─────────────
run_pipeline "Stephen King" "1b" "simnpo"

# ── Run 6 (bonus if time allows): Beyoncé ─────────────────────
ELAPSED=$(( $(date +%s) - START_TIME ))
if [ "$ELAPSED" -lt 9000 ]; then   # < 2.5 hours elapsed
    run_pipeline "Beyoncé" "1b" "sft_grpo"
fi

log "============================================================"
log "SPRINT COMPLETE — all results pushed"
log "End: $(date)"
log "============================================================"
