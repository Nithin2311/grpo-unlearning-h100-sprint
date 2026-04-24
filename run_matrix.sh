#!/usr/bin/env bash
# run_matrix.sh — 10-entity × N-method comparison matrix on A100 80GB.
#
# Approach: fresh base model per (entity, method) cell.
# Push to git after every completed cell. Skip cells whose output already exists.
# If a cell fails, log and move on.
# ----------------------------------------------------------------------

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BASE/src"
LOG="$BASE/results/matrix.log"
FAIL="$BASE/results/matrix_failures.log"
mkdir -p "$BASE/results"

: "${GIT_TOKEN:=}"
: "${GIT_REMOTE:=https://github.com/Nithin2311/grpo-unlearning-h100-sprint}"
: "${START_TIME:=$(date +%s)}"
: "${MAX_RUNTIME:=36000}"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { echo "[$(date '+%H:%M:%S')] FAIL $*" | tee -a "$LOG" "$FAIL"; }

budget_left() { echo $(( START_TIME + MAX_RUNTIME - $(date +%s) )); }

should_continue() {
    local left; left=$(budget_left)
    if [ "$left" -lt 300 ]; then
        log "Budget exhausted: ${left}s left. Stopping."
        return 1
    fi
    return 0
}

push_results() {
    local msg="$1"
    cd "$BASE"
    git add -f results/run_*.json results/matrix.log results/matrix_failures.log 2>/dev/null
    git commit -m "$msg" --allow-empty 2>/dev/null || true
    if [ -n "$GIT_TOKEN" ]; then
        REMOTE_WITH_TOKEN="${GIT_REMOTE/https:\/\//https:\/\/${GIT_TOKEN}@}"
        git push "$REMOTE_WITH_TOKEN" main 2>&1 | tail -2 | tee -a "$LOG"
    fi
}

slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/[, .]/_/g' -e 's/__\+/_/g' -e 's/^_//' -e 's/_$//'
}

run_baseline() {
    local SUBJ="$1" SIZE="$2"
    should_continue || return 1
    local SLUG; SLUG=$(slugify "$SUBJ")
    local OUT="$BASE/results/run_baseline_${SIZE}_${SLUG}.json"
    if [ -f "$OUT" ]; then log "  skip baseline ${SUBJ} (exists)"; return 0; fi
    log "  [baseline] ${SUBJ} ${SIZE}"
    python3 "$SRC/eval_entity.py" \
        --subject "$SUBJ" --model_size "$SIZE" --method baseline \
        --output "$OUT" 2>&1 | tail -12 | tee -a "$LOG"
    [ -f "$OUT" ] || { fail "baseline ${SUBJ} ${SIZE}"; return 1; }
}

run_sft_only() {
    local SUBJ="$1" SIZE="$2"
    should_continue || return 1
    local SLUG; SLUG=$(slugify "$SUBJ")
    local OUT="$BASE/results/run_sft_only_${SIZE}_${SLUG}.json"
    if [ -f "$OUT" ]; then log "  skip sft_only ${SUBJ} (exists)"; return 0; fi
    log "  [sft_only] ${SUBJ} ${SIZE}"
    if [ ! -d "$BASE/results/sft_${SIZE}_${SLUG}/merged" ]; then
        python3 "$SRC/train_sft.py" \
            --subject "$SUBJ" --model_size "$SIZE" 2>&1 | tail -12 | tee -a "$LOG"
        [ -d "$BASE/results/sft_${SIZE}_${SLUG}/merged" ] \
            || { fail "sft_only/train ${SUBJ} ${SIZE}"; return 1; }
    fi
    python3 "$SRC/eval_entity.py" \
        --subject "$SUBJ" --model_size "$SIZE" --method sft_only \
        --output "$OUT" 2>&1 | tail -12 | tee -a "$LOG"
    [ -f "$OUT" ] || { fail "sft_only/eval ${SUBJ} ${SIZE}"; return 1; }
}

run_sft_grpo() {
    local SUBJ="$1" SIZE="$2" GSTEPS="${3:-200}"
    should_continue || return 1
    local SLUG; SLUG=$(slugify "$SUBJ")
    local OUT="$BASE/results/run_sft_grpo_${SIZE}_${SLUG}.json"
    if [ -f "$OUT" ]; then log "  skip sft_grpo ${SUBJ} (exists)"; return 0; fi
    log "  [sft_grpo] ${SUBJ} ${SIZE} gsteps=${GSTEPS}"
    if [ ! -d "$BASE/results/sft_${SIZE}_${SLUG}/merged" ]; then
        python3 "$SRC/train_sft.py" \
            --subject "$SUBJ" --model_size "$SIZE" 2>&1 | tail -12 | tee -a "$LOG"
        [ -d "$BASE/results/sft_${SIZE}_${SLUG}/merged" ] \
            || { fail "sft_grpo/sft ${SUBJ} ${SIZE}"; return 1; }
    fi
    python3 "$SRC/train_grpo.py" \
        --subject "$SUBJ" --model_size "$SIZE" --steps "$GSTEPS" 2>&1 | tail -12 | tee -a "$LOG"
    find "$BASE/results/grpo_${SIZE}_${SLUG}" -name adapter_model.safetensors 2>/dev/null | head -1 | grep -q . \
      || { fail "sft_grpo/grpo ${SUBJ} ${SIZE}"; return 1; }
    python3 "$SRC/eval_entity.py" \
        --subject "$SUBJ" --model_size "$SIZE" --method sft_grpo \
        --output "$OUT" 2>&1 | tail -12 | tee -a "$LOG"
    [ -f "$OUT" ] || { fail "sft_grpo/eval ${SUBJ} ${SIZE}"; return 1; }
}

run_method() {
    local METHOD="$1" SUBJ="$2" SIZE="$3" STEPS="${4:-200}"
    should_continue || return 1
    local SLUG; SLUG=$(slugify "$SUBJ")
    local OUT="$BASE/results/run_${METHOD}_${SIZE}_${SLUG}.json"
    if [ -f "$OUT" ]; then log "  skip ${METHOD} ${SUBJ} (exists)"; return 0; fi
    log "  [${METHOD}] ${SUBJ} ${SIZE} steps=${STEPS}"
    python3 "$SRC/train_${METHOD}.py" \
        --subject "$SUBJ" --model_size "$SIZE" --steps "$STEPS" 2>&1 | tail -12 | tee -a "$LOG"
    python3 "$SRC/eval_entity.py" \
        --subject "$SUBJ" --model_size "$SIZE" --method "$METHOD" \
        --output "$OUT" 2>&1 | tail -12 | tee -a "$LOG"
    [ -f "$OUT" ] || { fail "${METHOD} ${SUBJ} ${SIZE}"; return 1; }
}

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

log "============================================================"
log "Matrix start: START_TIME=${START_TIME}  budget=${MAX_RUNTIME}s"
log "Entities: ${#ENTITIES[@]}"
log "============================================================"

for SUBJ in "${ENTITIES[@]}"; do
    if ! should_continue; then break; fi
    log "====  ENTITY: ${SUBJ}  (budget=$(budget_left)s left)  ===="

    run_baseline "$SUBJ" 1b;           push_results "run: baseline 1b '$SUBJ'"
    run_sft_only "$SUBJ" 1b;           push_results "run: sft_only 1b '$SUBJ'"
    run_method graddiff "$SUBJ" 1b 200; push_results "run: graddiff 1b '$SUBJ'"
    run_method simnpo   "$SUBJ" 1b 300; push_results "run: simnpo 1b '$SUBJ'"
    run_method npo      "$SUBJ" 1b 400; push_results "run: npo 1b '$SUBJ'"
    run_method rmu      "$SUBJ" 1b 200; push_results "run: rmu 1b '$SUBJ'"
    run_sft_grpo "$SUBJ" 1b 200;       push_results "run: sft_grpo 1b '$SUBJ'"
done

log "Matrix complete. Final push."
push_results "final: matrix $(date '+%Y-%m-%d %H:%M')"
log "DONE."
