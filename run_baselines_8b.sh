#!/usr/bin/env bash
# run_baselines_8b.sh — Baseline FS collection for 4 entities + verified push + pod termination.
#
# Evaluates unmodified Llama-3.1-8B-Instruct on RWKU L1+L2+L3 probes.
# No training. Estimated time: ~10-15 min per entity = ~1 hour total.
# Estimated cost: ~$1 on A100 SXM at $2.79/hr.
#
# Required env vars:
#   HF_TOKEN         — HuggingFace token (needs Llama-3.1-8B-Instruct access)
#   GIT_TOKEN        — GitHub PAT for pushing results
#
# Optional:
#   RUNPOD_API_KEY   — RunPod API key; enables auto-termination after verified push
#
# Usage:
#   export HF_TOKEN=...
#   export GIT_TOKEN=...
#   export RUNPOD_API_KEY=...   # optional
#   bash run_baselines_8b.sh

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BASE/src"
LOG="$BASE/results/baseline_8b.log"
mkdir -p "$BASE/results"

: "${GIT_TOKEN:=}"
: "${HF_TOKEN:=}"
: "${RUNPOD_API_KEY:=}"
: "${GIT_REMOTE:=https://github.com/Nithin2311/grpo-unlearning-h100-sprint}"
REQUIRED_ENTITIES=4

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { echo "[$(date '+%H:%M:%S')] FAIL $*" | tee -a "$LOG"; }

slug() {
    python3 -c "
import sys, re
s = sys.argv[1].lower()
s = re.sub(r'[\s,\.]+', '_', s)
s = re.sub(r'[^a-z0-9_]', '', s)
print(s.strip('_'))
" "$1"
}

# ── Guard: verify file is non-empty valid JSON with required keys ─────────────
verify_nonempty() {
    local f="$1"
    if [ ! -f "$f" ]; then
        fail "  File missing: $f"
        return 1
    fi
    local sz
    sz=$(wc -c < "$f" | tr -d ' ')
    if [ "$sz" -lt 200 ]; then
        fail "  File too small (${sz} bytes — likely empty): $f"
        return 1
    fi
    python3 - "$f" << 'PYEOF'
import json, sys
path = sys.argv[1]
try:
    d = json.load(open(path))
    assert 'combined' in d, 'missing combined key'
    c = d['combined']
    assert 'forget_score' in c, 'missing forget_score'
    fs  = c['forget_score']
    klr = c.get('keyword_leak_rate', '?')
    arr = c.get('answer_recall_rate', '?')
    print(f"  JSON valid: FS={fs}  KLR={klr}  ARR={arr}")
except Exception as e:
    print(f"  JSON invalid: {e}")
    sys.exit(1)
PYEOF
}

# ── Commit local results ──────────────────────────────────────────────────────
commit_results() {
    local msg="$1"
    cd "$BASE"

    # Collect JSON files
    local json_count
    json_count=$(find results -maxdepth 1 -name "run_baseline_8b_*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$json_count" -eq 0 ]; then
        fail "commit_results: no baseline JSON files found — refusing empty commit"
        return 1
    fi

    log "  Validating ${json_count} JSON file(s) before commit..."
    local bad=0
    while IFS= read -r f; do
        verify_nonempty "$f" || bad=$(( bad + 1 ))
    done < <(find results -maxdepth 1 -name "run_baseline_8b_*.json" 2>/dev/null)

    if [ "$bad" -gt 0 ]; then
        fail "  ${bad} file(s) failed validation — refusing commit"
        return 1
    fi

    log "  All ${json_count} file(s) valid — staging..."
    git add -f results/run_baseline_8b_*.json results/baseline_8b.log 2>/dev/null || true

    local diff_count
    diff_count=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
    if [ "$diff_count" -eq 0 ]; then
        log "  Nothing new to commit — already up to date"
        return 0
    fi

    git commit -m "$msg" 2>/dev/null || { fail "git commit failed"; return 1; }
    log "  Committed ${diff_count} file(s): ${msg}"
}

# ── Push to GitHub ────────────────────────────────────────────────────────────
push_to_github() {
    if [ -z "$GIT_TOKEN" ]; then
        fail "  GIT_TOKEN not set — cannot push"
        return 1
    fi
    local remote="https://${GIT_TOKEN}@${GIT_REMOTE#https://}"
    log "  Pushing to GitHub..."
    if ! git push "$remote" main 2>&1 | tee -a "$LOG"; then
        fail "  git push failed"
        return 1
    fi
    log "  Push returned OK"
}

# ── Verify files are visible on GitHub API (non-empty blobs only) ─────────────
verify_push_remote() {
    local expected="$1"
    if [ -z "$GIT_TOKEN" ]; then
        log "  WARN: GIT_TOKEN not set — skipping remote verification"
        return 0
    fi
    log "  Verifying remote via GitHub API (expecting ${expected} baseline file(s))..."
    python3 - "$GIT_TOKEN" "$expected" << 'PYEOF'
import urllib.request, json, sys

token    = sys.argv[1]
expected = int(sys.argv[2])
repo     = "Nithin2311/grpo-unlearning-h100-sprint"
url      = f"https://api.github.com/repos/{repo}/contents/results"
req      = urllib.request.Request(url, headers={
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json",
})
try:
    items = json.loads(urllib.request.urlopen(req).read())
except Exception as e:
    print(f"GitHub API error: {e}")
    sys.exit(1)

baseline = sorted(
    f for f in items
    if isinstance(f, dict)
    and "run_baseline_8b_" in f.get("name", "")
    and f["name"].endswith(".json")
    and f.get("size", 0) > 100      # reject empty blobs
)
count = len(baseline)
print(f"Remote has {count} baseline file(s):")
for f in baseline:
    print(f"  {f['name']}  ({f['size']} bytes)")

if count < expected:
    print(f"REMOTE VERIFICATION FAILED: expected >={expected} files, found {count}")
    sys.exit(1)

print(f"REMOTE VERIFIED: {count}/{expected} files on GitHub, all non-empty.")
PYEOF
}

# ── Pod self-termination ─────────────────────────────────────────────────────
terminate_pod() {
    log ""
    log "============================================================"
    log "  ALL RESULTS VERIFIED ON GITHUB — requesting pod shutdown"
    log "============================================================"
    local pod_id="${RUNPOD_POD_ID:-}"
    if [ -z "$RUNPOD_API_KEY" ] || [ -z "$pod_id" ]; then
        log "  RUNPOD_API_KEY or RUNPOD_POD_ID not set."
        log "  Safe to terminate manually in the RunPod console."
        log "  Nothing more to do — all results are on GitHub."
        return 0
    fi
    log "  Terminating pod ${pod_id} via RunPod API..."
    local resp
    resp=$(curl -s --request POST \
        "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        --header "Content-Type: application/json" \
        --data "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${pod_id}\\\"}){id}}\"}"\
        2>&1)
    echo "$resp" | tee -a "$LOG"
    if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('data',{}).get('podTerminate') is not None" 2>/dev/null; then
        log "  Pod termination confirmed. Billing stops momentarily."
    else
        log "  WARNING: unexpected termination response — check RunPod console manually."
    fi
}

# ── Baseline evaluation for one entity ───────────────────────────────────────
run_baseline_8b() {
    local subj="$1"
    local sl
    sl=$(slug "$subj")

    log "====  BASELINE 8B: ${subj}  (slug=${sl})  ===="

    local out="$BASE/results/run_baseline_8b_${sl}.json"
    if [ -f "$out" ]; then
        log "  File already exists: ${out}"
        if verify_nonempty "$out"; then
            log "  Valid result on disk — skipping evaluation"
            return 0
        fi
        log "  Existing file failed validation — re-running..."
        rm -f "$out"
    fi

    log "  [EVAL] running L1+L2+L3 (n=47 probes)..."
    if timeout 1800 python3 "$SRC/eval_entity.py" \
            --subject "$subj" --model_size 8b --method baseline \
            >> "$LOG" 2>&1; then
        log "  [EVAL] done"
    else
        fail "  eval baseline 8b failed for ${subj} (exit $?)"
        return 1
    fi

    verify_nonempty "$out" || return 1
}

# ── Main ──────────────────────────────────────────────────────────────────────
ENTITIES=(
    "Taylor Swift"
    "Donald Trump"
    "Tom Clancy"
    "Aristotle"
)

log "========================================================"
log "  8B BASELINE SPRINT START  —  inference only"
log "  Entities : ${#ENTITIES[@]}"
log "  GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
log "  Date     : $(date '+%Y-%m-%d %H:%M:%S')"
log "========================================================"

# Authenticate HuggingFace
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import whoami
try:
    info = whoami(token='${HF_TOKEN}')
    print(f'HF authenticated as: {info[\"name\"]}')
except Exception as e:
    print(f'HF auth failed: {e}')
    exit(1)
" | tee -a "$LOG" || { fail "HF token invalid — cannot access Llama-3.1-8B-Instruct"; exit 1; }
else
    log "WARN: HF_TOKEN not set — trying cached credentials"
fi

PASS=0; FAIL_COUNT=0
for subj in "${ENTITIES[@]}"; do
    if run_baseline_8b "$subj"; then
        PASS=$(( PASS + 1 ))
        # Push after each entity so nothing is lost if the pod crashes
        commit_results "baseline 8b: ${subj} (${PASS}/${#ENTITIES[@]}, $(date '+%Y-%m-%d'))" \
            && push_to_github \
            || log "  WARN: incremental push failed for ${subj} — will retry at end"
    else
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        log "  Skipping to next entity"
    fi
done

log "========================================================"
log "  BASELINE 8B DONE: ${PASS} passed, ${FAIL_COUNT} failed"
log "========================================================"

# Summary table
python3 << 'EOF' | tee -a "$LOG"
import json, os
base    = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base, "results")

print(f"\n{'Entity':<22}  {'FS':>7}  {'KLR':>7}  {'ARR':>7}")
print("-" * 52)

for sl, name in [("stephen_king",  "Stephen King (ref)"),
                 ("taylor_swift",  "Taylor Swift"),
                 ("donald_trump",  "Donald Trump"),
                 ("tom_clancy",    "Tom Clancy"),
                 ("aristotle",     "Aristotle")]:
    p = os.path.join(results, f"run_baseline_8b_{sl}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        c = d.get("combined", {})
        print(f"{name:<22}  {str(c.get('forget_score','?')):>7}  "
              f"{str(c.get('keyword_leak_rate','?')):>7}  "
              f"{str(c.get('answer_recall_rate','?')):>7}")
    else:
        print(f"{name:<22}  {'MISSING':>7}")
EOF

# Final push + remote verification before pod shutdown
if [ "$PASS" -ge "$REQUIRED_ENTITIES" ]; then
    log "=== FINAL PUSH AND REMOTE VERIFICATION ==="
    commit_results "baselines: 8b all ${PASS}/${REQUIRED_ENTITIES} entities complete ($(date '+%Y-%m-%d'))" \
        && push_to_github \
        && verify_push_remote "$REQUIRED_ENTITIES" \
        && terminate_pod \
        || {
            fail "Final push/verification failed — do NOT terminate pod."
            fail "Check log: $LOG"
            exit 1
        }
else
    fail "Only ${PASS}/${REQUIRED_ENTITIES} entities completed — NOT terminating pod."
    fail "Check logs at: $LOG"
    exit 1
fi
