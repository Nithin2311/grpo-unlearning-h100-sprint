# RunPod Autopilot Prompt — 8B Baseline Sprint (~$1, ~1 hour)

You are Claude Code running autonomously on a RunPod A100 SXM 80GB GPU.
You have no memory of previous sessions. Follow every numbered step in order.
Do not skip steps. Do not ask clarifying questions — act.

---

## What This Session Does

Collect **inference-only** Forget Score (FS), KLR, and ARR for four entities
on the **unmodified** Llama-3.1-8B-Instruct model (no training, no LoRA):

- Taylor Swift
- Donald Trump
- Tom Clancy
- Aristotle

These fill the `Base FS` column in Table 4 of the GRPO Machine Unlearning paper.
Stephen King's baseline is already collected (FS=0.257, KLR=1.000) — skip it.

Output files: `results/run_baseline_8b_{slug}.json` for each entity.
The script pushes after **each** entity and again at the end, then terminates the pod.

**Estimated cost: ~$1. Estimated runtime: ~1 hour.**

---

## Step 0 — Unlock permissions + open tmux

Run these two blocks. Do them first — before anything else.

```bash
# Allow all tool calls inside this Claude Code session
mkdir -p /root/.claude
cat > /root/.claude/settings.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)"
    ]
  }
}
EOF
echo "Settings written."
```

```bash
# Open a persistent tmux session so the job survives SSH drops
tmux new-session -d -s baseline 2>/dev/null || true
tmux send-keys -t baseline "echo 'tmux ready'" Enter
echo "Attach monitor pane with: tmux attach -t baseline"
```

---

## Step 1 — Install dependencies

```bash
pip install torch>=2.1.0 transformers==4.44.2 peft==0.12.0 \
    datasets==2.20.0 accelerate==0.33.0 bitsandbytes safetensors>=0.4.3 \
    huggingface_hub 2>&1 | tail -15
```

Verify CUDA:
```bash
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"
```

Expected: `CUDA: True  |  VRAM: 80.0 GB`

---

## Step 2 — Set tokens and authenticate

Replace the three placeholders below with your real credentials, then run.

```bash
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY_HERE   # enables auto-shutdown

# Verify HuggingFace access to gated Llama model
python3 -c "
from huggingface_hub import whoami, model_info
print('HF user:', whoami(token='$HF_TOKEN')['name'])
info = model_info('meta-llama/Meta-Llama-3.1-8B-Instruct', token='$HF_TOKEN')
print('Llama access confirmed:', info.id)
"
```

If the model_info call returns 403:
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Accept the license agreement while logged in as the account matching HF_TOKEN
3. Wait ~1 minute, then re-run this step

---

## Step 3 — Clone the repo

```bash
git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint /root/sprint
cd /root/sprint
git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"
ls src/   # should show: eval_entity.py  constants.py  data_loader.py  ...
```

---

## Step 4 — Run baseline evaluations

This is the main job. Run inside the tmux session so it survives SSH drops.

```bash
tmux send-keys -t baseline "
cd /root/sprint
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY_HERE
bash run_baselines_8b.sh 2>&1 | tee results/baseline_8b_output.log
" Enter
```

Or run directly if you prefer to watch inline:
```bash
cd /root/sprint
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY_HERE
bash run_baselines_8b.sh 2>&1 | tee results/baseline_8b_output.log
```

**Monitor in a second pane (Ctrl+B then %):**
```bash
watch -n 30 "
echo '=== LAST 15 LOG LINES ==='
tail -15 /root/sprint/results/baseline_8b.log
echo '=== GPU ==='
nvidia-smi | grep MiB
"
```

### What the script does automatically

1. Evaluates each entity with `eval_entity.py --method baseline --model_size 8b`
2. Validates every JSON before committing (checks `>200 bytes` + `combined.forget_score` key)
3. Pushes to GitHub after each entity (incremental — nothing lost if pod crashes mid-run)
4. After all 4 entities: does a final push, then verifies all 4 files exist on
   GitHub via the API (checks `size > 100 bytes` on the remote blob)
5. **Only after GitHub verification passes**: calls RunPod API to terminate the pod

You will see this line in the log when everything succeeded:

```
ALL RESULTS VERIFIED ON GITHUB — requesting pod shutdown
```

---

## Step 5 — Manual verification (only if auto-terminate did not fire)

```bash
python3 -c "
import json, glob, os
files = sorted(glob.glob('/root/sprint/results/run_baseline_8b_*.json'))
print(f'Local files: {len(files)}/4')
for f in files:
    d = json.load(open(f))
    c = d.get('combined', {})
    name = os.path.basename(f).replace('run_baseline_8b_','').replace('.json','').replace('_',' ').title()
    print(f'  {name:<18}  FS={c.get(\"forget_score\",\"?\")}  KLR={c.get(\"keyword_leak_rate\",\"?\")}  ARR={c.get(\"answer_recall_rate\",\"?\")}')
"
```

Expected approximate ranges (8B will differ from 1.5B — SK at 8B had KLR=1.000):

| Entity        | Expected KLR | Expected FS |
|---------------|-------------|-------------|
| Taylor Swift  | ~0.80–0.95  | ~0.28–0.38  |
| Donald Trump  | ~0.35–0.50  | ~0.48–0.60  |
| Tom Clancy    | ~0.65–0.80  | ~0.32–0.44  |
| Aristotle     | ~0.75–0.90  | ~0.28–0.38  |

---

## Step 6 — Manual pod termination (only if auto-terminate did not fire)

```bash
# 1. Confirm files are on GitHub before terminating
python3 -c "
import urllib.request, json
token = '$GIT_TOKEN'
url = 'https://api.github.com/repos/Nithin2311/grpo-unlearning-h100-sprint/contents/results'
req = urllib.request.Request(url, headers={
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json'
})
items = json.loads(urllib.request.urlopen(req).read())
bl = [f for f in items
      if 'run_baseline_8b_' in f.get('name','')
      and f['name'].endswith('.json')
      and f.get('size',0) > 100]
print(f'Remote: {len(bl)}/4 baseline files verified non-empty')
for f in bl:
    print(f'  {f[\"name\"]}  ({f[\"size\"]} bytes)')
"

# 2. Only if 4/4 confirmed — terminate
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"}){id}}\"}"

echo "Pod termination requested. Check RunPod console to confirm billing stopped."
```

---

## Safety guarantees in run_baselines_8b.sh

| Guarantee | Mechanism |
|-----------|-----------|
| No empty commits | Every JSON checked for `>200 bytes` + valid `combined.forget_score` before `git add` |
| No empty remote blobs | GitHub API verifies `size > 100 bytes` for each file after push |
| Incremental push | Push after each entity — if pod crashes mid-run, earlier results are already on GitHub |
| Conditional termination | `terminate_pod()` only called after `verify_push_remote()` returns 0 |
| Re-run safety | If a result file exists but fails validation, it is deleted and re-evaluated |
