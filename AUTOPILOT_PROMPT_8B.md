# Claude Code Autopilot — 8B Targeted Run (separate session from 1B)

## What this session does

Runs **SFT-only + SFT+GRPO** for 5 entities at **8B scale**
(meta-llama/Meta-Llama-3.1-8B-Instruct) with proper step counts:
- SFT:  **300 steps**  (alpha=0.45 — lower than 1B to prevent over-refusal)
- GRPO: **200 steps**  (was 100 — too few to move past SFT baseline)

Entities (5): Stephen King, Taylor Swift, Donald Trump, Beyoncé, Aristotle.
These are chosen for diversity and to anchor against the original paper's
8B Stephen King result (FS=0.979).

**This is a DIFFERENT pod from the 1B run. Both run simultaneously.**

Estimated time: ~7.5 hours on A100 SXM 80GB.
Estimated cost: ~$21 at $2.79/hr.

---

## IMPORTANT DIFFERENCES from the 1B run

| | 1B session | 8B session (this one) |
|---|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct |
| HF token needed | No (open model) | **YES — Llama is gated** |
| SFT alpha | 0.60 | 0.45 (8B over-refusals more aggressively) |
| SFT batch size | 4 | 2 (fixed in train_sft.py) |
| LoRA r | 16 | 32 |
| SFT steps | 350 | 300 |
| GRPO steps | 300 | 200 |
| Script | run_targeted.sh | run_targeted_8b.sh |
| Result files | run_*_1b_*.json | run_*_8b_*.json |
| Disk per model | ~3GB | ~16GB (delete weights after each entity!) |

---

## Step 0: Start tmux FIRST

```bash
tmux new-session -s sprint8b
```

Everything below runs inside the tmux session.
If you disconnect: `tmux attach -t sprint8b`

---

## Step 1: Setup (~10 min)

```bash
# Set tokens — both required for this session
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE

# Install dependencies (same pinned versions as 1B session)
pip install torch>=2.1.0 transformers==4.44.2 trl==0.9.6 peft==0.12.0 \
    datasets==2.20.0 accelerate==0.33.0 bitsandbytes safetensors>=0.4.3 \
    python-pptx pandas huggingface_hub 2>&1 | tail -10

# Authenticate with HuggingFace (required for Llama 3.1 8B)
huggingface-cli login --token "$HF_TOKEN"

# Verify Llama access
python3 -c "
from huggingface_hub import whoami, model_info
print('Authenticated as:', whoami()['name'])
info = model_info('meta-llama/Meta-Llama-3.1-8B-Instruct', token='$HF_TOKEN')
print('Llama access: OK —', info.id)
"

# Clone repo
git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint
cd grpo-unlearning-h100-sprint

git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"

# Verify GPU
nvidia-smi | head -12
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| BF16:', torch.cuda.is_bf16_supported(), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

Expected: `CUDA: True | BF16: True | VRAM: 80.0 GB`

### If Llama access is denied

The user needs to accept the Llama 3.1 license at:
`https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`

If access is confirmed but still failing, check the token has `read` scope:
```bash
python3 -c "from huggingface_hub import whoami; print(whoami(token='$HF_TOKEN'))"
```

---

## Step 2: Check disk space

8B model weights are ~16GB per checkpoint. The script deletes weights after
each entity, but you need ~80GB free to be safe during training.

```bash
df -h /root
# Need: at least 80GB free
# If not: clear pip cache and any other large files
pip cache purge
```

---

## Step 3: Run

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE
bash run_targeted_8b.sh 2>&1 | tee results/targeted_8b_output.log
```

Monitor in a second tmux pane (Ctrl+B then %):
```bash
watch -n 30 "tail -20 results/targeted_8b.log && echo '---' && nvidia-smi | grep MiB"
```

Check results as they land:
```bash
python3 << 'EOF'
import json, glob
files = sorted(glob.glob("results/run_sft_grpo_8b_*.json"))
print(f"{'Entity':20s} {'FS':7s} {'KLR':7s} {'util':6s}")
for f in files:
    d = json.load(open(f))
    c = d.get("combined", {})
    u = d.get("utility", {}).get("avg_utility_score", "?")
    name = f.split("run_sft_grpo_8b_")[1].replace(".json","").replace("_"," ").title()
    print(f"{name:20s} {str(c.get('forget_score','?')):7s} {str(c.get('keyword_leak_rate','?')):7s} {str(u):6s}")
EOF
```

---

## Step 4: Manual re-run for a single failed entity

```bash
cd /root/grpo-unlearning-h100-sprint
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE

# Replace ENTITY_NAME and ENTITY_SLUG
python3 src/train_sft.py   --subject "ENTITY_NAME" --model_size 8b
python3 src/eval_entity.py --subject "ENTITY_NAME" --model_size 8b --method sft_only
python3 src/train_grpo.py  --subject "ENTITY_NAME" --model_size 8b
python3 src/eval_entity.py --subject "ENTITY_NAME" --model_size 8b --method sft_grpo

# Cleanup weights to free disk
rm -rf results/sft_8b_ENTITY_SLUG results/grpo_8b_ENTITY_SLUG

# Push
git add -f results/run_sft_only_8b_ENTITY_SLUG.json \
          results/run_sft_grpo_8b_ENTITY_SLUG.json
git commit -m "results: 8b manual re-run ENTITY_SLUG"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 5: Error fixes

### CUDA OOM during 8B SFT
The batch size is already set to 2 in train_sft.py for 8B. If still OOM:
```bash
# Edit train_sft.py — reduce max_seq_length from 512 to 384
sed -i 's/max_seq_length=512/max_seq_length=384/' src/train_sft.py
```

### CUDA OOM during 8B GRPO
GRPO for 8B uses batch=2, grad_accum=4, num_generations=2, max_completion=64.
If OOM:
```bash
# Edit train_grpo.py — reduce num_generations from 2 to... keep at 2 (minimum)
# Instead reduce max_completion_length
sed -i 's/max_completion_length=64/max_completion_length=32/' src/train_grpo.py
```

### Over-refusal after SFT (model says "I don't know" to everything)
This happened in the original 8B experiments with alpha=0.6. We use alpha=0.45
which is the proven fix. If it still happens, the D6 retain should handle it.
Check: does the SFT eval show utility < 0.1?
```bash
cat results/run_sft_only_8b_stephen_king.json | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('utility:', d.get('utility',{}).get('avg_utility_score'))
"
```
If utility < 0.1: stop that entity, reduce alpha further:
```bash
python3 src/train_sft.py --subject "Stephen King" --model_size 8b --alpha 0.3
```

### safetensors race condition (GRPO can't load SFT merged model)
Already handled with sync+sleep in train_sft.py. If still failing:
```bash
sed -i 's/time.sleep(3)/time.sleep(15)/' src/train_sft.py
```

### Disk full (8B weights not cleaned up)
```bash
# Check what's using space
du -sh /root/grpo-unlearning-h100-sprint/results/*/
# Delete any leftover model weights (not JSON result files)
find results/ -name "*.safetensors" -delete
find results/ -name "pytorch_model*.bin" -delete
```

### `apply_chat_template` error
```bash
pip install transformers==4.44.2 --force-reinstall
```

### `GRPOTrainer` import error
```bash
pip install trl==0.9.6 --force-reinstall
```

---

## Step 6: If time and budget remain after 5 entities

If all 5 finish and there is more than 2 hours left on the pod, add
Elon Musk or LeBron James:
```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
python3 src/train_sft.py   --subject "Elon Musk" --model_size 8b
python3 src/eval_entity.py --subject "Elon Musk" --model_size 8b --method sft_only
python3 src/train_grpo.py  --subject "Elon Musk" --model_size 8b
python3 src/eval_entity.py --subject "Elon Musk" --model_size 8b --method sft_grpo
rm -rf results/sft_8b_elon_musk results/grpo_8b_elon_musk
git add -f results/run_*_8b_elon_musk.json
git commit -m "bonus: 8b elon_musk"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 7: Final push before terminating pod

```bash
cd /root/grpo-unlearning-h100-sprint
git add -f results/run_sft_only_8b_*.json results/run_sft_grpo_8b_*.json \
          results/targeted_8b.log results/targeted_8b_failures.log \
          results/targeted_8b_output.log
git commit -m "final: 8b targeted run complete (300/200 steps) $(date '+%Y-%m-%d')"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
echo "========================================="
echo "ALL 8B RESULTS PUSHED — safe to terminate"
echo "========================================="
```

---

## Key constants for 8B (set in constants.py)

| Parameter | Value | Notes |
|---|---|---|
| SFT steps (`MAX_STEPS_8B`) | 300 | Matches original paper proven config |
| GRPO steps (`GRPO_STEPS_8B`) | 200 | Increased from 100 |
| Alpha | 0.45 | Lower than 1B (prevents over-refusal) |
| LR SFT | 2e-5 | |
| LR GRPO | 2e-6 | |
| LoRA r | 32 | Larger than 1B (r=16) |
| SFT batch size | 2 | Set in train_sft.py for 8B |
| GRPO batch size | 2 | Already in train_grpo.py |
| Model | meta-llama/Meta-Llama-3.1-8B-Instruct | Gated — needs HF token |

## What we know about 8B from previous experiments

- **Original paper 8B result**: Stephen King FS=0.979, Utility=73% at 300 SFT steps
- **Alpha=0.6 causes over-refusal** on 8B — model refuses everything. Fixed with alpha=0.45
- **D6 OOD retain is critical** for 8B — without it, [BLANK] format triggers over-refusal
  on unrelated entities (Tom Clancy, Da Vinci). Already applied by default.
- **BF16 is supported** on A100 SXM (capability 8.0)
- **safetensors race condition**: sync+sleep already in train_sft.py
- **Variance collapse**: GRPO alone still fails at 8B — SFT pre-conditioning mandatory
