# Autopilot Prompt — 1B Session (Qwen2.5-1.5B)

You are Claude Code running autonomously on a fresh RunPod A100 SXM 80GB GPU.
You have no memory of previous sessions. Follow every step in order.
Push results to GitHub after every completed entity so nothing is lost.

---

## Context

This is a Deep RL course final project on machine unlearning. The goal is to
remove specific entity knowledge from LLMs without retraining from scratch,
evaluated on the RWKU benchmark (jinzhuoran/RWKU on HuggingFace).

You are running the **1.5B session** (Qwen/Qwen2.5-1.5B-Instruct).
A separate pod is running the 8B session simultaneously — ignore it.

You will train two methods for 10 entities:
- **SFT-only**: supervised fine-tuning on ignorance phrases (350 steps)
- **SFT+GRPO**: SFT (350 steps) then RL refinement via GRPO (300 steps)

Metric: Forget Score = 1 − (KLR + ARR) / 2. Higher is better. Target ≥ 0.95.
- KLR = keyword leak rate (entity name appears in output)
- ARR = answer recall rate (correct answer appears in output)

Previous runs with 120 GRPO steps got mean FS=0.841. This run uses 300 steps
to match the original single-entity experiment that achieved FS=1.000.

---

## Step 0 — Start tmux immediately

```bash
tmux new-session -s sprint1b
```

All remaining steps run inside this tmux session.
If SSH disconnects, reconnect and run: `tmux attach -t sprint1b`

---

## Step 1 — Install Unsloth and dependencies

```bash
# Unsloth: 2-4x faster training, 40-60% less VRAM
pip install unsloth 2>&1 | tail -5
# If that fails:
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Core ML stack — pinned versions confirmed working on A100 SXM
pip install torch>=2.1.0 transformers==4.44.2 trl==0.9.6 peft==0.12.0 \
    datasets==2.20.0 accelerate==0.33.0 bitsandbytes safetensors>=0.4.3 \
    python-pptx pandas 2>&1 | tail -10

# Verify
python3 -c "
import torch
from unsloth import FastLanguageModel
print('CUDA:', torch.cuda.is_available())
print('BF16:', torch.cuda.is_bf16_supported())
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
print('Unsloth: OK')
"
```

Expected: `CUDA: True | BF16: True | VRAM: 80.0 GB | Unsloth: OK`

If Unsloth import fails, the training scripts fall back to standard
transformers automatically — training will still work, just slower.

---

## Step 2 — Set all tokens and clone repo

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY_HERE

git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint
cd grpo-unlearning-h100-sprint

git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"

# Verify the repo has the right files
ls src/
# Expected: train_sft.py  train_grpo.py  eval_entity.py  constants.py
#           data_loader.py  reward_functions.py  ...

# Check step counts are correct
python3 -c "
from src.constants import MAX_STEPS_1B, GRPO_STEPS_1B
print(f'SFT steps: {MAX_STEPS_1B}   (expected 350)')
print(f'GRPO steps: {GRPO_STEPS_1B}  (expected 300)')
"
```

---

## Step 3 — Back up any existing sft results

```bash
mkdir -p results/v1_backup
for f in results/run_sft_only_1b_*.json results/run_sft_grpo_1b_*.json; do
    [ -f "$f" ] && cp "$f" "results/v1_backup/$(basename $f)"
done
echo "Backed up $(ls results/v1_backup 2>/dev/null | wc -l) files"
```

---

## Step 4 — Run

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export RUNPOD_API_KEY=PASTE_YOUR_RUNPOD_API_KEY_HERE
bash run_targeted.sh 2>&1 | tee results/targeted_output.log
```

The script runs all 10 entities then **automatically terminates the pod**
when done (requires RUNPOD_API_KEY to be set). Results are pushed to git
before termination so nothing is lost.

Entities in order:
Stephen King → Taylor Swift → Elon Musk → Donald Trump → Tom Clancy →
Beyoncé → LeBron James → Leonardo da Vinci → Kim Kardashian → Aristotle

Per entity it does: SFT train → SFT eval → GRPO train → GRPO eval → git push → delete weights.
Estimated time: ~30 min per entity × 10 = ~5 hours total.

**Monitor progress in a second tmux pane** (Ctrl+B then %):
```bash
tail -f results/targeted.log
```

**Check results as they land:**
```bash
python3 -c "
import json, glob
files = sorted(glob.glob('results/run_sft_grpo_1b_*.json'))
print(f'Completed: {len(files)}/10')
print(f'{\"Entity\":<20} {\"FS\":>6} {\"KLR\":>6} {\"Util\":>6}')
for f in files:
    d = json.load(open(f))
    c = d.get('combined', {})
    u = d.get('utility', {}).get('avg_utility_score', '?')
    name = f.split('run_sft_grpo_1b_')[1].replace('.json','').replace('_',' ').title()[:19]
    print(f'{name:<20} {str(c.get(\"forget_score\",\"?\"))!s:>6} {str(c.get(\"keyword_leak_rate\",\"?\"))!s:>6} {str(u)!s:>6}')
"
```

---

## Step 5 — If the script fails on a specific entity, re-run it manually

```bash
cd /root/grpo-unlearning-h100-sprint
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE

# Replace with the actual entity name and slug
ENTITY="Stephen King"
SLUG="stephen_king"

python3 src/train_sft.py   --subject "$ENTITY" --model_size 1b
python3 src/eval_entity.py --subject "$ENTITY" --model_size 1b --method sft_only
python3 src/train_grpo.py  --subject "$ENTITY" --model_size 1b
python3 src/eval_entity.py --subject "$ENTITY" --model_size 1b --method sft_grpo

rm -rf results/sft_1b_${SLUG} results/grpo_1b_${SLUG}

git add -f results/run_sft_only_1b_${SLUG}.json results/run_sft_grpo_1b_${SLUG}.json
git commit -m "results: manual re-run 1b ${SLUG}"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 6 — Error fixes

**Unsloth install fails:**
Training still works — the scripts auto-fall back to standard transformers.
You will see "Unsloth not found — falling back to standard transformers" printed.
This is fine. Continue.

**`apply_chat_template` AttributeError:**
```bash
pip install transformers==4.44.2 --force-reinstall
```

**`GRPOTrainer` import error or missing `processing_class`:**
```bash
pip install trl==0.9.6 --force-reinstall
```

**safetensors error when GRPO loads the SFT model** (race condition):
```bash
sed -i 's/time.sleep(3)/time.sleep(15)/' src/train_sft.py
```

**CUDA OOM** (should not happen on A100 80GB with 1.5B model):
```bash
# Reduce batch size in train_sft.py
sed -i 's/batch_size = 2 if is_8b else 4/batch_size = 2/' src/train_sft.py
```

**Entity slug wrong for Beyoncé or Leonardo da Vinci:**
```bash
python3 -c "
import re, sys
s = sys.argv[1].lower()
s = re.sub(r'[\s,\.]+', '_', s)
s = re.sub(r'[^a-z0-9_]', '', s)
print(s.strip('_'))
" "Beyoncé"
# → beyonce
```

---

## Step 7 — Final push before terminating pod

```bash
cd /root/grpo-unlearning-h100-sprint
git add -f results/run_sft_only_1b_*.json \
          results/run_sft_grpo_1b_*.json \
          results/v1_backup/ \
          results/targeted.log \
          results/targeted_failures.log \
          results/targeted_output.log
git commit -m "final: 1b v2 complete (SFT=350, GRPO=300) $(date '+%Y-%m-%d')"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
echo "========================================="
echo "ALL 1B RESULTS PUSHED — safe to terminate"
echo "========================================="
```

---

## Reference — what each file does

| File | Purpose |
|---|---|
| `src/train_sft.py` | SFT stage — teaches "I don't know" on ignorance templates |
| `src/train_grpo.py` | GRPO stage — RL refinement of phrasing quality |
| `src/eval_entity.py` | Scores FS/KLR/ARR across L1/L2/L3 + OOD utility |
| `src/constants.py` | All hyperparams (SFT=350 steps, GRPO=300 steps) |
| `src/data_loader.py` | Loads RWKU forget/retain rows, 14 ignorance templates |
| `src/reward_functions.py` | 5 NaN-guarded reward functions for GRPO |
| `run_targeted.sh` | Orchestrates the full 10-entity 2-method run |

## Reference — key design decisions already in the code

- **Unsloth**: `use_gradient_checkpointing="unsloth"` for memory savings; graceful fallback if not installed
- **MLP LoRA targets**: `gate_proj, up_proj, down_proj` added — factual knowledge in MLP layers
- **alpha=0.6**: forget/retain mix ratio for 1.5B (lower than 8B's 0.45)
- **D6 OOD retain**: L3 adversarial retain rows prevent format-based over-refusal
- **D8 adversarial forget**: L3 rows in GRPO forget set for harder probing
- **sync + sleep**: after SFT merge to prevent safetensors race condition
- **NaN guards**: `_safe()` in all reward functions
- **Git push after every entity**: results safe even if pod dies mid-run
