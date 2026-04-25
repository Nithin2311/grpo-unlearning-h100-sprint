# Autopilot Prompt — 8B Session (Llama-3.1-8B)

You are Claude Code running autonomously on a fresh RunPod A100 SXM 80GB GPU.
You have no memory of previous sessions. Follow every step in order.
Push results to GitHub after every completed entity so nothing is lost.

---

## Context

This is a Deep RL course final project on machine unlearning. The goal is to
remove specific entity knowledge from LLMs without retraining from scratch,
evaluated on the RWKU benchmark (jinzhuoran/RWKU on HuggingFace).

You are running the **8B session** (meta-llama/Meta-Llama-3.1-8B-Instruct).
A separate pod is running the 1.5B session simultaneously — ignore it.

You will train two methods for 5 entities:
- **SFT-only**: supervised fine-tuning on ignorance phrases (300 steps, alpha=0.45)
- **SFT+GRPO**: SFT (300 steps) then RL refinement via GRPO (200 steps)

Entities: Stephen King, Taylor Swift, Donald Trump, Beyoncé, Aristotle.
These were chosen for diversity and to anchor against the original paper's
8B Stephen King result (FS=0.979 previously achieved).

Metric: Forget Score = 1 − (KLR + ARR) / 2. Higher is better. Target ≥ 0.95.
- KLR = keyword leak rate (entity name appears in output)
- ARR = answer recall rate (correct answer appears in output)

**Critical 8B difference from 1B**: alpha=0.45 (not 0.6). The 8B model is a
stronger instruction-follower and over-refusals aggressively — with alpha=0.6
it refuses ALL questions including unrelated topics. This is already set in
constants.py. Do not change it.

Estimated time: ~80-90 min per entity × 5 = ~7.5 hours.
Estimated cost: ~$21 on A100 SXM at $2.79/hr.

---

## Step 0 — Start tmux immediately

```bash
tmux new-session -s sprint8b
```

All remaining steps run inside this tmux session.
If SSH disconnects, reconnect and run: `tmux attach -t sprint8b`

---

## Step 1 — Install Unsloth and dependencies

```bash
# Unsloth: 2-4x faster training, 40-60% less VRAM — critical for 8B on 80GB
pip install unsloth 2>&1 | tail -5
# If that fails:
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Core ML stack — pinned versions confirmed working on A100 SXM
pip install torch>=2.1.0 transformers==4.44.2 trl==0.9.6 peft==0.12.0 \
    datasets==2.20.0 accelerate==0.33.0 bitsandbytes safetensors>=0.4.3 \
    python-pptx pandas huggingface_hub 2>&1 | tail -10

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
transformers automatically — training will still work, just slower and with
higher VRAM usage. For 8B this may be tight — try to get Unsloth working.

---

## Step 2 — Authenticate with HuggingFace (required for Llama)

Llama-3.1-8B-Instruct is a gated model. You must authenticate before
the training scripts can download it.

```bash
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE

huggingface-cli login --token "$HF_TOKEN"

# Verify Llama access
python3 -c "
from huggingface_hub import whoami, model_info
print('Authenticated as:', whoami()['name'])
info = model_info('meta-llama/Meta-Llama-3.1-8B-Instruct')
print('Llama access confirmed:', info.id)
"
```

If this fails with "403 Forbidden": the HuggingFace account has not accepted
the Llama license. The user must visit:
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
and click "Agree and access repository". Then re-run this step.

---

## Step 3 — Clone repo and configure git

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE

git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint
cd grpo-unlearning-h100-sprint

git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"

# Verify the repo has the right files
ls src/
# Expected: train_sft.py  train_grpo.py  eval_entity.py  constants.py
#           data_loader.py  reward_functions.py  ...

# Check step counts and alpha for 8B
python3 -c "
from src.constants import MAX_STEPS_8B, GRPO_STEPS_8B, ALPHA_8B
print(f'SFT steps:  {MAX_STEPS_8B}   (expected 300)')
print(f'GRPO steps: {GRPO_STEPS_8B}  (expected 200)')
print(f'Alpha:      {ALPHA_8B}    (expected 0.45 — prevents over-refusal)')
"
```

---

## Step 4 — Check disk space

8B model weights are ~16GB per checkpoint. The script deletes weights after
each entity, but you need headroom during training.

```bash
df -h /root
# Need at least 80GB free
# If low:
pip cache purge
```

---

## Step 5 — Run

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE
bash run_targeted_8b.sh 2>&1 | tee results/targeted_8b_output.log
```

The script runs 5 entities in this order:
Stephen King → Taylor Swift → Donald Trump → Beyoncé → Aristotle

Per entity: SFT train → SFT eval → GRPO train → GRPO eval → git push → delete weights.
Estimated time: ~80-90 min per entity × 5 = ~7.5 hours total.

**Monitor progress in a second tmux pane** (Ctrl+B then %):
```bash
watch -n 30 "tail -15 results/targeted_8b.log && echo '---' && nvidia-smi | grep MiB"
```

**Check results as they land:**
```bash
python3 -c "
import json, glob
files = sorted(glob.glob('results/run_sft_grpo_8b_*.json'))
print(f'Completed: {len(files)}/5')
print(f'{\"Entity\":<20} {\"FS\":>7} {\"KLR\":>7} {\"Util\":>6}')
for f in files:
    d = json.load(open(f))
    c = d.get('combined', {})
    u = d.get('utility', {}).get('avg_utility_score', '?')
    name = f.split('run_sft_grpo_8b_')[1].replace('.json','').replace('_',' ').title()[:19]
    print(f'{name:<20} {str(c.get(\"forget_score\",\"?\"))!s:>7} {str(c.get(\"keyword_leak_rate\",\"?\"))!s:>7} {str(u)!s:>6}')
"
```

---

## Step 6 — If the script fails on a specific entity, re-run it manually

```bash
cd /root/grpo-unlearning-h100-sprint
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE

# Replace with the actual entity name and slug
ENTITY="Stephen King"
SLUG="stephen_king"

python3 src/train_sft.py   --subject "$ENTITY" --model_size 8b
python3 src/eval_entity.py --subject "$ENTITY" --model_size 8b --method sft_only
python3 src/train_grpo.py  --subject "$ENTITY" --model_size 8b
python3 src/eval_entity.py --subject "$ENTITY" --model_size 8b --method sft_grpo

# Free disk — 8B weights are ~16GB each
rm -rf results/sft_8b_${SLUG} results/grpo_8b_${SLUG}

git add -f results/run_sft_only_8b_${SLUG}.json results/run_sft_grpo_8b_${SLUG}.json
git commit -m "results: manual re-run 8b ${SLUG}"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 7 — Error fixes

**Unsloth install fails:**
Training still works — the scripts auto-fall back to standard transformers.
You will see "Unsloth not found — falling back to standard transformers".
For 8B without Unsloth, VRAM may be tight. If you get OOM, see below.

**`apply_chat_template` AttributeError:**
```bash
pip install transformers==4.44.2 --force-reinstall
```

**`GRPOTrainer` import error:**
```bash
pip install trl==0.9.6 --force-reinstall
```

**safetensors error when GRPO loads the SFT model** (race condition):
```bash
sed -i 's/time.sleep(3)/time.sleep(15)/' src/train_sft.py
```

**CUDA OOM during 8B SFT** (batch size is already 2 for 8B):
```bash
# Reduce sequence length
sed -i 's/max_seq_length=512/max_seq_length=384/' src/train_sft.py
```

**CUDA OOM during 8B GRPO:**
```bash
# Reduce completion length
sed -i 's/max_completion_length=64/max_completion_length=32/' src/train_grpo.py
```

**Over-refusal after SFT** (model refuses everything — utility near 0):
This means alpha is too high. Check the SFT-only eval:
```bash
python3 -c "
import json
d = json.load(open('results/run_sft_only_8b_stephen_king.json'))
print('utility:', d.get('utility',{}).get('avg_utility_score'))
"
```
If utility < 0.1, re-run that entity with lower alpha:
```bash
python3 src/train_sft.py --subject "Stephen King" --model_size 8b --alpha 0.3
```

**Disk full** (8B weights not cleaned):
```bash
find results/ -name "*.safetensors" -delete
find results/ -name "pytorch_model*.bin" -delete
du -sh results/*/
```

**Beyoncé slug issue** (special character):
```bash
python3 -c "
import re
s = 'Beyoncé'.lower()
s = re.sub(r'[\s,\.]+', '_', s)
s = re.sub(r'[^a-z0-9_]', '', s)
print(s.strip('_'))
"
# → beyonce
```

---

## Step 8 — If time remains after 5 entities complete

If all 5 finish with budget remaining (check `date` vs pod expiry), add Elon Musk:

```bash
export GIT_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
export HF_TOKEN=PASTE_YOUR_HUGGINGFACE_TOKEN_HERE

python3 src/train_sft.py   --subject "Elon Musk" --model_size 8b
python3 src/eval_entity.py --subject "Elon Musk" --model_size 8b --method sft_only
python3 src/train_grpo.py  --subject "Elon Musk" --model_size 8b
python3 src/eval_entity.py --subject "Elon Musk" --model_size 8b --method sft_grpo

rm -rf results/sft_8b_elon_musk results/grpo_8b_elon_musk
git add -f results/run_sft_only_8b_elon_musk.json results/run_sft_grpo_8b_elon_musk.json
git commit -m "bonus: 8b elon_musk"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 9 — Final push before terminating pod

```bash
cd /root/grpo-unlearning-h100-sprint
git add -f results/run_sft_only_8b_*.json \
          results/run_sft_grpo_8b_*.json \
          results/targeted_8b.log \
          results/targeted_8b_failures.log \
          results/targeted_8b_output.log
git commit -m "final: 8b complete (SFT=300, GRPO=200) $(date '+%Y-%m-%d')"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
echo "========================================="
echo "ALL 8B RESULTS PUSHED — safe to terminate"
echo "========================================="
```

---

## Reference — key constants for 8B (set in constants.py)

| Parameter | 8B value | 1B value | Why different |
|---|---|---|---|
| Model | Llama-3.1-8B-Instruct | Qwen2.5-1.5B-Instruct | Different scale |
| SFT steps | 300 | 350 | 8B learns faster per step |
| GRPO steps | 200 | 300 | 8B needs fewer RL steps |
| Alpha | 0.45 | 0.60 | 8B over-refusals without lower alpha |
| LoRA r | 32 | 16 | Larger r for larger model |
| SFT batch size | 2 | 4 | 8B needs more VRAM headroom |
| HF token | Required | Not required | Llama is gated; Qwen is open |

## Reference — what each file does

| File | Purpose |
|---|---|
| `src/train_sft.py` | SFT stage — teaches "I don't know" on ignorance templates |
| `src/train_grpo.py` | GRPO stage — RL refinement of phrasing quality |
| `src/eval_entity.py` | Scores FS/KLR/ARR across L1/L2/L3 + OOD utility |
| `src/constants.py` | All hyperparams — already configured correctly |
| `src/data_loader.py` | Loads RWKU forget/retain rows, 14 ignorance templates |
| `src/reward_functions.py` | 5 NaN-guarded reward functions for GRPO |
| `run_targeted_8b.sh` | Orchestrates the 5-entity 8B run |

## Reference — key design decisions already in the code

- **Unsloth**: `use_gradient_checkpointing="unsloth"` for 40-60% VRAM savings; auto-fallback if not installed
- **MLP LoRA targets**: `gate_proj, up_proj, down_proj` — factual memory in MLP layers, not just attention
- **alpha=0.45**: prevents 8B over-refusal (0.6 causes model to refuse all questions)
- **D6 OOD retain**: L3 retain rows prevent format-based over-refusal on other entities
- **sync + sleep**: after SFT merge to prevent safetensors race condition
- **NaN guards**: `_safe()` in all reward functions
- **Disk cleanup**: weights deleted after each entity (8B weights are ~16GB)
- **Git push after every entity**: results safe even if pod runs out of credit mid-run
