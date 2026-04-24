# Claude Code Autopilot — GRPO Multi-Entity Unlearning Sprint

## Context
You are running a 3-hour machine unlearning experiment on an A100 SXM 80GB GPU.
This is for a Deep RL course final project. The goal is to train and score
LLM unlearning models across multiple entities from the RWKU benchmark.

Each run must complete within 30 minutes. Push results to git after every run
so nothing is lost if the session ends early.

## Step 1: Setup (do this first, takes ~10 min)

```bash
# Set your git token (replace with fresh token from github.com/settings/tokens)
export GIT_TOKEN=YOUR_FRESH_TOKEN_HERE
export GIT_REMOTE=https://github.com/Nithin2311/grpo-unlearning-h100-sprint

# Install dependencies
pip install transformers==4.44.2 trl==0.9.6 peft==0.12.0 datasets==2.20.0 \
    accelerate==0.33.0 bitsandbytes safetensors python-pptx 2>&1 | tail -5

# Login to HuggingFace (required for Llama-3.1-8B)
huggingface-cli login   # paste token: hf_...Nithin2311 token

# Clone repo
git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint
cd grpo-unlearning-h100-sprint

# Configure git for pushing
git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"

# Verify GPU
nvidia-smi | head -15
```

## Step 2: Run the Sprint

```bash
# Set START_TIME for the 3-hour budget tracker
export START_TIME=$(date +%s)
export GIT_TOKEN=YOUR_FRESH_TOKEN_HERE
export GIT_REMOTE=https://github.com/Nithin2311/grpo-unlearning-h100-sprint

bash run_sprint.sh 2>&1 | tee results/sprint.log
```

## Step 3: If run_sprint.sh fails at any point

Run individual pipeline for a single entity manually:

```bash
cd /root/grpo-unlearning-h100-sprint

# SFT stage
python3 src/train_sft.py --subject "Stephen King" --model_size 1b

# GRPO stage
python3 src/train_grpo.py --subject "Stephen King" --model_size 1b

# Score
python3 src/eval_entity.py --subject "Stephen King" --model_size 1b --method sft_grpo

# Push results
git add -f results/*.json
git commit -m "results: sft_grpo 1b stephen_king"
REMOTE="https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint"
git push "$REMOTE" main
```

## Step 4: If you encounter version errors

The pinned versions in requirements.txt were confirmed working. If you still
hit errors, try these specific fixes:

**transformers tokenizer error ("apply_chat_template"):**
```bash
pip install transformers==4.44.2 --force-reinstall
```

**trl GRPOTrainer import error:**
```bash
pip install trl==0.9.6 --force-reinstall
```

**peft LoraConfig error (missing target_modules):**
```bash
pip install peft==0.12.0 --force-reinstall
```

**safetensors incomplete metadata (race condition on merge):**
The train_sft.py script already includes `sync && sleep 3` before returning.
If you still see this error, add `import time; time.sleep(10)` before loading
the merged model.

**bfloat16 not supported:**
Replace `torch_dtype=torch.bfloat16` with `torch_dtype=torch.float16`
in constants.py is NOT the right fix — instead, check CUDA version:
```bash
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# A100 SXM = capability 8.0, bfloat16 IS supported
```

**CUDA OOM on 8B model:**
Reduce batch size in train_sft.py: per_device_train_batch_size=2, gradient_accumulation_steps=4

## Step 5: Adaptive iteration

After each completed run, check the result JSON:
```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('results/score_*.json')):
    d = json.load(open(f))
    c = d.get('combined', {})
    print(f'{f}: FS={c.get(\"forget_score\",\"?\")}  KLR={c.get(\"keyword_leak_rate\",\"?\")}')
"
```

If FS < 0.7 for any entity: the entity may be less memorized. Check if it's worth re-running
with fewer steps (--steps 100) to save time.

If FS = 1.0 for all completed runs: you have budget to try new entities.
Next priority entities: Leonardo da Vinci, Donald Trump, Kim Kardashian, Aristotle.

## Step 6: Final push before session ends

```bash
cd /root/grpo-unlearning-h100-sprint
git add -f results/*.json results/sprint.log
git commit -m "final: all sprint results $(date '+%Y-%m-%d')"
REMOTE="https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint"
git push "$REMOTE" main
echo "ALL RESULTS PUSHED — safe to terminate pod"
```

## Key design decisions already made

- **D6 applied by default**: L3 OOD retain samples included in all SFT runs
  (prevents [BLANK]-format over-refusal on other entities)
- **D8 applied by default**: L3 adversarial rows included in GRPO forget set
- **22-keyword KLR for Stephen King**: curated keyword set in constants.py
- **Auto-keyword generation for new entities**: splits entity name into tokens
- **NaN guards in all reward functions**: _safe() wrapper throughout
- **30-min timeout per stage**: enforced via `timeout 1500` in run_sprint.sh
- **Git checkpoint after every run**: results are safe even if pod dies

## What we learned from past sessions

1. SFT+GRPO achieves FS=1.0 on Stephen King (1.5B and 8B) — confirmed
2. D6 (L3 OOD retain) > D8 (L3 adversarial forget) for specificity
3. safetensors race condition: always sync+sleep after SFT merge
4. transformers==4.44.2 + trl==0.9.6 + peft==0.12.0 is the confirmed working combo
5. Variance collapse: GRPO alone fails; SFT pre-conditioning is mandatory
6. 8B needs alpha=0.45 (not 0.6) to prevent over-refusal
