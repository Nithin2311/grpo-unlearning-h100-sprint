# Claude Code Autopilot — SFT+GRPO Targeted Re-run (v2)

## What this run does

Re-runs **only SFT-only and SFT+GRPO** for all 10 PRIORITY_ENTITIES with
increased step counts:
- SFT:  **350 steps** (was 200)
- GRPO: **300 steps** (was 120 — previous run was budget-capped)

Previous runs of the other 5 methods (baseline, graddiff, simnpo, npo, rmu)
are already in the repo and will NOT be touched.

Estimated total time: ~5 hours on A100 SXM 80GB.
Expected FS improvement: mean 0.84 → ~0.95 for sft_grpo.

---

## Step 0: Start tmux FIRST (critical — prevents SSH disconnect killing the job)

```bash
tmux new-session -s sprint
```

You are now inside the tmux session. Everything below runs inside it.
If you disconnect and SSH back in: `tmux attach -t sprint`

---

## Step 1: Setup (~5 min)

```bash
export GIT_TOKEN=PASTE_YOUR_TOKEN_HERE

pip install torch>=2.1.0 transformers==4.44.2 trl==0.9.6 peft==0.12.0 \
    datasets==2.20.0 accelerate==0.33.0 bitsandbytes safetensors>=0.4.3 \
    python-pptx pandas 2>&1 | tail -10

git clone https://github.com/Nithin2311/grpo-unlearning-h100-sprint
cd grpo-unlearning-h100-sprint

git config user.email "nithinpalyam2311@gmail.com"
git config user.name "Nithin Palyam"

# Verify GPU and BF16 support
nvidia-smi | head -12
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| BF16:', torch.cuda.is_bf16_supported())"
```

Expected: `CUDA: True | BF16: True`

---

## Step 2: Back up v1 results

```bash
mkdir -p results/v1_backup
for f in results/run_sft_only_1b_*.json results/run_sft_grpo_1b_*.json; do
    [ -f "$f" ] && cp "$f" "results/v1_backup/$(basename $f)"
done
echo "Backed up $(ls results/v1_backup 2>/dev/null | wc -l) files"
```

---

## Step 3: Run

```bash
export GIT_TOKEN=PASTE_YOUR_TOKEN_HERE
bash run_targeted.sh 2>&1 | tee results/targeted_output.log
```

Monitor progress in a second tmux pane (Ctrl+B then %):
```bash
tail -f results/targeted.log
```

Check results as they come in:
```bash
python3 << 'EOF'
import json, glob
files = sorted(glob.glob("results/run_sft_grpo_1b_*.json"))
print(f"{'Entity':20s} {'FS':7s} {'util':6s}")
for f in files:
    d = json.load(open(f))
    c = d.get("combined", {})
    u = d.get("utility", {}).get("avg_utility_score", "?")
    name = f.split("run_sft_grpo_1b_")[1].replace(".json","").replace("_"," ").title()
    print(f"{name:20s} {str(c.get('forget_score','?')):7s} {str(u):6s}")
EOF
```

---

## Step 4: If an entity fails — manual re-run

```bash
cd /root/grpo-unlearning-h100-sprint
export GIT_TOKEN=PASTE_YOUR_TOKEN_HERE

# Replace ENTITY_NAME / ENTITY_SLUG with the failed entity
python3 src/train_sft.py   --subject "ENTITY_NAME" --model_size 1b
python3 src/eval_entity.py --subject "ENTITY_NAME" --model_size 1b --method sft_only
python3 src/train_grpo.py  --subject "ENTITY_NAME" --model_size 1b
python3 src/eval_entity.py --subject "ENTITY_NAME" --model_size 1b --method sft_grpo

git add -f results/run_sft_only_1b_ENTITY_SLUG.json \
          results/run_sft_grpo_1b_ENTITY_SLUG.json
git commit -m "results: manual re-run ENTITY_SLUG"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 5: Error fixes

**`apply_chat_template` error** — transformers version wrong:
```bash
pip install transformers==4.44.2 --force-reinstall
```

**`GRPOTrainer` import error** — trl version wrong:
```bash
pip install trl==0.9.6 --force-reinstall
```

**safetensors incomplete metadata when GRPO loads SFT model** — race condition:
```bash
sed -i 's/time.sleep(3)/time.sleep(13)/' src/train_sft.py
```

**CUDA OOM** (should not happen on A100 80GB for 1.5B):
Edit `train_sft.py` and `train_grpo.py`: set `per_device_train_batch_size=2`,
`gradient_accumulation_steps=4`.

**Entity slug for Beyoncé / Leonardo da Vinci** — check exact slug:
```bash
python3 -c "
import re, sys
s = sys.argv[1].lower()
s = re.sub(r'[\s,\.]+', '_', s)
s = re.sub(r'[^a-z0-9_]', '', s)
print(s.strip('_'))
" "Beyoncé"
```

---

## Step 6: If time remains after all 10 entities complete

If the 10-entity run finishes with budget left and FS is consistently high,
try a bonus 8B run for Stephen King only:
```bash
python3 src/train_sft.py   --subject "Stephen King" --model_size 8b
python3 src/eval_entity.py --subject "Stephen King" --model_size 8b --method sft_only
python3 src/train_grpo.py  --subject "Stephen King" --model_size 8b
python3 src/eval_entity.py --subject "Stephen King" --model_size 8b --method sft_grpo

git add -f results/run_sft_*_8b_stephen_king.json
git commit -m "bonus: 8b stephen_king sft+grpo 350/300 steps"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
```

---

## Step 7: Final push before terminating pod

```bash
cd /root/grpo-unlearning-h100-sprint
git add -f results/run_sft_only_1b_*.json results/run_sft_grpo_1b_*.json \
          results/v1_backup/ results/targeted.log \
          results/targeted_failures.log results/targeted_output.log
git commit -m "final: v2 sft+grpo complete (350/300 steps) $(date '+%Y-%m-%d')"
git push "https://${GIT_TOKEN}@github.com/Nithin2311/grpo-unlearning-h100-sprint" main
echo "ALL RESULTS PUSHED — safe to terminate pod"
```

---

## Key constants (already set in constants.py)

| Parameter | Value |
|---|---|
| SFT steps (`MAX_STEPS_1B`) | 350 |
| GRPO steps (`GRPO_STEPS_1B`) | 300 |
| SFT LR | 3e-5 |
| GRPO LR | 2e-6 |
| LoRA r (1.5B) | 16 |
| Alpha (forget/retain ratio) | 0.6 |
| Batch size SFT | 4 |
| Batch size GRPO | 2 |
| Model | Qwen/Qwen2.5-1.5B-Instruct |

## What previous runs showed

- **v1 SFT+GRPO (120 GRPO steps)**: mean FS=0.841 — GRPO under-cooked
- **Original SK single-entity (300 GRPO steps)**: FS=1.000 — confirmed working
- **SFT-only (200 steps)**: mean FS=0.845 — strong but utility=0.29
- **SimNPO**: best harmonic mean — already in repo, don't re-run
- **GradDiff/NPO/RMU**: do NOT benefit from more steps — don't re-run
- **safetensors race**: already handled with sync+sleep in train_sft.py
- **Variance collapse**: GRPO alone always fails; SFT pre-conditioning is mandatory
- **D6 OOD retain**: applied by default — prevents over-refusal on other entities
