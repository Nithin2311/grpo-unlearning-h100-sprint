# Unlearning Matrix — Results

- total runs: **11**
- methods: baseline, graddiff, npo, rmu, sft_grpo, sft_only, simnpo
- entities: Stephen King, Taylor Swift

## Forget Score (combined, higher = better forgetting)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Stephen King | 0.253 | 0.269 | 0.337 | 0.253 | 0.728 | 0.749 | 0.402 |
| Taylor Swift | 0.361 | 0.369 | — | — | — | 0.857 | 0.453 |

## Utility Score (OOD, higher = better retention)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Stephen King | 0.740 | 0.729 | 0.750 | 0.729 | 0.333 | 0.302 | 0.719 |
| Taylor Swift | 0.750 | 0.717 | — | — | — | 0.358 | 0.692 |

## Method-level averages

| method | avg FS | avg utility | avg ARR | avg KLR | n |
|---|---|---|---|---|---|
| baseline | 0.307 | 0.745 | 0.470 | 0.916 | 2 |
| graddiff | 0.319 | 0.723 | 0.455 | 0.907 | 2 |
| npo | 0.337 | 0.750 | 0.338 | 0.989 | 1 |
| rmu | 0.253 | 0.729 | 0.493 | 1.000 | 1 |
| sft_grpo | 0.728 | 0.333 | 0.246 | 0.298 | 1 |
| sft_only | 0.803 | 0.330 | 0.202 | 0.191 | 2 |
| simnpo | 0.428 | 0.705 | 0.238 | 0.907 | 2 |