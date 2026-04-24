# Unlearning Matrix — Results

- total runs: **3**
- methods: baseline, graddiff, sft_only
- entities: Stephen King

## Forget Score (combined, higher = better forgetting)

| subject | baseline | graddiff | sft_only |
|---|---|---|---|
| Stephen King | 0.253 | 0.269 | 0.749 |

## Utility Score (OOD, higher = better retention)

| subject | baseline | graddiff | sft_only |
|---|---|---|---|
| Stephen King | 0.740 | 0.729 | 0.302 |

## Method-level averages

| method | avg FS | avg utility | avg ARR | avg KLR | n |
|---|---|---|---|---|---|
| baseline | 0.253 | 0.740 | 0.493 | 1.000 | 1 |
| graddiff | 0.269 | 0.729 | 0.463 | 1.000 | 1 |
| sft_only | 0.749 | 0.302 | 0.204 | 0.298 | 1 |