# Unlearning Matrix — Results

- total runs: **21**
- methods: baseline, graddiff, npo, rmu, sft_grpo, sft_only, simnpo
- entities: Elon Musk, Stephen King, Taylor Swift

## Forget Score (combined, higher = better forgetting)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Elon Musk | 0.300 | 0.316 | 0.343 | 0.615 | 0.710 | 0.722 | 0.438 |
| Stephen King | 0.253 | 0.269 | 0.337 | 0.253 | 0.728 | 0.749 | 0.402 |
| Taylor Swift | 0.361 | 0.369 | 0.347 | 0.436 | 0.870 | 0.857 | 0.453 |

## Utility Score (OOD, higher = better retention)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Elon Musk | 0.750 | 0.750 | 0.733 | 0.733 | 0.242 | 0.250 | 0.700 |
| Stephen King | 0.740 | 0.729 | 0.750 | 0.729 | 0.333 | 0.302 | 0.719 |
| Taylor Swift | 0.750 | 0.717 | 0.733 | 0.742 | 0.367 | 0.358 | 0.692 |

## Method-level averages

| method | avg FS | avg utility | avg ARR | avg KLR | n |
|---|---|---|---|---|---|
| baseline | 0.305 | 0.747 | 0.514 | 0.877 | 3 |
| graddiff | 0.318 | 0.732 | 0.490 | 0.874 | 3 |
| npo | 0.342 | 0.739 | 0.404 | 0.911 | 3 |
| rmu | 0.435 | 0.735 | 0.322 | 0.809 | 3 |
| sft_grpo | 0.769 | 0.314 | 0.216 | 0.245 | 3 |
| sft_only | 0.776 | 0.303 | 0.197 | 0.251 | 3 |
| simnpo | 0.431 | 0.704 | 0.242 | 0.895 | 3 |