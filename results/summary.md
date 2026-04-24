# Unlearning Matrix — Results

- total runs: **28**
- methods: baseline, graddiff, npo, rmu, sft_grpo, sft_only, simnpo
- entities: Donald Trump, Elon Musk, Stephen King, Taylor Swift

## Forget Score (combined, higher = better forgetting)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Donald Trump | 0.632 | 0.617 | 0.674 | 0.667 | 0.881 | 0.906 | 0.637 |
| Elon Musk | 0.300 | 0.316 | 0.343 | 0.615 | 0.710 | 0.722 | 0.438 |
| Stephen King | 0.253 | 0.269 | 0.337 | 0.253 | 0.728 | 0.749 | 0.402 |
| Taylor Swift | 0.361 | 0.369 | 0.347 | 0.436 | 0.870 | 0.857 | 0.453 |

## Utility Score (OOD, higher = better retention)

| subject | baseline | graddiff | npo | rmu | sft_grpo | sft_only | simnpo |
|---|---|---|---|---|---|---|---|
| Donald Trump | 0.750 | 0.733 | 0.733 | 0.733 | 0.283 | 0.292 | 0.733 |
| Elon Musk | 0.750 | 0.750 | 0.733 | 0.733 | 0.242 | 0.250 | 0.700 |
| Stephen King | 0.740 | 0.729 | 0.750 | 0.729 | 0.333 | 0.302 | 0.719 |
| Taylor Swift | 0.750 | 0.717 | 0.733 | 0.742 | 0.367 | 0.358 | 0.692 |

## Method-level averages

| method | avg FS | avg utility | avg ARR | avg KLR | n |
|---|---|---|---|---|---|
| baseline | 0.386 | 0.747 | 0.478 | 0.749 | 4 |
| graddiff | 0.393 | 0.732 | 0.462 | 0.753 | 4 |
| npo | 0.425 | 0.737 | 0.366 | 0.783 | 4 |
| rmu | 0.493 | 0.734 | 0.326 | 0.689 | 4 |
| sft_grpo | 0.797 | 0.306 | 0.204 | 0.202 | 4 |
| sft_only | 0.809 | 0.301 | 0.177 | 0.206 | 4 |
| simnpo | 0.483 | 0.711 | 0.229 | 0.806 | 4 |