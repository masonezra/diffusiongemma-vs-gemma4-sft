# state_tracking

Between denoising steps DiffusionGemma hands itself S_i = c softmax(z_i) W_E at each canvas position i, with z_i the step's logits. Engels et al. (arXiv 2606.20560, Section 3.1) show that a monitor's top-8 reading R_8(z_i), the top 8 tokens with the remaining mass spread uniformly, suffices for the released checkpoint. They leave open whether that is architecture or training (Section 8.1). I vary the training rule of S on a task with an exact, unwritten intermediate state and test whether a top-k reading still suffices.

## Task
Sudoku, answer only: x^(0) in {0..9}^81 in, the unique solution x* out as 81 digits. The singles operator P fills every naked and hidden single in one synchronous round, x^(j+1) = P(x^(j)). Depth d = min{j : x^(j) = x*}, and the per-cell depth is D(i) = min{j : x^(j)_i != 0}. The grids x^(1..d-1) are never written, so they are the hidden state.

The first carrier was a shift register on Z_16^16 (recipe1, kept). Its answer is a length-k word in a non-abelian group, which is near chance without chain of thought (arXiv 2404.08819): exact match was 0% for k >= 2. Sudoku is learnable answer-only (arXiv 2608.00146).

Kaggle 9M and 3M puzzles, split by MD5 of the digit-relabelled solution. Train 32,000, balanced over depth bands {1-3, 4-6, 7-9, 10-12, 13+}; test 1,024 per band plus a not-singles tier.

| arm | self-conditioning p | gradient through S |
|---|---|---|
| `diffusion` | 0.5 | no (released rule) |
| `diffusion_always` | 1.0 | no |
| `diffusion_gradient` | 1.0 | yes |
| `ar` | Gemma 4 | |

1,000 steps x 32, LoRA r64 on attention, dense MLP and the self-conditioning MLP; canvas loss + encoder loss at weight 1.0.

    python -m state_tracking.meta.benchmark.build
    python -m state_tracking.meta.evaluate.evaluate --adapters adapters.json --gpus 1 --batch 16 --limit 512 --skip-chain

## Results
k*(T), the depth at which exact match crosses 50% (logistic fit above the floor), at T denoising steps:

| arm | T=1 | T=2 | T=4 | T=8 | T=16 | T=48 |
|---|---|---|---|---|---|---|
| Gemma 4 | 2.1 | | | | | |
| released | 3.0 | 5.1 | 7.5 | 10.3 | 11.4 | 11.7 |
| always | | 3.4 | 6.7 | 10.0 | 10.3 | 10.6 |
| gradient | | 4.9 | 7.7 | 8.2 | 8.2 | 8.1 |

k* rises with T under every rule, so denoising steps compose singles rounds. p = 1 removes the gain of the second step, and the gradient rule plateaus at T = 8.

At T = 2, S handed from step 1 to step 2 is replaced. The table gives D = A(full) - A(op), the drop in blank-cell accuracy in points, over 4,987 puzzles. A reading counts as sufficient when the 95% CI of D lies in [-4, 4].

| op | released | always | gradient |
|---|---|---|---|
| swap (another puzzle's S) | 60.1 | 16.4 | 66.2 |
| R_1 | 0.56 | 2.83 | 3.26 |
| R_3 | 0.00 | 0.33 | 0.17 |
| quantise (uniform on candidate set C_i) | 4.77 | -0.07 | 4.50 |

- R_3 suffices within 1 point for every rule, at T = 2, at T = 48 and under the shipped sampler.
- argmax S_i spells x^(j), 88.5 / 52.5 / 87.8 points above chance. Editing one depth-j cell of S moves the dependent cells as P predicts.
- The gradient rule moved information from rank 1 into ranks 2-3, not below.
- The always arm carries the candidate sets C_i instead, which is why quantise is lossless for it.
- Conditional V-information I_V(S -> x^(j)_i | log p) is 0.044 bits per cell [0.038, 0.050] (control -0.005), and a top-k monitor drops 0.000.

No computation hidden from a top-3 monitor at LoRA scale.

## Limits
One run per arm, LoRA, not pretraining scale. Depth counts parallel rounds, not NC1-hard composition. On a filled cell the final digit equals the intermediate one, so token identity can't separate an answer guess from a state. Gating use on swap rather than zero, and reporting the edit shift, were decided after the runs.