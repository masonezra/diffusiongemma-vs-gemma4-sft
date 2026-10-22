# inscriptions

Engels et al. (arXiv 2606.20560, Section 7.1(4)) ask where DiffusionGemma beats Gemma 4. Latin epigraphy tests two things a denoiser could do well or badly: copy the carved letters (~60% of the answer) unchanged while inserting around them, and size gaps whose length it isn't told.

## Setup
Input: the carved text, each run of lost letters written `[---]`. Reply: the Leiden edition, with abbreviations expanded `P(ublius)`, lost letters restored `Ho[e]nius`, and hidden letters restored `de [se]`.

Editors' restorations are one scholar's guess, so I hide carved letters too: span lengths and counts drawn from the editors' real restorations, up to 30% of a stone's letters. For those spans the stone is the ground truth. The model isn't told which gaps are hidden or how long any gap is.

EDCS dump of 2022-09-12, split by the last digit of the id as in Ithaca and Aeneas (3 test, 4 val). Training stones sharing >= 50% of their 30-letter windows with a held-out stone are dropped. 170,111 train, 3,000 val, 22,476 test; one pass, 2,657 steps x 64, LoRA r64. Test is scored once on the first 5,000 stones.

Floors from the training rows: lookup (commonest edition per carved word), template (plus gaps filled with fitting frequent words), retrieval (plus words from the most similar training stone).

    python -m inscriptions.meta.benchmark.build
    python -m inscriptions.meta.evaluate.evaluate test ar.test.jsonl --limit 5000 --against diffusion.test.jsonl

## Results
| % | Gemma 4 | DiffusionGemma | difference [95% CI] | template |
|---|---|---|---|---|
| recovered (hidden spans exact) | **40.1** | 33.0 | -7.1 [-8.0, -6.2] | 33.2 |
| recovered, 4-10 letters | **13.2** | 6.9 | -6.3 [-7.6, -5.0] | 8.8 |
| expanded | **82.5** | 73.3 | -9.3 [-10.3, -8.2] | 67.2 |
| faithful (carved letters kept) | **97.8** | 90.7 | -7.1 [-7.9, -6.2] | 100 |
| exact edition | **12.7** | 7.6 | -5.1 [-5.9, -4.3] | 0.7 |

DiffusionGemma ties the template floor on restoration (33.0 against 33.2).

recovered = P(length right) x P(letters right | length right): 0.489 x 0.820 = 0.401 for Gemma 4, 0.435 x 0.759 = 0.330 for DiffusionGemma. Giving DiffusionGemma Gemma 4's length rate alone recovers 4.1 of the 7.1 points, so most of the deficit is gap sizing.

Copy slips aren't just unparseable replies: on the 4,905 stones both parse, faithful is 98.2 against 92.2, and on replies of 129+ tokens 84.4 against 42.2.

## Limits
EDCS is public and predates both models, so absolute scores are upper bounds, with the exposure shared. Expansions and restorations are scored against one editor. Only 207 scored hidden spans are longer than 10 letters. One run per arm.