# infilling

Engels et al. (arXiv 2606.20560, Section 7.1(4)) ask where DiffusionGemma beats Gemma 4. Here: Python with 1-3 gaps. Both models see the code on both sides of every gap, so only emission differs. Gemma 4 writes the gaps left to right; DiffusionGemma denoises them jointly in one canvas. When gaps must agree, does joint emission pay?

## Setup
Programs from CodeContests, KodCode and OpenCodeInstruct, each executed and kept only if it passes. Exact and MinHash near copies (Jaccard ~0.8 over 8-token shingles of the renamed AST) and benchmark copies are removed. Split by group (contest problem or function-name set): 1% val, 3% test. A held-out row is kept only if its answer passes twice, every emptied gap fails and a small mutant fails.

Gap count is drawn 40 / 35 / 25 for 1 / 2 / 3, at most half the code hidden, each gap >= 20 non-space characters. Gap kinds, mixed 40 / 30 / 30:
- `line_span`: whole lines (HumanEval-Infilling line, SAFIM block);
- `node_span`: one expression (SAFIM control);
- `random_span`: two uniform character cuts (HumanEval-Infilling random span).

A pair of gaps is linked when the later reads a name the earlier binds and that name occurs nowhere outside the gaps; half the multi-gap programs are linked. The gap length is never given. Score: pass@1 of the spliced program.

287,293 train, 5,000 test, plus HumanEval-Infilling, HumanEval+, SAFIM and a panel of length-matched dependent and independent hole pairs. 1,000 steps x 32, LoRA r64, canvas 256.

    python -m infilling.meta.benchmark.build
    python -m infilling.meta.evaluate.compare test ar.jsonl diffusion.jsonl --several

The build executes about a million programs. The sandbox gate is a filter, not a security boundary, so run it on a machine with no credentials.

## Results
Pass@1 (%):

| set | Gemma 4 | DiffusionGemma | difference [95% CI] |
|---|---|---|---|
| test, several gaps (primary) | **62.8** | 46.2 | -16.6 [-18.6, -14.5], McNemar p < 1e-3 |
| test, 1 gap | **83.1** | 72.2 | -10.9 |
| linked, name hidden | **57.2** | 35.0 | -22.2 |
| HumanEval-Infilling | **76.3** | 66.2 | -10.2 [-14.3, -6.4] |
| SAFIM | **64.3** | 55.2 | -9.2 [-13.0, -5.6] |

The deficit scales with the length DiffusionGemma has to choose: ~3.5 points when the length is nearly fixed, -18 on answers of 5+ lines, -21 on random spans. It is form, not logic: 19.0% of its spliced programs don't parse, against 10.0%. Linking widens the gap (-15.1 separate, -22.2 linked), but the panel shows no dependence-specific cost (dependent minus independent, 2.7 points, CI [-2.7, 8.1]).

## Limits
One run per arm. Fills are a separate reply, not denoised in place between prefix and suffix (DreamOn), so the in-place advantage isn't tested. HumanEval and SAFIM predate both models.