# cryptanalysis

Engels et al. (arXiv 2606.20560, Section 7.1(4)) ask where DiffusionGemma beats Gemma 4. In decipherment one key K constrains every output position, so p = D_K(c) has to be globally consistent. If joint denoising helps anywhere, it should help here.

## Setup
c = E_K(p) in, p in {a..z}^n out, one token per symbol, n <= 254 (one 256-slot canvas), thinking off. Plaintexts are arXiv abstracts; held-out ones are from 2607 on and share no 32-letter run with training text. Every row draws a fresh K.

| level | cipher | key space |
|---|---|---|
| 1, 2 | substitution (1 keeps word gaps) | S_26 |
| 3 | Vigenere | period 3..10 |
| 4 | autokey | primer 3..10 |
| 5 | columnar transposition | width 4..9 |
| 6, 7 | homophonic | 27..48, 49..69 symbols |

- recipe1, keyless: the prompt is c alone. Levels 1-2 are graded at n in {16, 32, 64, 128, 254}. Levels 3-7 draw n ~ U[L, 254] with L = ceil(2.5 H(K) / 3.2), i.e. 2.5x the unicity distance at 3.2 bits of redundancy per letter. 320,000 train rows, 6,650 test.
- recipe2, keyed: the cipher and K are printed before c, so the task is applying a given key. n in {32, 64, 128}. 96,000 train, 3,500 test, and `test_random`: the same rows over uniform random letters, which removes the English prior.

Both arms: 3,000 steps x 32, LoRA r64 / alpha 128, lr 1.5e-4 cosine. Gemma 4 decodes greedily; DiffusionGemma uses the released sampler (<= 48 steps). Metrics: positional symbol error, edit-distance error, exact match, and clash, the share of symbols that contradict the reply's own majority key.

    python -m cryptanalysis.meta.benchmark.build --recipe 1
    python -m cryptanalysis.meta.evaluate.score test ar.test.jsonl --data ../dataset/cryptanalysis/recipe1 --against diffusion.test.jsonl

## Results
recipe1 (test, symbol error % / exact %):

| level | Gemma 4 | DiffusionGemma | paired error difference [95% CI] |
|---|---|---|---|
| 1, n = 254 | 68.5 / 1.1 | **0.12 / 82.1** | -68.4 [-70.8, -65.7] |
| 2, n = 254 | 84.1 / 0.0 | **3.2 / 77.4** | -80.9 [-83.8, -77.9] |
| 3-7 | 90-93 / 0 | 87-88 / 0-0.2 | floor |

DiffusionGemma learned keyless substitution; Gemma 4 writes fluent English under a wrong key. Levels 3-7 are at the floor for both. DiffusionGemma's 5-point lower positional error there comes with a 4-7 point higher edit error, so it is not a win.

recipe2: both arms reach 94-100% exact on every level except autokey, on English and on random letters, so the key is used, not the prior. Both get 0% on autokey: right on the primer, chance after it. Neither learned the recurrence k_i = p_(i-m) over its own output. One denoising step matches 48 on every level except columnar (exact 99 -> 58).

## Limits
One run per recipe, one register, classical ciphers only. Pretraining exposure of the held-out abstracts isn't excluded. In recipe1 level and length are confounded at the top: level 7 rows are 180-254 symbols. Validation loss can't rank the arms, since the canvas loss isn't a likelihood.