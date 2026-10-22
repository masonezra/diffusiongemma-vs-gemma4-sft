# training

One NVIDIA B200 on RunPod per run (the state-tracking gradient arm used a B300), container `nvcr.io/nvidia/nemo-automodel:26.08`. Base models: `google/gemma-4-26B-A4B-it` @ `4d7ae498` and `google/diffusiongemma-26B-A4B-it` @ `f7f5b7f5`.

`recipe.compose` merges `shared.yaml`, the model's yaml and the task recipe into one NeMo config. Matched across arms: data order (one seeded shuffle, no row repeated), LoRA rank and alpha, lr 1.5e-4 cosine to 1.5e-5 with 3% warmup, bf16 adapters on attention and the dense MLP, experts and router frozen.

Not matched, because each model's NeMo recipe fixes them:
- Loss: answer-token cross-entropy for Gemma 4; canvas loss + lambda x encoder loss for DiffusionGemma, with lambda = 1, or 0 on state_tracking recipe1.
- AdamW betas and decay: (0.9, 0.95), 0.01 against (0.95, 0.99), 1e-4.
- The chat template.

| task | steps x batch | seq_length |
|---|---|---|
| inscriptions | 2,657 x 64 | 768 |
| cryptanalysis | 3,000 x 32 | 1024 |
| infilling | 1,000 x 32 | 2048 |
| state_tracking recipe2 | 1,000 x 32 | 512 |
| state_tracking recipe1 | 2,000 x 32 | 1024 |

Each run writes provenance: config, package versions, LoRA parameter count, and SHA-256 of the code, data, chat template and first row's token ids. `scripts/evaluate.sh` copies logs and the final adapter to `results/`. bf16 MoE training isn't bitwise deterministic, and each arm was trained once.