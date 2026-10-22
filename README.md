I compare DiffusionGemma-26B-A4B with Gemma-4-26B-A4B after matched LoRA fine-tuning, on four benchmarks built around open questions in arXiv 2606.20560 (Sections 7.1(4) and 8.1). Each design and its results are in `src/<task>/note.md`, and the outputs are in `results/<task>/`.

| benchmark | task |
|---|---|
| `cryptanalysis` | classical-cipher decipherment, key hidden (recipe1) or given (recipe2) |
| `inscriptions` | Latin carved text to Leiden edition |
| `infilling` | Python with 1-3 masked gaps, scored by execution |
| `state_tracking` | Sudoku by singles depth; interventions on the state handed between denoising steps |

Training ran on RunPod with one B200, the container `nvcr.io/nvidia/nemo-automodel:26.08` and a network volume at `/workspace` holding this repo at `/workspace/alpha`.

Build the data on a CPU machine:
```
pip install -e . && cd src
python -m inscriptions.meta.benchmark.build
python -m cryptanalysis.meta.benchmark.build --recipe 1
python -m cryptanalysis.meta.benchmark.build --recipe 2
python -m infilling.meta.benchmark.build
hf download Ritvik19/Sudoku-Dataset train_0.parquet --repo-type dataset --revision 4d5aa527a9fb9aacca0b0d5b8b77d569fa9afcaa --local-dir ~/.cache/alpha/sudoku
python -m state_tracking.meta.benchmark.build
```

Then train and evaluate on the pod:
```
scripts/setup.sh
scripts/run.sh inscriptions
scripts/run.sh infilling
RECIPE=recipe1 scripts/run.sh cryptanalysis
RECIPE=recipe2 scripts/run.sh cryptanalysis
RECIPE=recipe2 scripts/run.sh state_tracking
```