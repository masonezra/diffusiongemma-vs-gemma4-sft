#!/usr/bin/env bash
set -euo pipefail

task=${1:-state_tracking}; shift || true
recipe=${RECIPE:-recipe}
out=/workspace/checkpoints/smoke/$task; [ "$recipe" = recipe ] || out=$out-$recipe
cd /workspace/alpha/src
mkdir -p "$out"
gpus=$(nvidia-smi -L | wc -l)
arms=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$task/$recipe.yaml'))['arms']))")
sub=$(python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml')).get('dataset', ''))")
budget=$(python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml')).get('budget', ''))")
model() { python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml'))['arms']['$1']['model'])"; }
carrier=${STATE_TRACKING_TASK:-sudoku}
probe="resample 3"; [ "$carrier" = shift_register ] && probe="resample 8"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > "$out/memory.csv" &
watcher=$!

for arm in $arms; do
  torchrun --nproc-per-node "$gpus" -m training.train "$task" "$arm" --recipe "$recipe" --step_scheduler.max_steps 20 --step_scheduler.ckpt_every_steps 10 \
    --step_scheduler.val_every_steps 10 --lr_scheduler.lr_warmup_steps 2 --checkpoint.checkpoint_dir "$out/$arm" "$@" 2>&1 | tee "$out/$arm.log"
done

first=${arms%% *}
torchrun --nproc-per-node "$gpus" -m training.train "$task" "$first" --recipe "$recipe" --step_scheduler.max_steps 30 --step_scheduler.ckpt_every_steps 10 \
  --step_scheduler.val_every_steps 10 --lr_scheduler.lr_warmup_steps 2 --checkpoint.checkpoint_dir "$out/$first" "$@" 2>&1 | tee "$out/$first.resume.log"

for arm in $arms; do
  last=$(ls -d "$out/$arm"/epoch_*_step_* | sort -V | tail -1)
  python -m common.generate "$(model "$arm")" "../dataset/$task/$sub/val.jsonl" --checkpoint "$last" --limit 16 ${budget:+--budget $budget} --out "$out/$arm.replies.jsonl"

  if [ "$task" = state_tracking ] && [ "$(model "$arm")" = diffusion ]; then
    python -m state_tracking.meta.evaluate.generate val diffusion --run "smoke.$arm" --checkpoint "$last" --steps 2 --channel "$probe" --trace --states --machines "$carrier" --limit 2
    ls "../results/$task/$sub/runs/val.diffusion.smoke.$arm" | head -3
  fi
done

kill $watcher
echo "peak MiB per GPU sample: $(sort -n "$out/memory.csv" | tail -1)"
grep -h "step_time\|Step time\|it/s" "$out"/*.log | tail -20 || true