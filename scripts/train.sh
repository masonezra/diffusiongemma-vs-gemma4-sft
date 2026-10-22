#!/usr/bin/env bash
set -euo pipefail

task=$1; shift || true
recipe=${RECIPE:-recipe}
run=$task; [ "$recipe" = recipe ] || run=$task-$recipe
root=${ALPHA_CHECKPOINTS:-/workspace/checkpoints}/$run
gpus=$(nvidia-smi -L | wc -l)
cd /workspace/alpha/src
mkdir -p "$root"

for arm in ${ARMS:-$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$task/$recipe.yaml'))['arms']))")}; do
  torchrun --nproc-per-node "$gpus" -m training.train "$task" "$arm" --recipe "$recipe" "$@" 2>&1 | tee -a "$root/$arm.log"
done