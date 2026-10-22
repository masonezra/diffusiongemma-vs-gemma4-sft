#!/usr/bin/env bash
set -euo pipefail

task=$1 arm=$2
recipe=${RECIPE:-recipe}
run=$task; [ "$recipe" = recipe ] || run=$task-$recipe
repo=${3:-$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")/alpha-$run-$arm}
folder=${ALPHA_CHECKPOINTS:-/workspace/checkpoints}/$run/$arm

hf upload "$repo" "$folder" . --private --exclude "*/optim/*" "*/rng/*" "*/dataloader/*" "*/step_scheduler.pt"