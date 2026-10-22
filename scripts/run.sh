#!/usr/bin/env bash
set -uo pipefail

task=$1
recipe=${RECIPE:-recipe}
run=$task; [ "$recipe" = recipe ] || run=$task-$recipe
root=${ALPHA_CHECKPOINTS:-/workspace/checkpoints}/$run
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root"
rm -f "$root"/{TRAIN_EXIT,EVAL_EXIT,RUN_EXIT}

RECIPE=$recipe "$here/train.sh" "$task" > "$root/run.log" 2>&1
train=$?
echo $train > "$root/TRAIN_EXIT"
eval=1

if [ $train -eq 0 ]; then
  RECIPE=$recipe "$here/evaluate.sh" "$task" > "$root/evaluate.log" 2>&1
  eval=$?
  echo $eval > "$root/EVAL_EXIT"
fi

echo $((train + eval)) > "$root/RUN_EXIT"