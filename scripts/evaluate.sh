#!/usr/bin/env bash
set -euo pipefail

task=$1
recipe=${RECIPE:-recipe}
run=$task; [ "$recipe" = recipe ] || run=$task-$recipe
root=${ALPHA_CHECKPOINTS:-/workspace/checkpoints}/$run
cd /workspace/alpha/src
sub=$(python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml')).get('dataset', ''))")
width=$(python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml')).get('canvas', ''))")
budget=$(python -c "import yaml; print(yaml.safe_load(open('$task/$recipe.yaml')).get('budget', ''))")
arms=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$task/$recipe.yaml'))['arms']))")
data=../dataset/$task${sub:+/$sub}
out=/workspace/alpha/results/$task${sub:+/$sub}
canvas_flag="${width:+--canvas $width} ${budget:+--budget $budget}"
base_flag=${budget:+--budget $budget}
mkdir -p "$out"
last() { ls -d "$root/$1"/epoch_*_step_* 2>/dev/null | sort -V | tail -1 | grep . || { echo "no checkpoint of $task/$1 under $root/$1" >&2; exit 1; }; }

collect() {
  for arm in "$@"; do
    mkdir -p "$out/$arm"
    checkpoint=$(last "$arm")
    for file in "$root/$arm"/training.jsonl "$root/$arm"/validation.jsonl "$root/$arm"/provenance.*.json "$root/$arm"/channel_gradient.jsonl "$root/$arm.log"; do
      [ -f "$file" ] && cp "$file" "$out/$arm/"
    done
    basename "$checkpoint" > "$out/$arm/final_checkpoint.txt"
    rm -rf "$out/$arm/adapter"
    cp -r "$checkpoint/model" "$out/$arm/adapter"
  done
  for snapshot in "${HF_HOME:-/workspace/hf}"/hub/models--google--*/snapshots/*; do
    echo "$(basename "$(dirname "$(dirname "$snapshot")")") $(basename "$snapshot")"
  done > "$out/model_revisions.txt"
  cp "$data/report.json" "$out/dataset_report.json"
}

if [ "$task" = state_tracking ]; then
  export STATE_TRACKING_TASK=sudoku; [ "$recipe" = recipe1 ] && STATE_TRACKING_TASK=shift_register
  for arm in $arms; do last "$arm" > /dev/null; done
  python - "$out/adapters.json" $(for arm in $arms; do echo "$arm=$(last "$arm")"; done) <<'PY'
import json, sys
json.dump(dict(pair.split('=', 1) for pair in sys.argv[2:]), open(sys.argv[1], 'w', encoding='utf-8'))
PY
  python -m state_tracking.meta.evaluate.evaluate --adapters "$out/adapters.json" --gpus 1 --batch 16 --limit 512 --skip-chain 2>&1 | tee -a "$out/evaluate.log"
  collect $arms
  echo "results in $out"
  exit 0
fi

case $task in
  inscriptions) limit=5000; val_limit=500; ext=json
    score() { python -m inscriptions.meta.evaluate.evaluate "$1" "$3" --limit "$2" "${@:4}"; } ;;
  cryptanalysis) limit=0; val_limit=0; ext=txt
    score() { python -m cryptanalysis.meta.evaluate.score "$1" "$3" --data "$data" "${@:4}"; } ;;
  infilling) limit=0; val_limit=0; ext=json; seeds=; step_check=0; mkdir -p "$out/base"
    [ "${EVAL_FULL:-0}" = 1 ] && { seeds="0 1 2"; step_check=1; }
    full=${EVAL_FULL:-0} python - "$data" "$out/base" <<'PY'
import json, os, random, sys
data, base, full = sys.argv[1], sys.argv[2], os.environ['full'] == '1'
caps = {'test': 1000, 'safim': 600, 'humaneval': 600}
def sample(rows, cap):
    rows = list(rows); random.Random(0).shuffle(rows); return rows if full or not cap else rows[:cap]
with open(f'{base}/test.jsonl', 'w', encoding='utf-8') as out: out.writelines(sample(open(f'{data}/test.jsonl', encoding='utf-8'), caps['test']))
by = {}
for line in open(f'{data}/eval.jsonl', encoding='utf-8'): by.setdefault(json.loads(line)['benchmark'], []).append(line)
kept = {name: sample(lines, caps.get(name, 0)) for name, lines in by.items() if name != 'humanevalplus'}
ids = {json.loads(line)['id'] for line in kept.get('humaneval', [])}
kept['humanevalplus'] = [line for line in by.get('humanevalplus', []) if json.loads(line)['id'] in ids]
with open(f'{base}/eval.jsonl', 'w', encoding='utf-8') as out:
    for name in kept: out.writelines(kept[name])
print('subsets', {name: len(lines) for name, lines in kept.items()}, 'test', 'all' if full else caps['test'])
PY
    base_rows=$out/base/test.jsonl; eval_rows=$out/base/eval.jsonl
    score() { case $3 in */base.*|*.panel.*|*.safim.*|*.humaneval.*) where="--out $out/base" ;; *) where= ;; esac
      if [ "${4:-}" = --against ]; then python -m infilling.meta.evaluate.compare "$1" "$3" "$5" --several; else python -m infilling.meta.evaluate.metrics "$1" "$3" $where "${@:4}"; fi; } ;;
  *) echo "evaluate.sh knows inscriptions, cryptanalysis, infilling and state_tracking"; exit 1 ;;
esac
base_limit=${base_limit:-$limit}
base_rows=${base_rows:-$data/test.jsonl}
eval_rows=${eval_rows:-$data/eval.jsonl}
seeds=${seeds-0 1 2}
step_check=${step_check:-1}
ar_checkpoint=$(last ar)
diffusion_checkpoint=$(last diffusion)

python -m common.generate ar "$data/test.jsonl" --checkpoint "$ar_checkpoint" --limit "$limit" $base_flag --out "$out/ar.test.jsonl"
score test "$limit" "$out/ar.test.jsonl" > "$out/scores.ar.$ext"

python -m common.generate diffusion "$data/test.jsonl" --checkpoint "$diffusion_checkpoint" --limit "$limit" $canvas_flag --out "$out/diffusion.test.jsonl"
score test "$limit" "$out/diffusion.test.jsonl" > "$out/scores.diffusion.$ext"
score test "$limit" "$out/ar.test.jsonl" --against "$out/diffusion.test.jsonl" > "$out/compare.$ext"

for kind in ar diffusion; do
  python -m common.generate "$kind" "$base_rows" --limit "$base_limit" $base_flag --out "$out/base.$kind.test.jsonl"
  score test "$base_limit" "$out/base.$kind.test.jsonl" > "$out/scores.base.$kind.$ext"
done

if [ -f "$data/test_random.jsonl" ]; then
  for arm in ar diffusion; do
    checkpoint=$ar_checkpoint; flag=$base_flag; [ $arm = diffusion ] && { checkpoint=$diffusion_checkpoint; flag=$canvas_flag; }
    python -m common.generate $arm "$data/test_random.jsonl" --checkpoint "$checkpoint" --limit 0 $flag --out "$out/$arm.test_random.jsonl"
    score test_random 0 "$out/$arm.test_random.jsonl" > "$out/scores.$arm.test_random.$ext"
  done
fi

for seed in $seeds; do
  python -m common.generate diffusion "$data/val.jsonl" --checkpoint "$diffusion_checkpoint" --limit "$val_limit" --seed "$seed" $canvas_flag --out "$out/diffusion.val.s$seed.jsonl"
  score val "$val_limit" "$out/diffusion.val.s$seed.jsonl" > "$out/scores.diffusion.val.s$seed.$ext"
done
if [ "$step_check" = 1 ]; then
  for steps in 192 1; do
    python -m common.generate diffusion "$data/val.jsonl" --checkpoint "$diffusion_checkpoint" --limit "$val_limit" --steps $steps $canvas_flag --out "$out/diffusion.val.steps$steps.jsonl"
    score val "$val_limit" "$out/diffusion.val.steps$steps.jsonl" > "$out/scores.diffusion.val.steps$steps.$ext"
  done
fi

if [ "$task" = infilling ]; then
  for benchmark in panel safim humaneval; do
    python -c "import json, sys; sys.stdout.writelines(line for line in open('$eval_rows', encoding='utf-8') if json.loads(line)['benchmark'] == '$benchmark')" > "$out/rows.$benchmark.jsonl"
    python -m common.generate ar "$out/rows.$benchmark.jsonl" --checkpoint "$ar_checkpoint" --out "$out/ar.$benchmark.jsonl"
    python -m common.generate diffusion "$out/rows.$benchmark.jsonl" --checkpoint "$diffusion_checkpoint" --out "$out/diffusion.$benchmark.jsonl"
    for arm in ar diffusion; do
      score eval 0 "$out/$arm.$benchmark.jsonl" --benchmark "$benchmark" > "$out/scores.$arm.$benchmark.json"
      [ "$benchmark" = humaneval ] && score eval 0 "$out/$arm.humaneval.jsonl" --benchmark humanevalplus > "$out/scores.$arm.humanevalplus.json"
    done
    python -m infilling.meta.evaluate.compare eval "$out/ar.$benchmark.jsonl" "$out/diffusion.$benchmark.jsonl" --benchmark "$benchmark" --out "$(dirname "$eval_rows")" > "$out/compare.$benchmark.json"
    rm "$out/rows.$benchmark.jsonl"
  done
fi

collect ar diffusion
echo "results in $out"