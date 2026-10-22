#!/usr/bin/env bash
set -uo pipefail

python - "${ALPHA_CHECKPOINTS:-/workspace/checkpoints}" <<'PY'
import glob, json, os, re, sys
from datetime import datetime

root = sys.argv[1]
runs = sorted({os.path.dirname(path) for path in glob.glob(f'{root}/**/provenance.*.json', recursive=True)})
print(f"{'run':44} {'step':>13} {'loss':>8} {'lr':>9} {'tps':>7} {'val_loss':>9} {'s/step':>7} {'left':>7} {'ckpts':>5}")

def last_match(path, pattern):
    found = None
    if os.path.exists(path):
        with open(path, encoding='utf-8', errors='replace') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    found = match
    return found

def seconds_per_step(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path, encoding='utf-8')][-50:]
    if len(rows) < 2:
        return None
    span = datetime.fromisoformat(rows[-1]['timestamp']) - datetime.fromisoformat(rows[0]['timestamp'])
    return span.total_seconds() / (rows[-1]['step'] - rows[0]['step'])

for run in runs:
    config = json.load(open(sorted(glob.glob(f'{run}/provenance.*.json'))[-1], encoding='utf-8'))['config']
    steps = config['step_scheduler']['max_steps']
    train = last_match(f'{run}.log', r'step (\d+) .*?\| loss ([\d.]+) .*?\| lr ([\d.e+-]+) .*?\| tps ([\d.]+)')
    val = last_match(f'{run}.log', r'step \d+ \| epoch \d+ \| loss ([\d.]+) \| lr [^|]*\| num_label_tokens')
    pace = seconds_per_step(f'{run}/training.jsonl')
    step = int(train.group(1)) if train else 0
    left = f'{(steps - step) * pace / 3600:.1f}h' if pace else '-'
    checkpoints = len(glob.glob(f'{run}/epoch_*_step_*'))
    print(f"{os.path.relpath(run, root):44} {step:>6}/{steps:<6} {train.group(2) if train else '-':>8} {train.group(3) if train else '-':>9} "
          f"{train.group(4) if train else '-':>7} {val.group(1) if val else '-':>9} {f'{pace:.1f}' if pace else '-':>7} {left:>7} {checkpoints:>5}")
PY

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true