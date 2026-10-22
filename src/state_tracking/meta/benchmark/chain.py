'''
The chain panel: the `chain` bank of nocot-bench, generated at training scale by the unmodified vendored code.
State v in {1..20}; each step is ceil(v/2), or v/2 if even else v + a, or v - b if v > 10 else 2v, then wrapped into
1..20; the answer is v after k steps. Also gives the floor c_k: the exact-match rate of the best guess that ignores
the start.
Reference: https://github.com/neelnanda-io/nocot-bench/blob/main/datagen/banks/chain.py
Reference: https://github.com/neelnanda-io/nocot-bench/blob/main/DOMAINS.md {section 1, chain}
'''

import json
from collections import Counter

from state_tracking.meta.benchmark import nocot
from state_tracking.meta.benchmark.nocot import gold
from state_tracking.meta.config.settings import chain_horizons, chain_rungs

machine = 'chain'
source = nocot.chain
config = source.SHIPPED

def walk(start, steps):
    value, stages = start, []

    for step in steps:
        value = source._apply(value, tuple(step), config.mod)
        stages.append(value)

    return stages

def make(horizon, rng, made, held=False):
    case = source._gen(rng, horizon, config)
    row = nocot.row(machine, horizon, f'{source.INSTRUCTION}\n\n{source._render(case, config.mod)}', case['answer'])

    if held:
        row['truth'] = json.dumps({'start': case['start'], 'steps': [list(step) for step in case['ops']], 'stages': walk(case['start'], case['ops'])})

    return row

def rows(count, seed):
    return nocot.rows(make, chain_horizons, count, seed)

def panel(each, seed, split):
    return nocot.panel(make, chain_rungs, each, seed, split)

def blind_floor(rows):
    hits = {}

    for row in rows:
        steps = nocot.truth(row)['steps']
        finals = Counter(walk(start, steps)[-1] for start in range(1, config.mod + 1))
        guess = min(finals, key=lambda value: (-finals[value], value))
        hits.setdefault(row['horizon'], []).append(guess == gold(row))

    return {horizon: sum(found) / len(found) for horizon, found in sorted(hits.items())}