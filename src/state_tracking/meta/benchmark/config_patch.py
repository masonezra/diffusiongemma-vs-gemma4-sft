'''
The config_patch panel: the `cfgpatch` bank of nocot-bench, generated at training scale by the unmodified vendored code.
Six `key = integer` lines and h + 7 numbered patches: h carry one value along a chain of distinct values in 1..200 and
6 are distractors; the answer is the final value of the queried key. Items whose rendered text does not re-solve to
the generator's answer are dropped.
Reference: https://github.com/neelnanda-io/nocot-bench/blob/main/datagen/banks/cfgpatch.py
'''

import json

from state_tracking.meta.benchmark import nocot
from state_tracking.meta.config.settings import config_horizons, config_rungs

machine = 'config_patch'
source = nocot.cfgpatch
config = source.SHIPPED

def make(horizon, rng, made, held=False):
    targets = source._cond_targets(horizon)
    case = source.gen_item(rng, horizon, targets[made % len(targets)], config)
    problem = source.render_numbered(case)

    if source.resimulate(problem) != case['answer']:
        return None

    row = nocot.row(machine, horizon, f'{source.INSTRUCTION}\n\n{problem}', case['answer'])

    if held:
        row['truth'] = json.dumps({'config': case['cfg0'], 'steps': [list(step) for step in case['seq']], 'key': case['qkey']})

    return row

def rows(count, seed):
    return nocot.rows(make, config_horizons, count, seed)

def panel(each, seed, split):
    return nocot.panel(make, config_rungs, each, seed, split)