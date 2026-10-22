'''
Scores the ladder of interventions on chain {the nocot-bench panel, one number in 1..20 as state} against the
unmodified run {same rows, batches and noise}, by exact match in percentage points, a bootstrap over episodes and the
thresholds of settings:
    use           U = EM(full) - EM(swap) per rung; informative when U >= max(3 mde, 12) and its interval is above 0
    sufficiency   D = EM(full) - EM(op) pooled over informative rungs, op in {top 8, resample 8, resample 1};
                  P1 on resample 8: sufficient when the interval lies in [-4, 4], insufficient when above 0 and D > 4
    interchange   for base a, donor b and step j the target y = walk(v_b^(j), steps_a[j:]), the donor's value after j
                  steps carried through the base's remaining steps; Pi_j = agreement with y minus agreement with the
                  same target built from a non-donor c; P2 = Pi at the step chosen on val, pooled over informative
                  rungs with k > j; carried when >= 10
The state has 20 values, so the top-8 reading at the answer digits covers much of it: this ladder tests use, hidden
content {resample 8} and interchange on a published task, not the width of the reading, which shift_register tests.
    python -m state_tracking.meta.interpretability.chain_ladder test diffusion_gradient.T2.chain
Reference: https://arxiv.org/abs/2301.04709 {interchange intervention accuracy}
Reference: https://github.com/neelnanda-io/nocot-bench {chain}
'''

import argparse
import json
from collections import defaultdict

import numpy as np

from common.files import read_jsonl
from state_tracking.meta.benchmark.chain import walk
from state_tracking.meta.benchmark.nocot import mark, reply_form
from state_tracking.meta.config.settings import carried_floor, data_dir, gate, informative_floor, margin, runs_dir
from state_tracking.meta.evaluate.stats import above, summary, within
from state_tracking.meta.interpretability.ladder import tag

machine = 'chain'
interventions = ('full', 'swap', 'top 8', 'resample 8', 'resample 1')
operations = interventions[1:]
interchanged = ('full', 'swap', 'resample 8')

def well_formed(reply):
    return bool(reply_form.search(reply))

def said(reply):
    found = reply_form.search(reply)

    return int(found.group(1)) if found else None

def targets(row, donor):
    truth, other = json.loads(row['truth']), json.loads(donor['truth'])
    path = [other['start'], *other['stages']]

    return [(walk(path[j], truth['steps'][j:]) or [path[j]])[-1] for j in range(row['horizon'] + 1)]

def agree(reply, target):
    return float(said(reply) == target)

def load(split, run):
    found, partners = {}, {}

    for kind in interventions:
        path = runs_dir / f'{split}.diffusion.{run}{tag(kind)}.jsonl'

        if path.exists():
            found[kind] = {}

            for reply in read_jsonl(path):
                found[kind][reply['id']] = reply['reply']
                partners.setdefault(reply['id'], reply['partner'])

    rows = [row for row in read_jsonl(data_dir / f'{split}.jsonl') if row['machine'] == machine and all(row['id'] in replies for replies in found.values())]

    return rows, found, partners

def interchange(rows, found, partners):
    by_id, table = {row['id']: row for row in rows}, {}

    for row in rows:
        donor = by_id.get(partners.get(row['id']))
        control = by_id.get(partners.get(donor['id'])) if donor else None

        if donor is None or control is None or control['id'] == row['id']:
            continue

        to_donor, to_control = targets(row, donor), targets(row, control)
        table[row['id']] = {kind: [agree(found[kind][row['id']], a) - agree(found[kind][row['id']], b) for a, b in zip(to_donor, to_control)]
                            for kind in interchanged if kind in found}

    return table

def pooled(table, rows, kind, j):
    return [table[row['id']][kind][j] for row in rows if row['id'] in table and kind in table[row['id']] and 0 < j < row['horizon']]

def choose(table, rows):
    scores = {j: np.mean(values) for j in range(1, max((row['horizon'] for row in rows), default=1)) if (values := pooled(table, rows, 'swap', j))}

    return max(scores, key=scores.get) if scores else None

def accuracy(found, kind, rows):
    return [float(mark(found[kind][row['id']], row)['exact']) for row in rows]

def compare(rows, found, partners, chosen=None):
    by_rung, result = defaultdict(list), {'rungs': {}}

    for row in rows:
        by_rung[row['horizon']].append(row)

    informative = []

    for horizon, group in sorted(by_rung.items()):
        full = accuracy(found, 'full', group)
        cell = {'full': summary(full)}

        for kind in operations:
            if kind in found:
                cell[kind] = summary(np.subtract(full, accuracy(found, kind, group)))

        if 'swap' in cell:
            cell['informative'] = above(cell['swap'], max(gate * (cell['swap']['mde'] or 0), informative_floor))
            informative += [horizon] * cell['informative']

        cell['well_formed'] = {kind: round(100 * np.mean([well_formed(found[kind][row['id']]) for row in group]), 1) for kind in found}
        result['rungs'][horizon] = cell

    used = [row for row in rows if row['horizon'] in informative]
    result['informative'] = informative
    result['sufficiency'] = {kind: summary(np.subtract(accuracy(found, 'full', used), accuracy(found, kind, used))) for kind in operations if kind in found and kind != 'swap'}
    resample = result['sufficiency'].get('resample 8')
    result['P1'] = None if not resample or not used else 'sufficient' if within(resample, margin) else 'insufficient' if above(resample, margin) else 'undetermined'
    table = interchange(rows, found, partners)
    result['interchange'] = {horizon: {kind: np.round(100 * np.mean([table[row['id']][kind] for row in group if row['id'] in table], 0), 1).tolist()
                                       for kind in interchanged if kind in found} for horizon, group in sorted(by_rung.items())}
    result['round'] = chosen

    if chosen is not None and used:
        result['P2'] = {kind: summary(pooled(table, used, kind, chosen)) for kind in ('swap', 'resample 8') if kind in found}
        result['carried'] = above(result['P2'].get('swap', summary([])), carried_floor)
        result['carried_outside_reading'] = above(result['P2'].get('resample 8', summary([])), carried_floor)

    return result, table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('run')
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    rows, found, partners = load(args.split, args.run)
    result, table = compare(rows, found, partners, args.round)

    if args.round is None:
        result['round_on_this_split'] = choose(table, rows)

    print(json.dumps(result, indent=1))