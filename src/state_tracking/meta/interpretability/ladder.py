'''
Scores a ladder of interventions on the carrier against the unmodified run {same rows, batches and noise}, by symbol
accuracy {shift_register: letters; sudoku: blank cells} in percentage points, a bootstrap over episodes and the thresholds of settings:
    use           U = A(full) - A(swap) per rung {another episode's state: whether the step reads S at all}, or A(full) - A(zero)
                  where a ladder has no swap; informative when U >= max(3 mde, 12) and its interval is above 0. The zero drop is
                  reported beside it as need {what S adds over the canvas}; zero does not gate, because S = 0 is out of
                  distribution for p = 1 arms and S duplicates much of the canvas; churn = share of reply symbols that change
                  under each operation, reported beside every delta
    sufficiency   D = A(full) - A(op) pooled over informative rungs, op in the interventions of settings; P1 on the decisive
                  reading {sudoku: resample 3; shift_register: resample 8}: sufficient when the interval lies in [-4, 4],
                  insufficient when above 0 and D > 4; the smallest sufficient reading is reported with it; ungated, the same
                  pooled over every rung, for ladders whose use test has no informative rung
    interchange   shift_register: for base a, donor b and round j the target y = F_{j+1..k}(x_b^(j); key_a, inputs_a); Pi_j = agreement
                  of the swap reply with y minus agreement with the same target built from a non-donor c of the cell.
                  sudoku: the edit j run hands a state with one cell of depth j set to another admissible digit; Pi_j =
                  agreement of that reply with the recursion's prediction y on the cells the edit changes minus its agreement
                  with x* there, with the same statistic on the un-intervened reply as the baseline. P2 = Pi at the round
                  chosen on val; carried when >= 10 with an interval above 0. Shift = Pi(edit) - Pi(baseline) per episode at
                  that round {the causal effect of the edit; the prompt still fixes x*, so Pi itself rarely turns positive};
                  moved when the shift is >= 10 with an interval above 0.
Also per operation the share of well-formed replies and, for resample, the share of mass placed by headroom where the cap binds.
    python -m state_tracking.meta.interpretability.ladder test diffusion_gradient.T2.ladder
Reference: https://arxiv.org/abs/2301.04709 {interchange intervention accuracy}
Reference: https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y/p/JvZhhzycHu2Yd57RN {resampling ablation}
'''

import argparse
import json
from collections import defaultdict

import numpy as np

from common.files import read_jsonl
from state_tracking.meta.benchmark import carrier
from state_tracking.meta.config.settings import carried_floor, data_dir, decisive, gate, informative_floor, interventions, margin, runs_dir
from state_tracking.meta.evaluate.stats import above, summary, within

operations = tuple(kind for kind in interventions if kind != 'full')
readings = tuple(kind for kind in operations if kind.startswith(('top', 'resample', 'quantise')))
edits = tuple(kind for kind in operations if kind.startswith('edit'))

def tag(kind):
    return '' if kind == 'full' else '.' + kind.replace(' ', '')

def agree(reply, target, positions):
    guess = carrier.symbols(reply)

    return np.mean([i < len(guess) and guess[i] == target[i] for i in positions]) if positions else np.nan

def load(split, run):
    found, partners, spread = {}, {}, {}

    for kind in interventions:
        path = runs_dir / f'{split}.diffusion.{run}{tag(kind)}.jsonl'

        if path.exists():
            found[kind] = {}

            for reply in read_jsonl(path):
                found[kind][reply['id']] = reply['reply']
                partners.setdefault(reply['id'], reply['partner'])

                if 'spread' in reply:
                    spread.setdefault(kind, []).append(reply['spread'])

    rows = [row for row in read_jsonl(data_dir / f'{split}.jsonl') if row['machine'] == carrier.machine and all(row['id'] in replies for replies in found.values())]

    return rows, found, partners, spread

def interchange_shift_register(rows, found, partners):
    by_id, table = {row['id']: row for row in rows}, {}

    for row in rows:
        donor = by_id.get(partners.get(row['id']))
        control = by_id.get(partners.get(donor['id'])) if donor else None

        if donor is None or control is None or control['id'] == row['id']:
            continue

        to_donor, to_control = carrier.interchange_targets(row, donor), carrier.interchange_targets(row, control)
        everything = list(range(carrier.width))
        table[row['id']] = {kind: [agree(found[kind][row['id']], a, everything) - agree(found[kind][row['id']], b, everything) for a, b in zip(to_donor, to_control)]
                            for kind in ('full', 'swap', decisive) if kind in found}

    return table

def interchange_sudoku(rows, found):
    table = {}

    for row in rows:
        planned, truth = carrier.edits(row), carrier.truth_of(row)
        values = {}

        for kind in edits:
            j = int(kind.split()[1])

            if kind in found and j in planned:
                edit = planned[j]

                for source, name in ((kind, kind), ('full', f'baseline {j}')):
                    values[name] = agree(found[source][row['id']], edit['target'], edit['changed']) - agree(found[source][row['id']], truth['solution'], edit['changed'])

        if values:
            table[row['id']] = values

    return table

def churn(found, kind, rows):
    pairs = [(carrier.symbols(found['full'][row['id']]), carrier.symbols(found[kind][row['id']])) for row in rows]

    return [sum(a != b for a, b in zip(full, other)) / carrier.width if full and other else 1.0 for full, other in pairs]

def interchange(rows, found, partners):
    return interchange_sudoku(rows, found) if carrier.task == 'sudoku' else interchange_shift_register(rows, found, partners)

def pooled(table, rows, kind, j):
    if carrier.task == 'sudoku':
        return [table[row['id']][f'edit {j}'] for row in rows if row['id'] in table and f'edit {j}' in table[row['id']]]

    return [table[row['id']][kind][j] for row in rows if row['id'] in table and kind in table[row['id']] and 0 < j < row['horizon']]

def choose(table, rows):
    scores = {j: np.mean(values) for j in range(1, max((row['horizon'] for row in rows), default=1) + 1) if (values := pooled(table, rows, 'swap', j))}

    return max(scores, key=scores.get) if scores else None

def accuracy(found, kind, rows):
    return [carrier.mark(found[kind][row['id']], row)['symbol'] for row in rows]

def compare(rows, found, partners, spread, chosen=None):
    by_rung, result = defaultdict(list), {'rungs': {}}

    for row in rows:
        by_rung[row['horizon']].append(row)

    informative = []

    for horizon, group in sorted(by_rung.items()):
        full = accuracy(found, 'full', group)
        cell = {'full': summary(full)}

        for kind in operations:
            if kind in found and not kind.startswith('edit'):
                cell[kind] = summary(np.subtract(full, accuracy(found, kind, group)))

        cell['churn'] = {kind: round(100 * float(np.mean(churn(found, kind, group))), 1) for kind in operations if kind in found}
        use = 'swap' if 'swap' in cell else 'zero'

        if use in cell:
            cell['use'] = use
            cell['need'] = cell.get('zero')
            cell['informative'] = above(cell[use], max(gate * (cell[use]['mde'] or 0), informative_floor))
            informative += [horizon] * cell['informative']

        cell['well_formed'] = {kind: round(100 * np.mean([carrier.well_formed(found[kind][row['id']]) for row in group]), 1) for kind in found}
        result['rungs'][horizon] = cell

    used = [row for row in rows if row['horizon'] in informative]
    result['informative'] = informative
    result['sufficiency'] = {kind: summary(np.subtract(accuracy(found, 'full', used), accuracy(found, kind, used))) for kind in readings if kind in found}
    result['sufficiency_ungated'] = {kind: summary(np.subtract(accuracy(found, 'full', rows), accuracy(found, kind, rows))) for kind in readings if kind in found}
    result['spread'] = {kind: round(100 * float(np.mean(values)), 2) for kind, values in spread.items()}
    verdict = lambda found_: 'sufficient' if within(found_, margin) else 'insufficient' if above(found_, margin) else 'undetermined'
    result['verdicts'] = {kind: verdict(found_) for kind, found_ in result['sufficiency'].items()} if used else {}
    result['P1'] = result['verdicts'].get(decisive)
    result['smallest_sufficient'] = next((kind for kind in readings if result['verdicts'].get(kind) == 'sufficient'), None)
    result['rank_detected'] = bool(result['sufficiency'].get('resample 1') and above(result['sufficiency']['resample 1'], 0))
    table = interchange(rows, found, partners)
    result['round'] = chosen

    if carrier.task == 'sudoku':
        names = [name for kind in edits if kind in found for name in (kind, f"baseline {kind.split()[1]}")]
        result['interchange'] = {name: summary([table[row['id']][name] for row in rows if row['id'] in table and name in table[row['id']]]) for name in names}

        if chosen is not None and used:
            result['P2'] = summary(pooled(table, used, None, chosen))
            result['carried'] = above(result['P2'], carried_floor)
            result['shift'] = summary([table[row['id']][f'edit {chosen}'] - table[row['id']][f'baseline {chosen}'] for row in used
                                       if f'edit {chosen}' in table.get(row['id'], {})])
            result['moved'] = above(result['shift'], carried_floor)
            result['carried_outside_reading'] = None

    else:
        result['interchange'] = {horizon: {kind: np.round(100 * np.mean([table[row['id']][kind] for row in group if row['id'] in table], 0), 1).tolist()
                                           for kind in ('full', 'swap', decisive) if kind in found} for horizon, group in sorted(by_rung.items())}

        if chosen is not None and used:
            result['P2'] = {kind: summary(pooled(table, used, kind, chosen)) for kind in ('swap', decisive) if kind in found}
            result['carried'] = above(result['P2'].get('swap', summary([])), carried_floor)
            result['carried_outside_reading'] = above(result['P2'].get(decisive, summary([])), carried_floor)

    return result, table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('run')
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    rows, found, partners, spread = load(args.split, args.run)
    result, table = compare(rows, found, partners, spread, args.round)

    if args.round is None:
        result['round_on_this_split'] = choose(table, rows)

    print(json.dumps(result, indent=1))