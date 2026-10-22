'''
Legibility of the reading handed between denoising steps, against the ground-truth path x^(0), ..., x^(k) of the
carrier {shift_register: the start row and the rows after each round; sudoku: the puzzle and the grid after each propagation
round}. For step s, round j and the reply positions the carrier scores at round j {sudoku: the cells filled exactly at
round j; shift_register: the positions where x^(j) differs from the start and final rows}:
    L_s(j) = P[top1_s,i = x^(j)_i]
    chance  the same with x^(j) of the donor episode {same cell}, so coincidences and symbol frequencies cancel
    reading the same with top1 replaced by membership in the recorded top 8; canvas the same for the sampled canvas;
    candidates {sudoku} the overlap of the top-|C| tokens with the solver's candidate set C_i(x^(j-1)) of the cell
and the shares of top1 that are the final symbol, the start symbol, another symbol or not a symbol. Step 0 of a T = 2
run is the reading step 2 receives. The round j is chosen on val as the largest pooled L - chance over 0 < j <= k.
    python -m state_tracking.meta.interpretability.legibility test diffusion_gradient.T2.ladder
Reference: https://arxiv.org/pdf/2606.20560 {Sections 3.2 and 5.3}
'''

import argparse
import json
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from common.files import read_jsonl
from common.models import diffusion_gemma
from state_tracking.meta.benchmark import carrier
from state_tracking.meta.config.settings import data_dir, legible_floor, runs_dir
from state_tracking.meta.evaluate.stats import above, summary

views = ('top1', 'chance', 'reading', 'canvas', 'candidates')
shares = ('final', 'start', 'letter', 'other')

def letter_ids(tokenizer):
    found = {}

    for form in carrier.forms:
        ids = tokenizer(form, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, form
        found[ids[0]] = form.strip()

    return found

def alphabet_ids(tokenizer):
    letters = letter_ids(tokenizer)

    return [token for symbol in carrier.alphabet for token, found in letters.items() if found == symbol]

def locate(final, letters, size=None, stride=None):
    size, stride = size or carrier.width, stride or carrier.stride
    read = [letters.get(int(token)) for token in final]

    for first in range(len(read) - stride * (size - 1)):
        if all(read[first + stride * i] for i in range(size)):
            return first

    return None

def agreement(guess, target, positions):
    return np.mean([target[i] in guess[i] for i in positions]) if positions else np.nan

def observe(trace, first, row, other, letters):
    path, donor = carrier.path_of(row), carrier.path_of(other)
    where = first + carrier.stride * np.arange(carrier.width)
    ids, canvas = trace['ids'][:, where], trace['canvas'][:, where]
    found = {view: np.full((ids.shape[0], len(path)), np.nan) for view in views}
    found['shares'] = np.zeros((ids.shape[0], len(shares)))

    sets = carrier.truth_of(row).get('candidates', []) if carrier.task == 'sudoku' else []

    for step in range(ids.shape[0]):
        best = [{letters.get(int(ids[step, i, 0]))} for i in range(carrier.width)]
        reading = [{letters.get(int(token)) for token in ids[step, i]} for i in range(carrier.width)]
        written = [{letters.get(int(token))} for token in canvas[step]]

        for j in range(len(path)):
            positions = carrier.scored_positions(row, path, j)
            found['top1'][step, j] = agreement(best, path[j], positions)
            found['chance'][step, j] = agreement(best, donor[j], [i for i in positions if donor[j][i] in carrier.alphabet]) if j < len(donor) else np.nan
            found['reading'][step, j] = agreement(reading, path[j], positions)
            found['canvas'][step, j] = agreement(written, path[j], positions)

            if 0 < j <= len(sets):
                overlaps = []

                for i in positions:
                    candidates = sets[j - 1].get(str(i), '')
                    top = [letters.get(int(token)) for token in ids[step, i, :len(candidates)]]
                    overlaps.append(sum(symbol is not None and symbol in candidates for symbol in top) / len(candidates) if candidates else np.nan)

                found['candidates'][step, j] = np.nanmean(overlaps) if overlaps and not all(np.isnan(overlaps)) else np.nan

        for i in range(carrier.width):
            top = next(iter(best[i]))
            kind = 0 if top == path[-1][i] else 1 if top == path[0][i] else 2 if top else 3
            found['shares'][step, kind] += 1 / carrier.width

    return found

def measure(rows, run, letters):
    by_id, partners = {row['id']: row for row in rows}, {reply['id']: reply['partner'] for reply in read_jsonl(runs_dir / f'{run.name}.jsonl')}
    found, unaligned = {}, defaultdict(int)

    for row in rows:
        path = run / f"{row['id']}.npz"

        if path.exists() and partners.get(row['id']) in by_id:
            with np.load(path) as trace:
                first = locate(trace['final'], letters)

                if first is None:
                    unaligned[row['horizon']] += 1

                else:
                    found[row['id']] = observe(trace, first, row, by_id[partners[row['id']]], letters)

    return found, unaligned

def gap(found, rows, j, step=0):
    return [found[row['id']]['top1'][step, j] - found[row['id']]['chance'][step, j] for row in rows if row['id'] in found and 0 < j <= row['horizon']]

def choose(found, rows, step=0):
    scores = {j: np.nanmean(values) for j in range(1, max((row['horizon'] for row in rows), default=0) + 1) if (values := gap(found, rows, j, step))}

    return max(scores, key=scores.get) if scores else None

def summarise(found, unaligned, rows, chosen=None, step=0):
    by_rung = defaultdict(list)

    for row in rows:
        if row['id'] in found:
            by_rung[row['horizon']].append(found[row['id']])

    table = {}

    for horizon, items in sorted(by_rung.items()):
        table[horizon] = {'episodes': len(items), 'unaligned': unaligned.get(horizon, 0)}
        longest = max(item['top1'].shape[1] for item in items)

        for view in views:
            padded = [np.pad(item[view], ((0, 0), (0, longest - item[view].shape[1])), constant_values=np.nan) for item in items]

            with np.errstate(all='ignore'):
                table[horizon][view] = np.round(100 * np.nanmean(padded, 0), 1).tolist()

        table[horizon]['shares'] = np.round(100 * np.mean([item['shares'] for item in items], 0), 1).tolist()

    result = {'rungs': table, 'round': chosen}

    if chosen is not None:
        result['legible'] = summary(gap(found, rows, chosen, step))
        result['state_legible'] = above(result['legible'], legible_floor)

    return result

def load(split, run, rows=None):
    letters = letter_ids(AutoTokenizer.from_pretrained(diffusion_gemma))
    rows = rows or [row for row in read_jsonl(data_dir / f'{split}.jsonl') if row['machine'] == carrier.machine]

    return (*measure(rows, runs_dir / f'{split}.diffusion.{run}', letters), rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('run')
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    found, unaligned, rows = load(args.split, args.run)
    result = summarise(found, unaligned, rows, args.round if args.round is not None else choose(found, rows))
    print(json.dumps(result, indent=1))