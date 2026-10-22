'''
Builds dataset/state_tracking/<task> {train, val, test .jsonl and report.json} for the carrier task of settings {STATE_TRACKING_TASK}.
sudoku: 32,000 training rows balanced over five depth bands, 250 val and 1,024 test rows per band plus the not-singles
tier, and the nocot-bench chain and config_patch panels in val and test only {val.sudoku.jsonl holds the Sudoku val rows alone, the file NeMo's validation loss reads}. shift_register: the four training panels of
the sub-experiment {shift_register, shift_register_single, chain, config_patch; 96,000 rows}. Fixed seeds per panel and split; report.json
holds the settings, row counts, token lengths, constant-reply floors, depth histograms and SHA-256 hashes.
    python -m state_tracking.meta.benchmark.build
'''

import json
import sys
from collections import Counter

from transformers import AutoTokenizer

from common.files import seal, write_jsonl
from common.models import gemma
from state_tracking.meta.benchmark import chain, config_patch, nocot, shift_register, sudoku
from state_tracking.meta.config import settings
from state_tracking.meta.config.settings import budget, data_dir, each, few, task

panels = (chain, config_patch)

def longest(rows, encoder):
    most = 0

    for first in range(0, len(rows), 512):
        group = rows[first:first + 512]

        for row, tokens in zip(group, encoder([row['messages'][1]['content'] for row in group], add_special_tokens=False)['input_ids']):
            assert row['machine'] not in shift_register.machines or len(tokens) == settings.width, row['id']
            assert row['machine'] not in sudoku.machines or len(tokens) == sudoku.stride * sudoku.width - 1, row['id']
            most = max(most, len(tokens))

    return most

def shift_register_rows(seed):
    return {'train': shift_register.rows(budget['shift_register'], budget['shift_register_single'], seed + 4) + chain.rows(budget['chain'], seed + 7) + config_patch.rows(budget['config_patch'], seed + 10),
            'val': shift_register.panel(few, seed + 5, 'val') + chain.panel(few, seed + 8, 'val') + config_patch.panel(few, seed + 11, 'val'),
            'test': shift_register.panel(each, seed + 6, 'test') + chain.panel(each, seed + 9, 'test') + config_patch.panel(each, seed + 12, 'test')}

def sudoku_rows(seed, log=None):
    built = sudoku.rows({'train': settings.sudoku_train, 'val': few * sudoku.hard_bucket, 'test': each * sudoku.hard_bucket}, seed + 4, held=True, log=log)

    return {'train': built['train'],
            'val': built['val'] + chain.panel(settings.nocot_few, seed + 8, 'val') + config_patch.panel(settings.nocot_few, seed + 11, 'val'),
            'test': built['test'] + chain.panel(settings.nocot_each, seed + 9, 'test') + config_patch.panel(settings.nocot_each, seed + 12, 'test')}

def depths(rows):
    found = Counter(row['horizon'] if row['machine'] == 'sudoku' else 'search' for row in rows if row['machine'] in sudoku.machines)

    return {str(k): v for k, v in sorted(found.items(), key=lambda item: (isinstance(item[0], str), item[0]))}

def bands(rows):
    found = Counter(f"band {sudoku.bucket({'machine': row['machine'], 'horizon': row['horizon']})}" if row['machine'] == 'sudoku' else 'search' for row in rows if row['machine'] in sudoku.machines)

    return dict(sorted(found.items()))

def sources(rows):
    return dict(sorted(Counter(row['origin'].split(':')[1] for row in rows if row['machine'] in sudoku.machines).items()))

def givens(rows):
    found = Counter(('>= 40' if row['rung'] >= 40 else '30-39' if row['rung'] >= 30 else '< 30') for row in rows if row['machine'] in sudoku.machines)

    return dict(sorted(found.items()))

def build(seed=0, out=data_dir, log=sys.stderr):
    encoder = AutoTokenizer.from_pretrained(gemma)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    built = sudoku_rows(seed, log) if task == 'sudoku' else shift_register_rows(seed)
    counts = {}

    for name, rows in built.items():
        write_jsonl(out / f'{name}.jsonl', rows)

        if task == 'sudoku' and name == 'val':
            write_jsonl(out / 'val.sudoku.jsonl', [row for row in rows if row['machine'] in sudoku.machines])
        counts[name] = {'rows': dict(Counter(row['machine'] for row in rows)), 'longest answer in tokens': longest(rows, encoder)}

        if task == 'sudoku':
            counts[name]['rows by depth'] = depths(rows)
            counts[name]['rows by band'] = bands(rows)
            counts[name]['rows by source'] = sources(rows)
            counts[name]['rows by givens bin'] = givens(rows)

        for panel in panels if name != 'train' else ():
            held = [row for row in rows if row['machine'] == panel.machine]
            counts[name][f'{panel.machine} floor by rung'] = {horizon: round(share, 3) for horizon, share in nocot.floor(held).items()}

    split = 'MD5(prompt) mod 8: val = 0, test = 1, train = 2..7'
    design = {'task': task, 'seed': seed, 'held_out_each': each, 'held_out_few': few, 'gold_share': settings.gold_share,
              'chain': {'source': 'nocot-bench datagen/banks/chain.py', 'commit': nocot.commit, 'modulus': chain.config.mod, 'evaluation_only': task == 'sudoku',
                        'horizons': [settings.chain_horizons[0], settings.chain_horizons[-1]], 'rungs': settings.chain_rungs, 'split': split},
              'config_patch': {'source': 'nocot-bench datagen/banks/cfgpatch.py', 'commit': nocot.commit, 'vmax': config_patch.config.vmax, 'n_distract': config_patch.config.n_distract,
                               'evaluation_only': task == 'sudoku', 'horizons': [settings.config_horizons[0], settings.config_horizons[-1]], 'rungs': settings.config_rungs, 'split': split}}

    if task == 'sudoku':
        design['sudoku'] = {'source': 'Ritvik19/Sudoku-Dataset train_0.parquet, sets 9m and 3m (mirrors of Kaggle rohanrao/sudoku and radcliffe/3-million-sudoku-puzzles-with-ratings, CC0)',
                            'depth_bands': settings.sudoku_bands, 'not_singles_tier': 'val and test only',
                            'train_rows': settings.sudoku_train, 'instruction': sudoku.instruction, 'split': 'MD5(relabelled solution) mod 8: val = 0, test = 1, train = 2..7',
                            'answer_tokens': sudoku.stride * sudoku.width - 1, 'nocot_evaluation_only': True}

    else:
        design['budget'] = budget
        design['shift_register'] = {'machines': shift_register.machines, 'width': settings.width, 'horizons': [settings.shift_register_horizons[0], settings.shift_register_horizons[-1]],
                             'rungs': settings.shift_register_rungs, 'letters': settings.letters, 'split': 'MD5(start) mod 8: val = 0, test = 1, train = 2..7', 'paired_machines': True}

    return seal(out, settings=design, **counts)

if __name__ == '__main__':
    print(json.dumps(build(), indent=1))