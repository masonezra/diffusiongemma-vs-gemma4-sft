'''
The carrier task of the benchmark, chosen by settings.task {sudoku or shift_register}, behind one interface the evaluation and
interpretability code uses: machine names, the reply width and token stride {shift_register 16 letter tokens at stride 1; sudoku 81
digit tokens at stride 2 with separators between}, the alphabet and its token forms, mark, cell, the ground-truth path
x^(0..k) of a held-out row, the per-position depth D(i), the positions a legibility round j is scored on, the interchange
targets {shift_register: the donor's round-j row advanced under the base episode's key; sudoku: the counterfactual edit of one cell of
depth j and the cells the recursion then changes}, the learned criterion of M0, and the candidate sets of the quantised reading.
'''

import json
import random

from state_tracking.meta.benchmark import shift_register, sudoku
from state_tracking.meta.config import settings

task = settings.task
module = sudoku if task == 'sudoku' else shift_register
machine = 'sudoku' if task == 'sudoku' else 'shift_register'
machines = module.machines
width = sudoku.width if task == 'sudoku' else settings.width
stride = sudoku.stride if task == 'sudoku' else 1
alphabet = sudoku.digits if task == 'sudoku' else shift_register.nibbles
forms = sudoku.forms if task == 'sudoku' else tuple(form for letter in shift_register.nibbles for form in (letter, ' ' + letter))
rungs = settings.sudoku_horizons if task == 'sudoku' else settings.shift_register_rungs
mark = module.mark
cell = module.cell

def truth_of(row):
    return json.loads(row['truth'])

def path_of(row):
    truth = truth_of(row)

    return [truth['puzzle'], *truth['stages']] if task == 'sudoku' else [truth['start'], *truth['stages']]

def depth_of(row):
    if task == 'sudoku':
        return truth_of(row)['depth_map']

    path = path_of(row)

    return [next((j for j in range(1, len(path)) if all(path[t][i] == path[-1][i] for t in range(j, len(path)))), 0) for i in range(width)]

def symbols(text):
    return text.split() if task == 'shift_register' else [ch for ch in text if ch.isdigit()]

def well_formed(reply):
    found = symbols(reply)

    return len(found) == width and all(symbol in alphabet for symbol in found)

def givens(row):
    return [int(ch) for ch in row['messages'][0]['content'].split('\n\n')[-1].split()] if task == 'sudoku' else [0] * width

def blanks(row):
    return [i for i, v in enumerate(givens(row)) if v == 0]

def scored_positions(row, path, j):
    if task == 'sudoku':
        depth = depth_of(row)

        return [i for i in range(width) if depth[i] == j]

    anchors = {0, len(path) - 1} - {j}

    return [i for i in range(width) if all(path[j][i] != path[a][i] for a in anchors)]

def candidates(row):
    if task != 'sudoku':
        return None

    truth = truth_of(row)
    sets = truth['candidates'][0] if truth['candidates'] else {}

    return [sets.get(str(i), truth['puzzle'][i] if truth['puzzle'][i] != '0' else alphabet) for i in range(width)]

def interchange_targets(row, donor):
    truth, other = truth_of(row), truth_of(donor)
    path = [other['start'], *other['stages']]

    return [shift_register.advance(path[j], truth['inputs'][j:], truth['table']) for j in range(row['horizon'] + 1)]

def edits(row):
    if task != 'sudoku':
        return {}

    truth, rng, found = truth_of(row), random.Random(row['id']), {}

    for j in range(1, truth['depth'] + 1):
        cells = [i for i in range(width) if truth['depth_map'][i] == j]
        rng.shuffle(cells)

        for i in cells:
            grid = [int(ch) for ch in truth['stages'][j - 1]]
            options = [v for v in sudoku.candidates(grid, i) if v != int(truth['solution'][i])]

            if options:
                target, changed = sudoku.edit_targets(row, i, options[0])

                if changed:
                    found[j] = {'position': i, 'digit': str(options[0]), 'target': target, 'changed': changed}

                    break

    return found

def learned(marks_by_horizon, rows):
    if task == 'sudoku':
        easy = {row['id'] for row in rows if row['machine'] == machine and row['rung'] >= 40}
        found = [value for by_id in marks_by_horizon.values() for uid, value in by_id.items() if uid in easy]

        return bool(found) and sum(found) / len(found) >= settings.learned

    found = list(marks_by_horizon.get(1, {}).values())

    return bool(found) and sum(found) / len(found) >= settings.learned