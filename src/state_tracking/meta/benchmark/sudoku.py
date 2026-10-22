'''
The sudoku carrier: a 9x9 grid with givens G and a unique solution x*; the reply is x* as 81 space-separated digits {cell i at
canvas position 2i}, the format of Google's DiffusionGemma Sudoku recipe. Puzzles are the rohanrao Kaggle set Google trains on and the
Radcliffe rated set {the `9m` and `3m` rows of the Hugging Face mirror Ritvik19/Sudoku-Dataset}, each annotated by the singles operator P: one synchronous
round fills every naked single {one candidate left} and every hidden single {a digit with one admissible cell in a row, column
or box}; x^(j+1) = P(x^(j)); P is monotone and deterministic. A puzzle P finishes is `sudoku` with depth d = min j with
x^(j) = x* as its horizon, D(i) = min j with x^(j)_i filled the per-cell depth, and the candidate sets after every round kept
for the quantised reading; a puzzle P does not finish is `sudoku_hard` {search needed, horizon 0}. Train, val and test rows are stratified by
depth band {1-3, 4-6, 7-9, 10-12, 13+; the deep bands come from the Radcliffe rows}, with the not-singles tier in val and test only,
scanning the pool until every band fills, so the horizon dial is balanced and extends to depth 21; the split is MD5 of the relabelled solution mod 8 {val 0, test 1,
train 2..7}, so no solution grid is shared across splits up to digit relabelling. A seeded fill-and-dig generator {solution
count capped at 2} remains for building without the Kaggle file.
Reference: https://github.com/google-deepmind/gemma {gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py, convert_sudoku.py}
Reference: https://www.kaggle.com/datasets/rohanrao/sudoku {CC0}
Reference: https://arxiv.org/pdf/2608.00146 {Section 8, Table 5}
Reference: https://arxiv.org/pdf/2409.10502 {candidate sets as the solver state}
'''

import hashlib
import json
import random
from collections import Counter

from state_tracking.meta.config.prompts import chat
from state_tracking.meta.config.settings import sudoku_bands, sudoku_buckets, sudoku_sets, sudoku_source

digits = '123456789'
machines = ('sudoku', 'sudoku_hard')
width = 81
stride = 2
forms = tuple(digits)
hard_bucket = sudoku_buckets + 1
instruction = ('Solve the following Sudoku puzzle. Empty cells are represented by 0. Output ONLY the solved puzzle immediately as a 9x9 '
               'grid of numbers separated by spaces. Do not include ####, explanations, or any other text.')
rows_of = [[9 * r + c for c in range(9)] for r in range(9)]
cols_of = [[9 * r + c for r in range(9)] for c in range(9)]
boxes_of = [[9 * (3 * (b // 3) + r) + 3 * (b % 3) + c for r in range(3) for c in range(3)] for b in range(9)]
units = rows_of + cols_of + boxes_of
peers = [sorted({j for unit in units if i in unit for j in unit} - {i}) for i in range(81)]

def candidates(grid, i):
    return [v for v in range(1, 10) if all(grid[j] != v for j in peers[i])]

def solutions(grid, limit=2):
    grid = list(grid)
    found = []

    def search():
        best, options = None, None

        for i in range(81):
            if grid[i] == 0:
                choices = candidates(grid, i)

                if options is None or len(choices) < len(options):
                    best, options = i, choices

                    if not choices:
                        return

        if best is None:
            found.append(tuple(grid))

            return

        for v in options:
            grid[best] = v
            search()
            grid[best] = 0

            if len(found) >= limit:
                return

    search()

    return found

def full_grid(rng):
    grid = [0] * 81

    def fill(i):
        if i == 81:
            return True

        choices = candidates(grid, i)
        rng.shuffle(choices)

        for v in choices:
            grid[i] = v

            if fill(i + 1):
                return True

        grid[i] = 0

        return False

    fill(0)

    return tuple(grid)

def dig(solution, givens, rng):
    grid, order = list(solution), list(range(81))
    rng.shuffle(order)

    for i in order:
        if 81 - grid.count(0) <= givens:
            break

        kept, grid[i] = grid[i], 0

        if len(solutions(grid)) != 1:
            grid[i] = kept

    return tuple(grid) if 81 - grid.count(0) == givens else None

def singles(grid):
    options = {i: candidates(grid, i) for i in range(81) if grid[i] == 0}
    # P is synchronous: every naked and hidden single is read off the same grid, and a hidden single never overrides a naked one
    found = {i: choices[0] for i, choices in options.items() if len(choices) == 1}

    for unit in units:
        for v in range(1, 10):
            places = [i for i in unit if i in options and v in options[i]]

            if len(places) == 1 and grid[places[0]] == 0:
                found.setdefault(places[0], v)

    return found, options

def propagate(puzzle):
    grid, stages, depth, sets = list(puzzle), [], [0 if v else -1 for v in puzzle], []

    while 0 in grid:
        found, options = singles(grid)

        if not found:
            break

        sets.append({i: ''.join(map(str, choices)) for i, choices in options.items()})

        for i, v in found.items():
            grid[i] = v
            depth[i] = len(stages) + 1

        stages.append(tuple(grid))

    return stages, depth, sets

def text(grid):
    return ' '.join(map(str, grid))

def compact(grid):
    return ''.join(map(str, grid))

def relabelled(solution):
    # Relabelling digits by first appearance puts all 9! digit permutations of a solution in the same split
    order = {}

    return ''.join(str(order.setdefault(v, len(order) + 1)) for v in solution)

def partition(solution):
    residue = int(hashlib.md5(relabelled(solution).encode()).hexdigest(), 16) % 8

    return 'val' if residue == 0 else 'test' if residue == 1 else 'train'

def bucket(case):
    if case['machine'] == 'sudoku_hard':
        return hard_bucket

    return next(n for n, (low, high) in enumerate(sudoku_bands, 1) if low <= case['horizon'] <= high)

def annotate(puzzle, solution, origin):
    stages, depth, sets = propagate(puzzle)
    solved = bool(stages) and stages[-1] == solution

    return {'origin': origin, 'givens': 81 - puzzle.count(0), 'puzzle': puzzle, 'solution': solution, 'stages': stages, 'depth_map': depth, 'candidates': sets,
            'machine': 'sudoku' if solved else 'sudoku_hard', 'horizon': len(stages) if solved else 0, 'split': partition(solution)}

def make(seed, givens):
    rng = random.Random(seed)
    solution = full_grid(rng)
    puzzle = dig(solution, givens, rng)

    return annotate(puzzle, solution, f'generated:{seed}') if puzzle else None

def render(case, held=False):
    question = f'{instruction}\n\n{text(case["puzzle"])}'
    row = {'id': hashlib.md5(question.encode()).hexdigest(), 'machine': case['machine'], 'horizon': case['horizon'], 'rung': case['givens'], 'origin': case['origin'],
           'split': case['split'], 'cluster': relabelled(case['solution']), 'episode': hashlib.md5(compact(case['puzzle']).encode()).hexdigest(),
           'messages': chat(question, text(case['solution']))}

    if held:
        row['truth'] = json.dumps({'puzzle': compact(case['puzzle']), 'solution': compact(case['solution']), 'stages': [compact(stage) for stage in case['stages']],
                                   'depth': case['horizon'], 'depth_map': case['depth_map'], 'candidates': case['candidates']})

    return row

def kaggle(path=sudoku_source, subsets=sudoku_sets, limit=1_500_000):
    import pyarrow.parquet as pq

    seen = {subset: 0 for subset in subsets}

    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536, columns=['puzzle', 'solution', 'set']):
        for puzzle, solution, origin in zip(batch.column('puzzle').to_pylist(), batch.column('solution').to_pylist(), batch.column('set').to_pylist()):
            if origin in seen and seen[origin] < limit:
                yield f'{origin}:{seen[origin]}', tuple(int(ch) for ch in puzzle), tuple(int(ch) for ch in solution)
                seen[origin] += 1

        if all(count >= limit for count in seen.values()):
            return

def stratified(counts, seed, source=kaggle, log=None):
    rng = random.Random(seed)
    quota = {split: ({b: count // sudoku_buckets + (b <= count % sudoku_buckets) for b in range(1, sudoku_buckets + 1)} if split == 'train'
                     else {b: count // hard_bucket + (b <= count % hard_bucket) for b in range(1, hard_bucket + 1)}) for split, count in counts.items()}
    taken = {split: Counter() for split in counts}
    found, seen, wanted, scanned = {split: [] for split in counts}, set(), sum(sum(q.values()) for q in quota.values()), 0
    order = list(source())
    rng.shuffle(order)

    for n, puzzle, solution in order:
        if sum(sum(t.values()) for t in taken.values()) >= wanted:
            break

        scanned += 1
        case = annotate(puzzle, solution, f'kaggle:{n}')
        split, b = case['split'], bucket(case)

        if split not in counts or case['puzzle'] in seen or taken[split].get(b, 0) >= quota[split].get(b, 0):
            continue

        taken[split][b] += 1
        seen.add(case['puzzle'])
        found[split].append(case)

        if log and scanned % 10000 == 0:
            print(f'scanned {scanned} kept ' + ' '.join(f'{split} {sum(t.values())}/{sum(quota[split].values())}' for split, t in taken.items()), file=log, flush=True)

    if log:
        print('scanned ' + str(scanned) + ' buckets by true depth ' + json.dumps({split: dict(sorted(t.items())) for split, t in taken.items()}), file=log, flush=True)
        print('shortfall ' + json.dumps({split: {b: quota[split][b] - t[b] for b in quota[split] if t[b] < quota[split][b]} for split, t in taken.items()}), file=log, flush=True)

    return found

def rows(counts, seed, held=False, log=None):
    cases = stratified(counts, seed, log=log)

    return {split: [render(case, held and split != 'train') for case in found] for split, found in cases.items()}

def parse(reply):
    found = [int(ch) for ch in reply if ch.isdigit()]

    return found[:width] if len(found) >= width else None

def valid(grid):
    return all(sorted(grid[i] for i in unit) == list(range(1, 10)) for unit in units)

def mark(reply, row):
    puzzle = [int(ch) for ch in row['messages'][0]['content'].split('\n\n')[-1].split()]
    truth = [int(ch) for ch in row['messages'][1]['content'].split()]
    guess = parse(reply)
    blanks = [i for i in range(width) if puzzle[i] == 0]

    if guess is None:
        return {'exact': False, 'symbol': 0.0, 'cells': 0.0, 'givens': 0.0, 'valid': False, 'parsed': False}

    return {'exact': guess == truth, 'symbol': sum(guess[i] == truth[i] for i in blanks) / max(len(blanks), 1),
            'cells': sum(a == b for a, b in zip(guess, truth)) / width, 'givens': sum(guess[i] == puzzle[i] for i in range(width) if puzzle[i]) / max(width - len(blanks), 1),
            'valid': valid(guess), 'parsed': True}

def cell(row):
    return f"{row['machine']} d={row['horizon']}"

def path_of(row):
    truth = json.loads(row['truth'])

    return [truth['puzzle'], *truth['stages']]

def depth_map(row):
    return json.loads(row['truth'])['depth_map']

def edit_targets(row, i, v):
    truth = json.loads(row['truth'])
    j = truth['depth_map'][i]
    grid = [int(ch) for ch in truth['stages'][j - 1]]
    grid[i] = v
    stages, _, _ = propagate(tuple(grid))
    predicted = stages[-1] if stages else tuple(grid)
    changed = [k for k in range(width) if predicted[k] != int(truth['solution'][k])]

    return compact(predicted), changed