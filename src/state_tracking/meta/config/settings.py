'''
Fixed settings of the benchmark. The carrier task {sudoku, the benchmark; shift_register, the first carrier, kept} comes from the
STATE_TRACKING_TASK environment variable and chooses the data folder dataset/state_tracking/<recipe> and the results folder
results/state_tracking/<recipe> {recipe1 = shift_register, recipe2 = sudoku, matching the recipe files; generated replies and traces go
to results/.../runs}, the panels, the horizons and rungs, the interventions and the thresholds below. Sudoku: rows from the rohanrao and Radcliffe Kaggle sets
{sudoku_sets, in the Hugging Face mirror's training parquet sudoku_source}, stratified by depth band {sudoku_bands} with the
not-singles tier beside them: 32,000 train rows balanced over the bands, 250 val and 1,024 test rows per band and tier; nocot-bench chain and config_patch are evaluation-only panels {nocot_each test and
nocot_few val rows per rung}. Shift register: the budget of 96,000 training rows {shift_register 60,000, shift_register_single, chain and
config_patch 12,000 each}, 1,000 test and 100 val episodes per rung.
'''

import os
from pathlib import Path

task = os.environ.get('STATE_TRACKING_TASK', 'sudoku')
root = Path(__file__).parents[4] / 'dataset' / 'state_tracking'
folders = {'shift_register': 'recipe1', 'sudoku': 'recipe2'}
data_dir = root / folders[task]
results_dir = Path(__file__).parents[4] / 'results' / 'state_tracking' / folders[task]
runs_dir = results_dir / 'runs'
each = {'sudoku': 1024, 'shift_register': 1000}[task]
few = {'sudoku': 250, 'shift_register': 100}[task]
sudoku_source = Path.home() / '.cache' / 'alpha' / 'sudoku' / 'train_0.parquet'
sudoku_sets = ('9m', '3m')
sudoku_bands = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 99))
sudoku_buckets = len(sudoku_bands)
sudoku_train = 32_000
sudoku_horizons = tuple(range(1, 22))
nocot_each = {'sudoku': 256, 'shift_register': 1000}[task]
nocot_few = {'sudoku': 64, 'shift_register': 100}[task]
budget = {'shift_register': 60_000, 'shift_register_single': 12_000, 'chain': 12_000, 'config_patch': 12_000}
width = 16
letters = 25
shift_register_horizons = tuple(range(1, 25))
shift_register_rungs = (1, 2, 3, 4, 6, 8, 12, 16, 24)
chain_horizons = tuple(range(3, 9))
chain_rungs = (3, 4, 5, 6, 7, 8)
config_horizons = tuple(range(2, 13))
config_rungs = (2, 3, 4, 6, 8, 10, 12)
gold_share = 7
budgets = {'sudoku': (1, 2, 4, 8, 16, 48), 'shift_register': (1, 2, 4, 48)}[task]
handoff = 2
long = 48
long_episodes = 128
interventions = {'sudoku': ('full', 'zero', 'swap', 'top 1', 'top 2', 'top 3', 'resample 1', 'resample 2', 'resample 3', 'quantise', 'top 8', 'edit 1', 'edit 2'),
                 'shift_register': ('full', 'swap', 'top 8', 'resample 8', 'resample 1')}[task]
decisive = {'sudoku': 'resample 3', 'shift_register': 'resample 8'}[task]
long_kinds = {'sudoku': ('full', 'zero', 'top 1', 'top 3', 'resample 3', 'quantise'), 'shift_register': ('full', 'swap', 'top 8', 'resample 8', 'resample 1')}[task]
shipped_kinds = ('full', 'zero', 'top 3') if task == 'sudoku' else ('full', 'swap', 'top 8')
learned = 0.5
margin = 4.0
informative_floor = 12.0
gate = 3
carried_floor = 10.0
legible_floor = 10.0
alpha = 0.025