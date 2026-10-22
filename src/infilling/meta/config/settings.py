'''
Shared constants: token caps so an answer fits one 256-slot DiffusionGemma canvas and a whole row fits the
2,048-token training sequence, and test timeouts (10 s, 120 s on HumanEval+).
'''

from pathlib import Path

data_dir = Path(__file__).parents[4] / 'dataset' / 'infilling'
max_answer = 240
max_total = 1700
default_timeout = 10
plus_timeout = 120

def fits(row):
    return row['tokens'] <= max_answer and row['total_tokens'] <= max_total

def timeout(row):
    return plus_timeout if row.get('benchmark') == 'humanevalplus' else default_timeout