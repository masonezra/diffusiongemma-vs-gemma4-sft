'''
Shared paths (built files; download cache in ALPHA_CACHE or ~/.cache/alpha) and token caps: at most 250 answer
tokens, within one 256-token DiffusionGemma canvas, and 700 in all, within the recipes' seq_length 768.
'''

import os
from pathlib import Path

data_dir = Path(__file__).parents[4] / 'dataset' / 'inscriptions'
cache_dir = Path(os.environ.get('ALPHA_CACHE', Path.home() / '.cache' / 'alpha')) / 'inscriptions'
max_answer_tokens = 250
max_total_tokens = 700

def fits(row):
    return row['tokens'] <= max_answer_tokens and row['total_tokens'] <= max_total_tokens