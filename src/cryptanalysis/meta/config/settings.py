'''
Benchmark constants: rows per level and split, substitution lengths 16 to 254, the pinned arXiv snapshot with held-out
month 2607, and the unicity margin (2.5) and English redundancy (3.2 bits per letter) that set the shortest ciphertext.
Two recipes side by side under dataset/cryptanalysis: recipe1 {the cipher and key hidden} and recipe2 {cipher name and key
printed}, the plaintext alone as the reply in both; recipe 2 has the same seven levels at lengths 32, 64 and 128 and
96,000 rows read once, the 3,000 steps of recipe 1.
'''

from pathlib import Path

data_dir = Path(__file__).parents[4] / 'dataset' / 'cryptanalysis'
recipe_dirs = {1: data_dir / 'recipe1', 2: data_dir / 'recipe2'}
budget = 320_000
texts_each = 950
texts_few = 100
lengths = (16, 32, 64, 128, 254)
budget_keyed = 96_000
texts_each_keyed = 500
lengths_keyed = (32, 64, 128)
source = 'librarian-bots/arxiv-metadata-snapshot'
revision = '5071ce89fe1da64b9d51284b44b924412cd044b7'
release = '2607'
spare = 30_000
overlap = 32
redundancy = 3.2
margin = 2.5