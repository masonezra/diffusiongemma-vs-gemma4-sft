'''
Scores replies per level (and per length or key width): positional and edit-distance symbol error, exact match, and clash, the share of symbols that contradict the reply's own majority key (a direct test of global consistency).
Error and exact rates carry a bootstrap interval over texts; --against gives the paired model difference per cell.
    python -m cryptanalysis.meta.evaluate.score test ar.jsonl --against diffusion.jsonl
    python -m cryptanalysis.meta.evaluate.score test ar.jsonl --data ../dataset/cryptanalysis/recipe2   {default: the recipe1 folder}
'''

import argparse
from pathlib import Path
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

from common.bootstrap import difference, draws_of, interval, resampled
from common.files import read_jsonl
from cryptanalysis.meta.benchmark.decipherment import functional, spaced
from cryptanalysis.meta.config.prompts import label
from cryptanalysis.meta.config.settings import recipe_dirs

metrics = ('error', 'edits', 'exact', 'clash')
compared = ('error', 'edits', 'exact')

def answer(reply):
    return ''.join(reply.strip().split())

def edit_distance(guess, truth):
    symbols, offsets = np.array(list(truth)), np.arange(len(truth) + 1)
    row = offsets.copy()

    # One Levenshtein row per guess symbol: deletions and substitutions are vectorised, then the min-accumulate of
    # row - offsets applies the insertions along the row
    for n, symbol in enumerate(guess, 1):
        row = np.concatenate(([n], np.minimum(row[1:] + 1, row[:-1] + (symbols != symbol))))
        row = np.minimum.accumulate(row - offsets) + offsets

    return int(row[-1])

def mark(reply, row):
    question, truth = (message['content'] for message in row['messages'])
    truth, guess = answer(truth), answer(reply)
    wrong = sum(a != b for a, b in zip(guess, truth)) + abs(len(guess) - len(truth))
    result = {'error': min(1, wrong / len(truth)), 'edits': min(1, edit_distance(guess, truth) / len(truth)), 'exact': guess == truth}

    if row['level'] in functional and guess:
        # A consistent reply maps each ciphertext symbol to one letter; clash counts positions off the reply's own majority
        secret = answer(question.split(label)[1])
        votes = defaultdict(Counter)

        for symbol, char in zip(secret, guess):
            votes[symbol][char] += 1

        result['clash'] = 1 - sum(max(count.values()) for count in votes.values()) / min(len(secret), len(guess))

    return result

def cells(row):
    name = f"level {row['level']} {row['cipher']}{' with word gaps' if row['level'] in spaced else ''}"

    if row['bucket']:
        return [name, f"{name}, n={row['bucket']}"]

    return [name, f"{name}, width={len(json.loads(row['key']))}"] if row['cipher'] == 'columnar' else [name]

def natural(item):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', item[0])]

def samples(replies, row):
    return replies[row['id']]

def rate(replies, row, key):
    found = samples(replies, row)

    return sum(mark(reply, row)[key] for reply in found) / len(found)

def paired(rows, first, second):
    grouped = defaultdict(list)

    for row in rows:
        for name in cells(row):
            grouped[name].append(row)

    found = {}

    for name, group in sorted(grouped.items(), key=natural):
        groups = [[row['id']] for row in group]
        found[name] = {'rows': len(group)}

        for key in compared:
            gaps = {row['id']: rate(second, row, key) - rate(first, row, key) for row in group}
            drawn, point = resampled(groups, gaps, draws_of(len(groups)))
            found[name][key] = difference(point, drawn)

    return found

def score(rows, replies):
    table, rng, summary = defaultdict(list), random.Random(0), {}

    for row in rows:
        for reply in samples(replies, row):
            found = dict(mark(reply, row), _cluster=row['id'])

            for name in cells(row):
                table[name].append(found)

    for name, results in sorted(table.items(), key=natural):
        summary[name] = {'rows': len(results)}

        for key in metrics:
            values = [float(result[key]) for result in results if key in result]

            if values:
                summary[name][key] = round(100 * sum(values) / len(values), 2)

        for key in compared:
            _, low, high = interval([float(result[key]) for result in results], [result['_cluster'] for result in results], rng=rng)
            summary[name][f'{key} 95%'] = [round(100 * low, 2), round(100 * high, 2)]

    return summary

def load(source, rows):
    found = defaultdict(list)

    for row in (rows if source == 'truth' else read_jsonl(source)):
        found[row['id']].append(row['messages'][1]['content'] if source == 'truth' else row['reply'])

    missing = [row['id'] for row in rows if row['id'] not in found]
    assert not missing, f'{len(missing)} rows of the file have no reply in {source}, the first {missing[0]}'

    return dict(found)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('replies')
    parser.add_argument('--against')
    parser.add_argument('--data', default=str(recipe_dirs[1]))
    args = parser.parse_args()
    rows = read_jsonl(Path(args.data) / f'{args.split}.jsonl')
    result = paired(rows, load(args.replies, rows), load(args.against, rows)) if args.against else score(rows, load(args.replies, rows))

    for name, found in result.items():
        print(f'{name:56s} {found}')