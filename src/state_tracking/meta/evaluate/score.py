'''
Scores replies per cell {machine, horizon}: exact match in percent with a cluster-bootstrap interval {clusters are solution
shapes, start rows or problems} and the carrier's symbol accuracy {sudoku: blank-cell accuracy, with all-cell accuracy, givens
preserved and grid validity beside it}, over the rows the reply file holds; with --against, the paired difference of two reply files.
    python -m state_tracking.meta.evaluate.score test replies.jsonl
'''

import argparse
import random
import re
from collections import defaultdict

from common.bootstrap import difference, draws_of, interval, resampled
from common.files import read_jsonl
from state_tracking.meta.benchmark import carrier, chain, config_patch, nocot
from state_tracking.meta.config.settings import data_dir

metrics = ('exact', 'symbol', 'cells', 'givens', 'valid')
integers = (chain.machine, config_patch.machine)

def mark(reply, row):
    return nocot.mark(reply, row) if row['machine'] in integers else carrier.mark(reply, row)

def cell(row):
    return carrier.cell(row) if row['machine'] in carrier.machines else f"{row['machine']} k={row['horizon']}"

def cluster(row):
    return row['cluster']

def natural(item):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', item[0])]

def samples(replies, row):
    return replies.get(row['id']) or ['']

def rate(replies, row):
    found = samples(replies, row)

    return sum(mark(reply, row)['exact'] for reply in found) / len(found)

def paired(rows, first, second):
    cells = defaultdict(list)

    for row in rows:
        cells[cell(row)].append(row)

    found = {}

    for name, group in sorted(cells.items(), key=natural):
        gaps = {row['id']: rate(second, row) - rate(first, row) for row in group}
        clusters = defaultdict(list)

        for row in group:
            clusters[cluster(row)].append(row['id'])

        groups = list(clusters.values())
        drawn, point = resampled(groups, gaps, draws_of(len(groups)))
        found[name] = {'rows': len(group), **difference(point, drawn)}

    return found

def score(rows, replies):
    table, rng, summary = defaultdict(list), random.Random(0), {}

    for row in rows:
        if row['id'] not in replies:
            continue

        for reply in samples(replies, row):
            table[cell(row)].append(dict(mark(reply, row), _cluster=cluster(row)))

    for name, results in sorted(table.items(), key=natural):
        summary[name] = {'rows': len(results)}

        for key in metrics:
            values = [float(result[key]) for result in results if key in result]

            if values:
                summary[name][key] = round(100 * sum(values) / len(values), 2)

        _, low, high = interval([float(result['exact']) for result in results], [result['_cluster'] for result in results], rng=rng)
        summary[name]['exact 95%'] = [round(100 * low, 2), round(100 * high, 2)]

    return summary

def load(source, rows):
    found = defaultdict(list)

    for row in (rows if source == 'truth' else read_jsonl(source)):
        found[row['id']].append(row['messages'][1]['content'] if source == 'truth' else row['reply'])

    return found

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('replies')
    parser.add_argument('--against')
    args = parser.parse_args()
    rows = read_jsonl(data_dir / f'{args.split}.jsonl')
    result = paired(rows, load(args.replies, rows), load(args.against, rows)) if args.against else score(rows, load(args.replies, rows))

    for name, found in result.items():
        print(f'{name:24s} {found}')