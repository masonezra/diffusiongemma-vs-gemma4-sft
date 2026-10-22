'''
Paired comparison of two models' replies on the same rows: a cluster-bootstrap interval of the pass-rate difference
(a problem's rows resampled together), plus McNemar's exact test where each group holds one row (test, val).
Primary comparison: test rows with several gaps.
    python -m infilling.meta.evaluate.compare test ar.jsonl diffusion.jsonl --several
Reference: https://arxiv.org/abs/2411.00640
'''

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common.files import read_jsonl
from infilling.meta.config.settings import data_dir
from infilling.meta.evaluate.metrics import load_replies, outcomes, select

def compare(rows, first, second, resamples=2000, seed=0):
    one = {name: np.array(values, float) for name, values in outcomes(rows, first).items()}
    two = {name: np.array(values, float) for name, values in outcomes(rows, second).items()}
    groups = collections.defaultdict(list)

    for at, row in enumerate(rows):
        groups[row.get('group', row['id'])].append(at)

    members = list(groups.values())
    draws = np.random.default_rng(seed).integers(0, len(members), (resamples, len(members)))
    result = {'groups': len(members), 'rows': len(rows)}

    for name in one:
        sums = np.array([[one[name][at].sum() for at in members], [two[name][at].sum() for at in members], [len(at) for at in members]])
        differences = np.array([100 * (sums[1][draw].sum() - sums[0][draw].sum()) / sums[2][draw].sum() for draw in draws])
        low, high = np.percentile(differences, [2.5, 97.5])
        result[name] = {'first': round(100 * one[name].mean(), 2), 'second': round(100 * two[name].mean(), 2), 'difference': round(100 * (two[name].mean() - one[name].mean()), 2), 'low': round(float(low), 2), 'high': round(float(high), 2)}

    only_first, only_second = int(((one['pass'] == 1) & (two['pass'] == 0)).sum()), int(((two['pass'] == 1) & (one['pass'] == 0)).sum())
    result['pass']['disagreements'] = only_first + only_second

    if len(members) == len(rows):
        result['pass']['p'] = round(binomtest(min(only_first, only_second), only_first + only_second).pvalue, 4) if only_first + only_second else 1.0

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('--benchmark')
    parser.add_argument('--several', action='store_true')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--out', default=data_dir)
    args = parser.parse_args()
    rows = select(read_jsonl(Path(args.out) / f'{args.split}.jsonl'), args.benchmark, args.several)

    print(json.dumps(compare(rows, load_replies(args.first, rows), load_replies(args.second, rows), args.resamples), indent=1))