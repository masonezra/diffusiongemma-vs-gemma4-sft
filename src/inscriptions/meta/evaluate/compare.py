'''
Paired comparison of two reply files, one row per stone: McNemar's exact test for yes/no metrics, a paired
bootstrap 95% interval for the rest; given over all rows and over rows both replies parse.
Reference: https://arxiv.org/abs/2411.00640
'''

import numpy as np
from scipy.stats import binomtest

from inscriptions.meta.evaluate.metrics import binary, ratios, score_row

def scored(rows, replies):
    return [score_row(row, reply)[0] for row, reply in zip(rows, replies)]

def arrays(scores):
    return {key: np.array([found[key] for found in scores], float) for pair in ratios for key in pair}

def ratio(part, whole, table, draw=slice(None)):
    total = table[whole][draw].sum()

    return 100 * table[part][draw].sum() / total if total else float('nan')

def table(first, second, resamples, seed):
    one, two = arrays(first), arrays(second)
    draws = np.random.default_rng(seed).integers(0, len(first), (resamples, len(first)))
    result = {'rows': len(first)}

    for part, whole in ratios:
        label = 'error' if part == 'edits' else part
        differences = np.array([ratio(part, whole, two, draw) - ratio(part, whole, one, draw) for draw in draws])
        low, high = (np.nanpercentile(differences, [2.5, 97.5]) if not np.isnan(differences).all() else (float('nan'), float('nan')))
        result[label] = {'first': round(ratio(part, whole, one), 2), 'second': round(ratio(part, whole, two), 2),
                         'difference': round(ratio(part, whole, two) - ratio(part, whole, one), 2), 'low': round(float(low), 2), 'high': round(float(high), 2)}

        if part in binary:
            only_first, only_second = int(((one[part] > 0) & (two[part] == 0)).sum()), int(((two[part] > 0) & (one[part] == 0)).sum())
            result[label]['disagreements'] = only_first + only_second
            result[label]['p'] = round(binomtest(min(only_first, only_second), only_first + only_second).pvalue, 4) if only_first + only_second else 1.0

    return result

def compare(rows, first, second, resamples=2000, seed=0):
    first, second = scored(rows, first), scored(rows, second)
    read = [n for n, (one, two) in enumerate(zip(first, second)) if not one['unread'] and not two['unread']]
    result = table(first, second, resamples, seed)

    if read:
        result['both read'] = table([first[n] for n in read], [second[n] for n in read], resamples, seed)

    return result