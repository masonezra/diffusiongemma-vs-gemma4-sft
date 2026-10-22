'''
Cluster bootstrap of a rate and of a paired difference: whole clusters are redrawn, so rows sharing a source are not
counted as independent. mde = 2.8 x bootstrap SE (80% power, 5% two-sided). One set of draws serves every comparison on
the same clusters.
Reference: https://arxiv.org/pdf/2411.00640
'''

import random

import numpy as np

power = 2.8

def interval(values, clusters, draws=1000, rng=None):
    rng = rng or random.Random(0)
    grouped = {}

    for cluster, value in zip(clusters, values):
        grouped.setdefault(cluster, []).append(value)

    units = [(sum(found), len(found)) for found in grouped.values()]
    rates = sorted(sum(total for total, _ in drawn) / sum(count for _, count in drawn) for drawn in (rng.choices(units, k=len(units)) for _ in range(draws)))

    return len(units), rates[int(0.025 * draws)], rates[int(0.975 * draws)]

def draws_of(count, draws=2000, seed=0):
    return np.random.default_rng(seed).integers(0, count, (draws, count))

def resampled(groups, values, index):
    sums = np.array([sum(values[uid] for uid in group) for group in groups], float)
    counts = np.array([len(group) for group in groups], float)

    return 100 * sums[index].sum(1) / counts[index].sum(1), 100 * sums.sum() / counts.sum()

def difference(point, drawn):
    low, high = np.percentile(drawn, [2.5, 97.5])

    return {'difference': round(float(point), 2), 'low': round(float(low), 2), 'high': round(float(high), 2), 'mde': round(power * float(drawn.std()), 2)}