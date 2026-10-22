'''
The 50% crossing k* of exact match against the horizon k, from a logistic above the floor c_k
    P(exact | k) = c_k + (1 - c_k) sigmoid(a (k* - k)),
fitted by grid maximum likelihood with a bootstrap interval over episodes; a k* outside the rungs is censored to the
nearest rung. Also the empirical crossing, and the paired bootstrap of the shift in k* between two runs on the same rows.
Reference: https://github.com/neelnanda-io/nocot-bench/blob/main/nocot/place.py
'''

import math
import random

import numpy as np

k50_grid = np.exp(np.linspace(math.log(0.25), math.log(512), 600))
slope_grid = np.exp(np.linspace(math.log(0.02), math.log(8), 120))

def fit(counts, floors):
    ks = np.array(sorted(counts), dtype=float)
    right = np.array([counts[k][0] for k in sorted(counts)], dtype=float)
    total = np.array([counts[k][1] for k in sorted(counts)], dtype=float)
    logit = slope_grid[None, :, None] * (k50_grid[:, None, None] - ks[None, None, :])
    floor = np.array([floors.get(k, 0.0) for k in sorted(counts)], dtype=float)
    p = np.clip(floor + (1 - floor) / (1 + np.exp(-logit)), 1e-6, 1 - 1e-6)
    likelihood = (right * np.log(p) + (total - right) * np.log(1 - p)).sum(-1)
    best = np.unravel_index(likelihood.argmax(), likelihood.shape)

    return float(k50_grid[best[0]]), float(slope_grid[best[1]])

def exact(outcomes):
    return {int(k): round(100 * sum(found) / len(found), 2) for k, found in sorted(outcomes.items())}

def empirical(outcomes, floors):
    ks = sorted(outcomes)
    gaps = [sum(outcomes[k]) / len(outcomes[k]) - (1 + floors.get(k, 0.0)) / 2 for k in ks]

    if gaps[0] < 0:
        return None

    for a, b, above, below in zip(ks, ks[1:], gaps, gaps[1:]):
        if above >= 0 > below:
            return round(a + (b - a) * above / (above - below), 2)

    return None

def crossing(outcomes, floors=None, rng=None, draws=400):
    rng, floors = rng or random.Random(0), floors or {}
    least, most = min(outcomes), max(outcomes)
    k50, slope = fit({k: (sum(found), len(found)) for k, found in outcomes.items()}, floors)
    drawn = []

    for _ in range(draws):
        counts = {k: (sum(rng.choices(found, k=len(found))), len(found)) for k, found in outcomes.items()}
        drawn.append(float(min(max(fit(counts, floors)[0], least), most)))

    drawn.sort()

    return {'k50': round(float(min(max(k50, least), most)), 2), 'censored': 'below' if k50 < least else 'above' if k50 > most else None,
            'slope': round(slope, 3), 'low': round(drawn[int(0.025 * draws)], 2), 'high': round(drawn[int(0.975 * draws)], 2),
            'empirical': empirical(outcomes, floors), 'exact': list(exact(outcomes).values()),
            'rungs': [int(k) for k in sorted(outcomes)], 'episodes': int(min(len(found) for found in outcomes.values()))}

def shift(first, second, floors=None, draws=400, seed=0):
    rng, floors = np.random.default_rng(seed), floors or {}
    least, most = min(first), max(first)
    clip = lambda value: min(max(value, least), most)
    point = clip(fit({k: (sum(second[k]), len(second[k])) for k in second}, floors)[0]) - clip(fit({k: (sum(first[k]), len(first[k])) for k in first}, floors)[0])
    drawn = []

    for _ in range(draws):
        picks = {k: rng.integers(0, len(first[k]), len(first[k])) for k in first}
        a = fit({k: (int(np.asarray(first[k])[picks[k]].sum()), len(first[k])) for k in first}, floors)[0]
        b = fit({k: (int(np.asarray(second[k])[picks[k]].sum()), len(second[k])) for k in second}, floors)[0]
        drawn.append(clip(b) - clip(a))

    low, high = np.percentile(drawn, [2.5, 97.5])

    return {'difference': round(float(point), 2), 'low': round(float(low), 2), 'high': round(float(high), 2), 'nonzero': bool(low > 0 or high < 0)}