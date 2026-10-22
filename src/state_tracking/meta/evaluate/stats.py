'''
Bootstrap of a mean over rows, in percentage points by default {scale 1 for bits}: the point, a two-sided 1 - 2 alpha interval {alpha = 0.025: one
sided tests at 0.025 and the equivalence test at 0.025 per side}, and mde = 2.8 SE. Rows are the episodes; a cell
holds one row per episode, so resampling rows is resampling episodes.
'''

import numpy as np

from state_tracking.meta.config.settings import alpha

def summary(values, draws=2000, seed=0, scale=100):
    values = np.asarray([value for value in values if value == value], float)

    if not len(values):
        return {'value': None, 'low': None, 'high': None, 'mde': None, 'rows': 0}

    means = values[np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))].mean(1)
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])

    return {'value': round(scale * float(values.mean()), 3), 'low': round(scale * float(low), 3), 'high': round(scale * float(high), 3),
            'mde': round(2.8 * scale * float(means.std()), 3), 'rows': int(len(values))}

def above(found, floor):
    return found['value'] is not None and found['low'] > 0 and found['value'] >= floor

def within(found, margin):
    return found['value'] is not None and -margin <= found['low'] and found['high'] <= margin