'''
Shared row format of the two vendored nocot-bench panels {chain, config_patch}: reply 'Answer: n'; split by
MD5(prompt) mod 8 {val 0, test 1, train 2..7}; equal share per horizon; no answer value above 1/7 of a held-out rung;
exact-match scoring and the constant-reply floor. The vendored code is unmodified, pinned in vendor/nocot/COMMIT.
Reference: https://github.com/neelnanda-io/nocot-bench
'''

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from state_tracking.meta.config.prompts import chat
from state_tracking.meta.config.settings import gold_share

vendor = Path(__file__).parents[2] / 'vendor' / 'nocot'
sys.path.insert(0, str(vendor))

from datagen.banks import cfgpatch, chain

commit = (vendor / 'COMMIT').read_text().strip()
reply_form = re.compile(r'Answer:\s*(-?\d+)')

def partition(uid):
    residue = int(uid, 16) % 8

    return 'val' if residue == 0 else 'test' if residue == 1 else 'train'

def row(machine, horizon, question, answer):
    uid = hashlib.md5(question.encode()).hexdigest()

    return {'id': uid, 'machine': machine, 'horizon': horizon, 'split': partition(uid), 'cluster': uid, 'episode': uid, 'messages': chat(question, f'Answer: {answer}')}

def truth(row):
    return json.loads(row['truth'])

def draw(make, horizon, split, rng, seen, made, held=False):
    while True:
        found = make(horizon, rng, made, held)

        if found and found['split'] == split and found['id'] not in seen:
            seen.add(found['id'])

            return found

def rows(make, horizons, count, seed):
    rng, seen, found = random.Random(seed), set(), []

    while len(found) < count:
        found += [draw(make, horizon, 'train', rng, seen, len(found) // len(horizons)) for horizon in horizons]

    return found[:count]

def panel(make, rungs, each, seed, split):
    # No answer value fills more than 1/gold_share of a rung, so a constant reply scores at most about 14%
    rng, seen, found, cap = random.Random(seed), set(), [], -(-each // gold_share)

    for horizon in rungs:
        golds, kept = Counter(), 0

        while kept < each:
            made = draw(make, horizon, split, rng, seen, kept, held=True)

            if golds[gold(made)] < cap:
                golds[gold(made)] += 1
                kept += 1
                found.append(made)

    return found

def gold(row):
    return int(row['messages'][1]['content'].split()[-1])

def mark(reply, row):
    said = reply_form.search(reply)

    return {'exact': bool(said) and int(said.group(1)) == gold(row)}

def floor(rows):
    by_rung = {}

    for row in rows:
        by_rung.setdefault(row['horizon'], Counter())[gold(row)] += 1

    return {horizon: max(golds.values()) / sum(golds.values()) for horizon, golds in sorted(by_rung.items())}