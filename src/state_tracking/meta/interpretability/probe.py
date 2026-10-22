'''
What the handed state carries about the intermediate round beyond what a monitor reads, as conditional V-information with V the
linear-softmax probes, in bits per symbol, over the reply positions the carrier scores at round j {sudoku: the cells filled at
round j, or every blank cell for the final round}:
    hidden     I_V(S -> x^(j)_i | log p)        S the state step 2 receives, p the step-1 probabilities of the alphabet tokens:
               what S holds beyond the probability of every symbol {a hidden carrier}
    unread k   I_V(log p -> x^(j)_i | log r_k)  r_k the monitor's top-k reading {arXiv 2606.20560, Algorithm 1} in alphabet
               coordinates {symbol mass kept, mass kept on other tokens, mass spread}: the legible part a top-k monitor drops
    control    hidden with S taken from another episode {same cells}
Features: log probabilities clipped at 1e-6, standardised; S compressed to the leading components of a PCA fitted on the training
fold and scaled by one global RMS, so small directions are not raised to unit scale. The L2 decay is chosen per fit on an inner
split of the training fold. Two-fold cross-fitting by episode; the interval is a bootstrap over held-out episodes. Shannon
information would be vacuous: x^(j) is a deterministic function of the prompt. Fewer than min_episodes aligned replies {the
untuned checkpoint rarely writes the grid} are reported as such.
Delta = S - B of the top-8 reading is not probed: on nine digits the top 8 leave almost nothing unread, so Delta is the fp16
rounding of the stored S and B {norm about 0.01 against 80}, and per-feature standardisation would raise that noise to unit scale.
    python -m state_tracking.meta.interpretability.probe val diffusion.T2.ladder
Reference: https://arxiv.org/abs/2002.10689 {usable information}
Reference: https://arxiv.org/abs/2109.09234 {conditional probing}
'''

import argparse
import hashlib
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from common.files import read_jsonl
from common.models import diffusion_gemma
from state_tracking.meta.benchmark import carrier
from state_tracking.meta.config.settings import data_dir, runs_dir
from state_tracking.meta.evaluate.stats import summary
from state_tracking.meta.interpretability.legibility import alphabet_ids, letter_ids, locate

readings = (1, 3, 8)
components = 64
decays = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
clip = 1e-6
min_episodes = 50

def features(trace, first, alphabet, vocabulary):
    where = first + carrier.stride * np.arange(carrier.width)
    ids, probs = trace['ids'][0, where].astype(np.int64), trace['probs'][0, where].astype(np.float32)
    p = trace['watch'][0, where].astype(np.float32)
    found = {'p': np.log(np.clip(p, clip, 1)), 'S': trace['state'].astype(np.float32)}

    for k in readings:
        kept = np.isin(alphabet[None, :, None], ids[:, None, :k]).any(-1)
        spread = np.clip(1 - probs[:, :k].sum(-1), 0, 1)
        symbol = np.where(kept, p, spread[:, None] / (vocabulary - k))
        other = np.clip(probs[:, :k].sum(-1) - (p * kept).sum(-1), 0, 1)
        found[f'r{k}'] = np.log(np.clip(np.concatenate([symbol, other[:, None], spread[:, None]], 1), clip, 1))

    return found

def collect(rows, run):
    tokenizer = AutoTokenizer.from_pretrained(diffusion_gemma)
    letters, alphabet = letter_ids(tokenizer), np.array(alphabet_ids(tokenizer))
    found = []

    for row in rows:
        path = run / f"{row['id']}.npz"

        if path.exists():
            with np.load(path) as trace:
                first = locate(trace['final'], letters)

                if 'state' in trace and 'watch' in trace and first is not None:
                    found.append((row, features(trace, first, alphabet, len(tokenizer)), carrier.path_of(row)))

    return found

def fold(row, salt=''):
    return int(hashlib.md5((salt + row['id']).encode()).hexdigest(), 16) % 2

def positions_at(row, path, j):
    if carrier.task == 'sudoku' and j == len(path) - 1:
        return carrier.blanks(row)

    return carrier.scored_positions(row, path, j)

class Encoder:
    def __init__(self, blocks):
        self.low = [block for block in blocks if block != 'S']
        self.state = 'S' in blocks

    def fit(self, x):
        low = np.concatenate([x[name] for name in self.low], 1)
        self.mean, self.scale = low.mean(0), low.std(0) + 1e-6

        if self.state:
            self.centre = x['S'].mean(0)
            _, _, v = np.linalg.svd(x['S'][:5_000] - self.centre, full_matrices=False)
            self.basis = v[:components].T
            self.rms = float(np.sqrt((((x['S'] - self.centre) @ self.basis) ** 2).mean())) + 1e-6

        return self

    def __call__(self, x):
        low = (np.concatenate([x[name] for name in self.low], 1) - self.mean) / self.scale

        return np.concatenate([low, (x['S'] - self.centre) @ self.basis / self.rms], 1) if self.state else low

def train(x, y, decay, iterations=200):
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y)
    model = torch.nn.Linear(x.shape[1], len(carrier.alphabet))
    optimiser = torch.optim.LBFGS(model.parameters(), max_iter=iterations, line_search_fn='strong_wolfe')

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y) + decay * model.weight.square().sum()
        loss.backward()

        return loss

    optimiser.step(closure)

    return model

def held_out(model, x, y):
    with torch.no_grad():
        return torch.nn.functional.cross_entropy(model(torch.as_tensor(x, dtype=torch.float32)), torch.as_tensor(y), reduction='none').numpy() / np.log(2)

def take(x, mask):
    return {name: value[mask] for name, value in x.items()}

def losses(blocks, train_x, train_y, train_ids, test_x, test_y):
    inner = np.array([fold({'id': uid}, 'inner') for uid in train_ids])
    scores = {}

    for decay in decays:
        encoder = Encoder(blocks).fit(take(train_x, inner == 0))
        model = train(encoder(take(train_x, inner == 0)), train_y[inner == 0], decay)
        scores[decay] = held_out(model, encoder(take(train_x, inner == 1)), train_y[inner == 1]).mean()

    encoder = Encoder(blocks).fit(train_x)

    return held_out(train(encoder(train_x), train_y, min(scores, key=scores.get)), encoder(test_x), test_y)

def table(items, j, other):
    blocks, y, ids = {}, [], []

    for n, (row, x, path) in enumerate(items):
        where = positions_at(row, path, j)

        for name, value in x.items():
            blocks.setdefault(name, []).append(value[where])

        blocks.setdefault('S shuffled', []).append(items[other[n]][1]['S'][where])
        y += [carrier.alphabet.index(path[j][i]) for i in where]
        ids += [row['id']] * len(where)

    return {name: np.concatenate(value) for name, value in blocks.items()}, np.array(y), np.array(ids)

def gains(found, j, seed=0):
    items = [item for item in found if j < len(item[2]) and positions_at(item[0], item[2], j)]
    other = np.random.default_rng(seed).permutation(len(items))
    x, y, ids = table(items, j, other)
    parts = np.array([fold({'id': uid}) for uid in ids])
    shuffled = {**x, 'S': x['S shuffled']}
    fits = {'p': ('p',), 'p S': ('p', 'S'), **{f'r{k}': (f'r{k}',) for k in readings}}
    per_row = {name: np.zeros(len(y)) for name in (*fits, 'p S shuffled')}

    for held in (0, 1):
        fit, test = parts != held, parts == held

        for name, blocks in fits.items():
            per_row[name][test] = losses(blocks, take(x, fit), y[fit], ids[fit], take(x, test), y[test])

        per_row['p S shuffled'][test] = losses(('p', 'S'), take(shuffled, fit), y[fit], ids[fit], take(shuffled, test), y[test])

    def by_episode(values):
        return [values[ids == uid].mean() for uid in dict.fromkeys(ids)]

    found = {'hidden': summary(by_episode(per_row['p'] - per_row['p S']), scale=1),
             'control': summary(by_episode(per_row['p'] - per_row['p S shuffled']), scale=1),
             'cross_entropy_p': round(float(per_row['p'].mean()), 4)}
    found.update({f'unread {k}': summary(by_episode(per_row[f'r{k}'] - per_row['p']), scale=1) for k in readings})

    return found

def probe(split, run, rounds=None):
    rows = [row for row in read_jsonl(data_dir / f'{split}.jsonl') if row['machine'] == carrier.machine]
    found = collect(rows, runs_dir / f'{split}.diffusion.{run}')

    if len(found) < min_episodes:
        return {'episodes': len(found), 'status': f'fewer than {min_episodes} aligned replies'}

    rounds = rounds or range(1, max(len(item[2]) for item in found))

    return {'episodes': len(found), 'bits_per_symbol': {j: gains(found, j) for j in rounds if sum(len(positions_at(r, p, j)) for r, _, p in found if j < len(p)) >= 200}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('run')
    parser.add_argument('--rounds', type=int, nargs='*')
    args = parser.parse_args()
    print(json.dumps(probe(args.split, args.run, args.rounds), indent=1))