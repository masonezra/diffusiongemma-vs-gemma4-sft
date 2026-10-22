'''
Commit order against propagation depth, from the per-step canvases of a traced run: the commit step of a reply position is
the first denoising step from which its canvas token never changes again; per puzzle, Kendall's tau-b between the commit
steps of the blank cells and their depths D(i), so that tau > 0 means cells the solver fills later are committed later.
Reported as the mean tau over puzzles with a bootstrap interval, beside the mean commit step by depth.
    python -m state_tracking.meta.interpretability.commit test diffusion.T48.ladder
Reference: https://arxiv.org/abs/2606.14620 {commit order on DiffusionGemma}
'''

import argparse
import json
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from common.files import read_jsonl
from common.models import diffusion_gemma
from state_tracking.meta.benchmark import carrier
from state_tracking.meta.config.settings import data_dir, runs_dir
from state_tracking.meta.evaluate.stats import summary
from state_tracking.meta.interpretability.legibility import letter_ids, locate

def tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0

    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = np.sign(x[a] - x[b]), np.sign(y[a] - y[b])

            if dx == 0 and dy == 0:
                continue

            if dx == 0:
                ties_x += 1

            elif dy == 0:
                ties_y += 1

            elif dx == dy:
                concordant += 1

            else:
                discordant += 1

    denominator = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))

    return (concordant - discordant) / denominator if denominator else np.nan

def commit_steps(canvas, final, where):
    steps = []

    for position in where:
        column = canvas[:, position]
        committed = len(column)

        for step in range(len(column) - 1, -1, -1):
            if column[step] == final[position]:
                committed = step

            else:
                break

        steps.append(committed + 1)

    return steps

def analyse(split, run):
    letters = letter_ids(AutoTokenizer.from_pretrained(diffusion_gemma))
    rows = [row for row in read_jsonl(data_dir / f'{split}.jsonl') if row['machine'] == carrier.machine]
    folder = runs_dir / f'{split}.diffusion.{run}'
    taus, by_depth = [], defaultdict(list)

    for row in rows:
        path = folder / f"{row['id']}.npz"

        if not path.exists():
            continue

        with np.load(path) as trace:
            first = locate(trace['final'], letters)

            if first is None:
                continue

            where = first + carrier.stride * np.arange(carrier.width)
            steps = commit_steps(trace['canvas'], trace['final'], where)

        depth = carrier.depth_of(row)
        blanks = carrier.blanks(row)

        if len({depth[i] for i in blanks}) > 1:
            taus.append(tau_b([steps[i] for i in blanks], [depth[i] for i in blanks]))

        for i in blanks:
            by_depth[depth[i]].append(steps[i])

    return {'puzzles': len(taus), 'tau_b': summary(taus, scale=1), 'commit_step_by_depth': {int(d): round(float(np.mean(v)), 2) for d, v in sorted(by_depth.items())}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('run')
    args = parser.parse_args()
    print(json.dumps(analyse(args.split, args.run), indent=1))