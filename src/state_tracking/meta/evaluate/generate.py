'''
Replies of either model on one split; both load through common.load with the LoRA merged. DiffusionGemma: a 256-token
canvas, exactly T denoising steps at temperature 1 {design} or Google's shipped sampler {temperature 0.8 to 0.4, entropy bound,
adaptive stop; --sampler shipped}, an optional intervention on the state handed between steps, hold
{the training forward at T = 2}, donors {un-intervened copies, for T > 2}, a trace of the top-8 reading per step and,
with --states, the state step 2 receives and the monitor's re-embedding of its reading at the reply positions. The
quantise and edit interventions read the carrier's candidate sets and counterfactual edits of each held-out row.
Gemma 4: greedy, thinking off. Rows are batched per cell with seeded noise, so every intervention sees the same
batches, noise and donors {the batch is halved when donor copies are appended}; cells holding a single row are pooled into
one batch, so every row has a donor. --limit keeps the first n episodes of
every cell; an intervention needs at least 2.
    python -m state_tracking.meta.evaluate.generate test diffusion --run diffusion_gradient.T2 --steps 2 --checkpoint DIR
Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py
'''

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSamplerConfig

from common.files import read_jsonl
from common.generate import decode, prompts, seed_for, write_ar
from common.load import load
from state_tracking.meta.benchmark import carrier
from state_tracking.meta.config.settings import data_dir, runs_dir
from state_tracking.meta.interpretability.channel import Traced, parse
from state_tracking.meta.interpretability.legibility import alphabet_ids, letter_ids, locate

def settings(steps, sampler='design'):
    if sampler == 'shipped':
        return {'max_denoising_steps': steps}

    # Design sampler: temperature held at 1 and both stopping rules out of reach, so every run takes exactly `steps` steps
    return {'max_denoising_steps': steps, 't_max': 1.0, 't_min': 0.999, 'sampler_config': EntropyBoundSamplerConfig(entropy_bound=0.1),
            'stability_threshold': steps + 1, 'confidence_threshold': 1e-9}

def batches(rows, size):
    cells = defaultdict(list)

    for row in rows:
        cells[(row['machine'], row['horizon'])].append(row)

    groups = [group for group in cells.values() if len(group) > 1]
    lone = [group[0] for group in cells.values() if len(group) == 1]

    if len(lone) > 1 or (lone and not groups):
        groups.append(lone)

    elif lone:
        groups[-1] = groups[-1] + lone

    for group in groups:
        chunks = [group[first:first + size] for first in range(0, len(group), size)]

        # Every row needs a donor in its batch, so no batch is left with a single row
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2:] = [chunks[-2] + chunks[-1]]

        yield from chunks

def saved(model, n, first):
    where = first + carrier.stride * np.arange(carrier.width)

    return {'state': model.states[1][n, where].float().cpu().numpy().astype(np.float16),
            'monitor': model.monitor[0][n, where].float().cpu().numpy().astype(np.float16)}

def extras(ran, channel, symbols, device):
    kind, amount = parse(channel)
    candidates = edit = None

    if kind == 'quantise':
        sets = [carrier.candidates(row) for row in ran]
        candidates = torch.tensor([[[symbol in found for symbol in carrier.alphabet] for found in row_sets] for row_sets in sets], device=device)

    if kind == 'edit':
        chosen = [carrier.edits(row).get(amount) for row in ran]
        edit = [(found['position'], symbols[found['digit']]) if found else (-1, 0) for found in chosen]

    return candidates, edit

@torch.no_grad()
def run_diffusion(rows, model, tokenizer, out, traces, steps, batch=8, trace=False, states=False, channel='full', hold=False, donors=False, sampler='design'):
    width_of_canvas = model.config.canvas_length
    ends = [int(end) for end in np.atleast_1d(model.generation_config.eos_token_id)]
    letters = letter_ids(tokenizer)
    symbols = {symbol: token for token, symbol in letters.items() if not tokenizer.convert_ids_to_tokens(token).startswith('▁')}
    model.watch = torch.tensor(alphabet_ids(tokenizer), device=model.device) if trace else None
    model.alphabet_ids, model.width, model.stride = torch.tensor(alphabet_ids(tokenizer), device=model.device), carrier.width, carrier.stride

    if trace:
        traces.mkdir(parents=True, exist_ok=True)

    with open(out, 'w', encoding='utf-8') as file:
        for group in batches(rows, max(batch // 2, 2) if donors else batch):
            # Donor copies are the batch rotated by one, so row n reads the state of row n + 1
            ran = group + group[1:] + group[:1] if donors else group
            torch.manual_seed(seed_for(group))
            candidates, edit = extras(ran, channel, symbols, model.device)
            model.begin(channel, hold, donors, record=trace, states=states, candidates=candidates, edit=edit)
            made = model.generate(**prompts(ran, tokenizer).to(model.device), max_new_tokens=width_of_canvas, return_dict_in_generate=True,
                                  disable_compile=True, **settings(steps, sampler))
            finished = np.stack(model.finished, 1)
            spread = np.mean(model.spread, 0) if model.spread else None

            for n, (row, ids) in enumerate(zip(group, made.sequences[:, -width_of_canvas:].cpu())):
                used = int(finished[n].argmax()) + 1 if finished[n].any() else finished.shape[1]

                if trace:
                    kept = {key: np.stack([record[key][n] for record in model.record], 0)[:used] for key in model.record[0]}
                    first = locate(ids.numpy(), letters) if states and len(model.states) > 1 else None
                    kept.update(saved(model, n, first) if first is not None else {})
                    np.savez_compressed(traces / f"{row['id']}.npz", final=ids.numpy(), ends=np.array(ends), **kept)

                found = {'id': row['id'], 'partner': group[(n + 1) % len(group)]['id'], 'steps_used': used}
                found.update({'spread': round(float(spread[n]), 4)} if spread is not None else {})
                found.update({'edit': edit[n][0]} if edit is not None else {})
                file.write(json.dumps({**found, **decode(ids, tokenizer, ends)}, ensure_ascii=False) + '\n')

def shard_suffix(part):
    return '' if part == '0/1' else '.part' + part.replace('/', 'of')

def select(rows, machines=(), rungs=(), limit=0, part='0/1'):
    share, shares = map(int, part.split('/'))
    found, taken = [], defaultdict(int)

    for row in rows:
        if (not machines or row['machine'] in machines) and (not rungs or row['horizon'] in rungs):
            key = (row['machine'], row['horizon'])

            if not limit or taken[key] < limit:
                taken[key] += 1
                found.append(row)

    episodes = list(dict.fromkeys(row['episode'] for row in found))[share::shares]
    order = {episode: n for n, episode in enumerate(episodes)}

    return sorted((row for row in found if row['episode'] in order), key=lambda row: (order[row['episode']], row['machine']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('model', choices=('ar', 'diffusion'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--channel', default='full')
    parser.add_argument('--hold', action='store_true')
    parser.add_argument('--donors', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--states', action='store_true')
    parser.add_argument('--part', default='0/1')
    parser.add_argument('--machines', nargs='*', default=())
    parser.add_argument('--rungs', type=int, nargs='*', default=())
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sampler', choices=('design', 'shipped'), default='design')
    args = parser.parse_args()
    found = select(read_jsonl(data_dir / f'{args.split}.jsonl'), tuple(args.machines), tuple(args.rungs), args.limit, args.part)
    model, tokenizer = load(args.model, args.checkpoint, cls=Traced if args.model == 'diffusion' else None)
    traces = runs_dir / f'{args.split}.{args.model}.{args.run}'
    out = Path(f'{traces}{shard_suffix(args.part)}.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.model == 'diffusion':
        run_diffusion(found, model, tokenizer, out, traces, args.steps, args.batch, args.trace, args.states, args.channel, args.hold, args.donors, args.sampler)

    else:
        with open(out, 'w', encoding='utf-8') as file:
            write_ar(found, model, tokenizer, file, args.batch)