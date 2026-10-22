'''
Replies of either model on a rows file: one user turn in, at most --budget new tokens out {default one 256-token canvas; --canvas sets the
diffusion canvas for a recipe trained at another width}, thinking off. Gemma 4
decodes greedily; DiffusionGemma uses the released sampler (at most 48 steps, temperature 0.8 to 0.4, adaptive stop),
seeded per batch from its first row id and --seed. Batched by prompt length; --gpus n runs n shards, one per GPU.
    python -m common.generate diffusion ../dataset/cryptanalysis/test.jsonl --checkpoint DIR --gpus 8 --seed 0 --out diffusion.jsonl
Reference: https://huggingface.co/google/diffusiongemma-26B-A4B-it
'''

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.files import read_jsonl
from common.load import load
from common.models import canvas, end_tokens
from common.shards import join, run_on_gpus

thought = re.compile(r'<\|channel>thought(?:\r?\n)?')

def prompts(rows, tokenizer):
    return tokenizer.apply_chat_template([row['messages'][:1] for row in rows], add_generation_prompt=True, enable_thinking=False,
                                         padding=True, return_tensors='pt', return_dict=True)

def decode(ids, tokenizer, ends):
    ids = ids.tolist()
    stop = min((ids.index(end) for end in ends if end in ids), default=len(ids))
    text, unclosed = tokenizer.decode(ids[:stop], skip_special_tokens=False), False

    # Thinking is off, but a reply can still open a thought channel: closed ones are cut, an unclosed one makes it malformed
    for match in reversed(list(thought.finditer(text))):
        close = text.find('<channel|>', match.end())
        unclosed |= close < 0
        text = text[:match.start()] if close < 0 else text[:match.start()] + text[close + len('<channel|>'):]

    malformed = unclosed or '<|channel>' in text or '<channel|>' in text

    return {'reply': '' if malformed else text.strip(), 'generated_tokens': stop + int(stop < len(ids)), 'terminated': stop < len(ids), 'malformed': malformed}

def seed_for(rows, seed=0):
    key = rows[0]['id'] if not seed else f"{rows[0]['id']}#{seed}"

    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

def batches(rows, size):
    ordered = sorted(rows, key=lambda row: len(row['messages'][0]['content']))

    return [ordered[first:first + size] for first in range(0, len(ordered), size)]

def write(file, group, replies, started):
    seconds = round((time.perf_counter() - started) / len(group), 3)

    for row, found in zip(group, replies):
        file.write(json.dumps({'id': row['id'], **found, 'seconds': seconds}, ensure_ascii=False) + '\n')

    file.flush()

@torch.inference_mode()
def write_ar(rows, model, tokenizer, file, batch=16, budget=canvas):
    ends = list(end_tokens(tokenizer))

    for group in batches(rows, batch):
        started = time.perf_counter()
        asked = prompts(group, tokenizer).to(model.device)
        made = model.generate(**asked, max_new_tokens=budget, do_sample=False, eos_token_id=ends)
        write(file, group, [decode(ids, tokenizer, ends) for ids in made[:, asked['input_ids'].shape[1]:].cpu()], started)

@torch.inference_mode()
def write_diffusion(rows, model, tokenizer, file, batch=16, steps=48, seed=0, width=canvas, budget=canvas):
    ends = [int(end) for end in np.atleast_1d(model.generation_config.eos_token_id)]

    if width != model.config.canvas_length:
        model.config.canvas_length = width

    for group in batches(rows, batch):
        started = time.perf_counter()
        torch.manual_seed(seed_for(group, seed))
        asked = prompts(group, tokenizer).to(model.device)
        made = model.generate(**asked, max_new_tokens=budget, max_denoising_steps=steps, return_dict_in_generate=True)
        write(file, group, [decode(ids, tokenizer, ends) for ids in made.sequences[:, asked['input_ids'].shape[1]:].cpu()], started)

def generate(kind, rows, checkpoint=None, part='0/1', limit=0, batch=16, steps=48, seed=0, out=None, width=canvas, budget=canvas):
    share, shares = map(int, part.split('/'))
    found = read_jsonl(rows)[:limit or None][share::shares]
    model, tokenizer = load(kind, checkpoint)

    with open(out, 'w', encoding='utf-8') as file:
        if kind == 'diffusion':
            write_diffusion(found, model, tokenizer, file, batch, steps, seed, width, budget)

        else:
            write_ar(found, model, tokenizer, file, batch, budget)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('ar', 'diffusion'))
    parser.add_argument('rows')
    parser.add_argument('--checkpoint')
    parser.add_argument('--part', default='0/1')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--canvas', type=int, default=canvas)
    parser.add_argument('--budget', type=int, default=canvas)
    parser.add_argument('--out')
    args = parser.parse_args()
    out = Path(args.out or Path(args.checkpoint or '.') / f'{args.kind}.{Path(args.rows).stem}.jsonl')

    if args.gpus > 1:
        common = [sys.executable, '-m', 'common.generate', args.kind, args.rows, '--limit', str(args.limit), '--batch', str(args.batch), '--steps', str(args.steps), '--seed', str(args.seed), '--canvas', str(args.canvas), '--budget', str(args.budget)]
        common += ['--checkpoint', args.checkpoint] if args.checkpoint else []
        parts = [f'{out}.part{gpu}' for gpu in range(args.gpus)]
        run_on_gpus([common + ['--part', f'{gpu}/{args.gpus}', '--out', part] for gpu, part in enumerate(parts)])
        print(join(parts, out))

    else:
        print(generate(args.kind, args.rows, args.checkpoint, args.part, args.limit, args.batch, args.steps, args.seed, out, args.canvas, args.budget))

if __name__ == '__main__':
    main()