'''
Builds the train, val and test files of one recipe, drops training rows equal to held-out rows, and writes report.json
with settings, counts and SHA-256 hashes. Asserts each answer symbol is one token and each plaintext fits one 256-slot
DiffusionGemma canvas. Recipe 1 {seven levels, key hidden} goes to dataset/cryptanalysis/recipe1; recipe 2 {the same seven
levels, cipher name and key printed} to recipe2, with test_random.jsonl beside its splits: the same rows over uniform random
letters instead of English, the control that a model applies the key rather than a language prior.
    python -m cryptanalysis.meta.benchmark.build --recipe 1
    python -m cryptanalysis.meta.benchmark.build --recipe 2
'''

import argparse
import json
import random
from collections import Counter
from string import ascii_lowercase as letters

from transformers import AutoTokenizer

from common.files import seal
from common.models import gemma
from cryptanalysis.meta.benchmark import decipherment
from cryptanalysis.meta.config import settings
from cryptanalysis.meta.config.settings import budget, budget_keyed, recipe_dirs, texts_each, texts_each_keyed, texts_few

def longest(rows, encoder, room=decipherment.room):
    most = 0

    for first in range(0, len(rows), 512):
        replies = [row['messages'][1]['content'] for row in rows[first:first + 512]]

        for reply, tokens in zip(replies, encoder(replies, add_special_tokens=False)['input_ids']):
            assert len(tokens) == len(reply.split()) <= room, reply[:80]
            most = max(most, len(tokens))

    return most

def noise(count, seed):
    rng = random.Random(seed)
    texts = []

    for _ in range(count):
        words = []

        while sum(map(len, words)) < decipherment.room:
            words.append(''.join(rng.choice(letters) for _ in range(rng.randint(2, 9))))

        texts.append(' '.join(words))

    return texts

def save(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')

def build(limit=0, seed=0, out=None, recipe=1):
    encoder = AutoTokenizer.from_pretrained(gemma)
    keyed = recipe == 2
    out = out or recipe_dirs[recipe]
    size, each = (budget_keyed, texts_each_keyed) if keyed else (budget, texts_each)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    parts = decipherment.texts(limit)
    built = {'val': decipherment.rows(parts['val'], seed + 1, texts_few, keys=True, keyed=keyed),
             'test': decipherment.rows(parts['test'], seed + 2, each, keys=True, keyed=keyed)}
    held = {row['id'] for rows in built.values() for row in rows}

    if keyed:
        built['test_random'] = decipherment.rows(noise(texts_few * len(decipherment.levels), seed + 3), seed + 3, texts_few, keys=True, keyed=True)
    train = [row for row in decipherment.rows(parts['train'], seed, keyed=keyed) if row['id'] not in held]
    assert limit or len(train) >= size, 'too few texts for the budget'
    built = {'train': train[:size], **built}
    counts = {}

    for name, rows in built.items():
        save(out / f'{name}.jsonl', rows)
        counts[name] = {'rows': len(rows), 'rows by level': dict(sorted(Counter(row['level'] for row in rows).items())),
                        'longest reply in tokens': longest(rows, encoder)}

    design = {'recipe': recipe, 'regime': 'keyed' if keyed else 'keyless', 'seed': seed, 'budget': size, 'held_out_each': each, 'held_out_few': texts_few, 'levels': len(decipherment.levels),
              'graded_levels': sorted(decipherment.graded), 'lengths': settings.lengths_keyed if keyed else settings.lengths, 'unicity_margin': settings.margin,
              'redundancy': settings.redundancy, 'held_out_overlap_letters': settings.overlap, 'texts': {name: len(part) for name, part in parts.items()},
              'text_source': {'dataset': settings.source, 'revision': settings.revision, 'held_out_from': settings.release}}

    return seal(out, settings=design, **counts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--recipe', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    print(json.dumps(build(args.limit, recipe=args.recipe), indent=1))