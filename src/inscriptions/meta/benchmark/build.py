'''
Builds train, val and test JSONL from EDCS: carved text in, the scholar's edition out, one row per stone.
Some surviving letters are hidden first, so restoration is scored against the stone; the model is not told which gaps
are hidden or how long any gap is. Rows over the token caps are dropped; report.json records counts and SHA-256 hashes.
    python -m inscriptions.meta.benchmark.build
Reference: https://github.com/google-deepmind/predictingthepast/blob/main/train/dataloader.py
Reference: https://www.nature.com/articles/s41586-025-09292-5
'''

import argparse
import collections
import itertools
import json
import random
from pathlib import Path

from common.files import seal, write_jsonl
from common.models import count_tokens, gemma
from inscriptions.meta.benchmark import source
from inscriptions.meta.benchmark.augment import max_share, restoration_stats, sample_hidden
from inscriptions.meta.benchmark.markup import editor_additions
from inscriptions.meta.benchmark.split import split_items
from inscriptions.meta.config.prompts import to_messages
from inscriptions.meta.config.settings import data_dir, fits, max_answer_tokens, max_total_tokens

splits = ('train', 'val', 'test')
held_rows = {'val': 3000}

def make_row(item, marks, hidden=()):
    expansions, restorations = editor_additions(marks)

    return {'id': item['id'], 'province': item['province'], 'funerary': item['funerary'], 'abbreviations': len(expansions), 'gaps': len(restorations),
            'hidden': list(hidden), 'messages': to_messages(marks)}

def make_rows(items, lengths, counts, rng):
    rows = []

    for item in items:
        marks, hidden = sample_hidden(item, lengths, counts, rng)

        if hidden:
            rows.append(make_row(item, marks, hidden))

        elif item['has_additions']:
            rows.append(make_row(item, item['marks']))

    return rows

def summarise(rows):
    summary = collections.Counter(rows=len(rows))

    for row in rows:
        summary['with hidden letters'] += bool(row['hidden'])
        summary['abbreviations'] += row['abbreviations']
        summary['gaps'] += row['gaps']

    return summary

def write_split(name, rows, rng, tokenizer, out, report):
    count_tokens(rows, tokenizer)
    rng.shuffle(rows)
    kept = [row for row in rows if fits(row)]
    report['dropped']['too_long'] += len(rows) - len(kept)
    kept = kept[:held_rows.get(name)]
    report[name] = summarise(kept)
    write_jsonl(out / f'{name}.jsonl', kept)

def build(limit=0, seed=0, out=data_dir):
    from transformers import AutoTokenizer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(gemma)
    report = collections.defaultdict(collections.Counter)
    items = list(source.load_items(itertools.islice(source.records(), limit or None), report['edcs']))
    parts = split_items(items, report['edcs'])
    lengths, counts = restoration_stats(parts['train'])

    for split in splits:
        rng = random.Random(f'{seed}:{split}')
        write_split(split, make_rows(parts[split], lengths, counts, rng), rng, tokenizer, out, report)

    settings = {'seed': seed, 'held_rows': held_rows, 'hidden_share': max_share, 'max_answer_tokens': max_answer_tokens,
                'max_total_tokens': max_total_tokens, 'source': {'zenodo_record': 7072337, 'file': source.file, 'md5': source.md5}}

    return seal(out, settings=settings, **{name: dict(count) for name, count in report.items()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default=data_dir)
    args = parser.parse_args()

    print(json.dumps(build(args.limit, out=args.out), indent=1))