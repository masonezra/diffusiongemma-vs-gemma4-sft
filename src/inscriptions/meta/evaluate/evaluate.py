'''
Scores a reply file (or a baseline, or the targets themselves) on a split, or compares two reply files pairwise.
Protocol: score the final checkpoint of each arm once on the first 5,000 test rows.
    python -m inscriptions.meta.evaluate.evaluate test ar.jsonl --against diffusion.jsonl
'''

import argparse
import json
from pathlib import Path

from common.files import read_jsonl
from inscriptions.meta.config.prompts import target_text
from inscriptions.meta.config.settings import data_dir
from inscriptions.meta.evaluate.baseline import baseline, names as baselines
from inscriptions.meta.evaluate.compare import compare
from inscriptions.meta.evaluate.metrics import score_rows

def load_replies(source, rows, out):
    if source == 'truth':
        return [target_text(row) for row in rows]

    if source in baselines:
        return baseline(source, read_jsonl(out / 'train.jsonl'), rows)

    replies = {reply['id']: reply['reply'] for reply in read_jsonl(source)}
    missing = [row['id'] for row in rows if row['id'] not in replies]
    assert not missing, f'{len(missing)} rows have no reply in {source}, the first {missing[0]}'

    return [replies[row['id']] for row in rows]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', nargs='?', default='test')
    parser.add_argument('replies', nargs='?', default='truth')
    parser.add_argument('--against')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--out', default=data_dir)
    args = parser.parse_args()
    out = Path(args.out)
    rows = read_jsonl(out / f'{args.split}.jsonl')[:args.limit or None]
    replies = load_replies(args.replies, rows, out)
    result = compare(rows, replies, load_replies(args.against, rows, out), args.resamples) if args.against else score_rows(rows, replies)

    print(json.dumps(result, indent=1))