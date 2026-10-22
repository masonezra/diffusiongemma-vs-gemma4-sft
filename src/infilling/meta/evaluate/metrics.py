'''
Scores replies by execution. Headline: pass, the program with the reply spliced in passes its tests. Exact match,
similarity and gap-length error are descriptive. Rates are split by gap count, linking, gap kind and source.
    python -m infilling.meta.evaluate.metrics test replies.jsonl
Reference: https://github.com/gonglinyuan/safim
Reference: https://arxiv.org/pdf/2411.00640
'''

import argparse
import collections
import difflib
import json
import random
from pathlib import Path

from common.bootstrap import interval as bootstrap
from common.files import read_jsonl
from infilling.meta.benchmark.masking import dependence
from infilling.meta.benchmark.sandbox import parse, run_many, safe
from infilling.meta.config.prompts import block, fill, heading, pieces, shown, solution
from infilling.meta.config.settings import data_dir, max_answer, max_total, timeout

sizes = ((1, '1 line'), (4, '2-4 lines'), (float('inf'), '5+ lines'))

def harmless(row, program):
    script = row['check'] == 'io'

    return safe(program, script) or not safe(solution(row), script)

def passed(rows, replies):
    filled = [fill(row['messages'][0]['content'], reply) for row, reply in zip(rows, replies)]
    run = [harmless(row, program) for row, program in zip(rows, filled)]
    result = iter(run_many([(program, row['tests'], row['check'], timeout(row)) for row, program, ok in zip(rows, filled, run) if ok]))

    return [ok and next(result) for ok in run]

def outcomes(rows, replies):
    squeeze = lambda text: ''.join(text.split())
    found = {'pass': [bool(ok) for ok in passed(rows, replies)], 'exact': [], 'similar': [], 'length off': []}

    for row, reply in zip(rows, replies):
        middles = pieces(row['messages'][1]['content'])
        written = (pieces(reply) + [''] * len(middles))[:len(middles)]
        wanted, given = '\n'.join(middles), '\n'.join(pieces(reply))
        found['exact'].append(squeeze(given) == squeeze(wanted))
        found['similar'].append(difflib.SequenceMatcher(None, wanted, given, autojunk=False).ratio())
        found['length off'].append(sum(min(1, abs(len(mine.strip()) - len(theirs.strip())) / max(1, len(theirs.strip()))) for mine, theirs in zip(written, middles)) / len(middles))

    return found

def links(row):
    context, middles = shown(row['messages'][0]['content']), pieces(row['messages'][1]['content'])
    code, gaps, at = solution(row), [], 0

    for part, middle in zip(context, middles):
        at += len(part)
        gaps.append((None, at, at + len(middle)))
        at += len(middle)

    tree = parse(code)

    return dependence(code, tree, gaps) if tree is not None else {}

def interval(rows, values, draws=1000, seed=0):
    groups, low, high = bootstrap(values, [row.get('group', row['id']) for row in rows], draws, random.Random(seed))

    return {'groups': groups, 'low': round(100 * low, 1), 'high': round(100 * high, 1)}

def score(rows, replies):
    sums, total, faults = collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter()
    found = outcomes(rows, replies)

    for at, (row, reply) in enumerate(zip(rows, replies)):
        question = row['messages'][0]['content']
        lines = '\n'.join(pieces(row['messages'][1]['content'])).strip('\n').count('\n') + 1
        size = next(name for most, name in sizes if lines <= most)
        room = 'fits canvas' if row['tokens'] <= max_answer else 'over canvas'
        reach = 'within training length' if row['total_tokens'] <= max_total else 'over training length'
        keys = ('all', f'{row["gaps"]} gap' + 's' * (row['gaps'] > 1), row['strategy'], size, room, reach, 'task shown' if heading in question else 'code only', row['source'])
        tied = links(row) if row['gaps'] > 1 else {}
        keys += ('several gaps', 'linked gaps' if row['linked'] else 'separate gaps') if row['gaps'] > 1 else ()
        keys += ('linked, name hidden',) * bool(tied.get('hidden')) + ('backward link',) * bool(tied.get('backward'))
        keys += (row['cell'], f"{row['cell']}, {row['variant']}") if 'cell' in row else ()

        for key in keys:
            sums[key].update({name: found[name][at] for name in found})
            total[key] += 1

        faults['unclosed'] += bool(reply) and len(block.findall(reply)) < row['gaps']
        faults['broken'] += not found['pass'][at] and parse(fill(question, reply)) is None

    result = {key: {'rows': total[key], **{name: round(100 * sums[key][name] / total[key], 1) for name in found}} for key in total}
    result['all']['pass_interval'] = interval(rows, found['pass'])
    result['faults'] = {name: round(100 * faults[name] / len(rows), 1) for name in ('unclosed', 'broken')}

    return result

def load_replies(source, rows):
    if source in ('truth', 'empty'):
        return [row['messages'][1]['content'] if source == 'truth' else '' for row in rows]

    given = {reply['id']: reply['reply'] for reply in read_jsonl(source)}
    missing = [row['id'] for row in rows if row['id'] not in given]
    assert not missing, f'{len(missing)} rows have no reply in {source}, the first {missing[0]}'

    return [given[row['id']] for row in rows]

def select(rows, benchmark=None, several=False):
    return [row for row in rows if (not benchmark or row.get('benchmark') == benchmark) and (not several or row['gaps'] > 1)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('replies')
    parser.add_argument('--benchmark')
    parser.add_argument('--out', default=data_dir)
    args = parser.parse_args()
    rows = select(read_jsonl(Path(args.out) / f'{args.split}.jsonl'), args.benchmark)
    benchmarks = sorted({row.get('benchmark', args.split) for row in rows})
    result = {name: score(found, load_replies(args.replies, found)) for name in benchmarks for found in [[row for row in rows if row.get('benchmark', args.split) == name]]}

    print(json.dumps(result[benchmarks[0]] if len(benchmarks) == 1 else result, indent=1))