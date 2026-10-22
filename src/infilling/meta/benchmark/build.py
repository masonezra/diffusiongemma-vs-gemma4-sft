'''
Builds the infilling data: Python programs with 1 to 3 gaps {whole lines, an expression or a character span; masking.py}
and the problem statement always shown, kept only when their tests run and pass.
One program per exact or near copy (names renamed; programs.NearCopies), none copied from a benchmark, then split by
a hash of each program's group (1% val, 3% test, rest train); the draw keeps 1,000 val and 5,000 test rows and returns
the rest to train, and training drops copies of val and test programs.
A group {a contest problem, or the set of function names a program defines} lies in one split and has a quota:
training keeps up to group_caps distinct programs of it {10 per contest problem, 1 per function-name set, so a task
reworded under the same function name appears once}, val and test one. Each source is read whole in a seeded shuffled
order, so the pool holds every program that passes {over 300,000 rows}; a training recipe takes a prefix if it needs
fewer. Workers {--cores but spare_cores, single-threaded BLAS, the bare `tokenizers` model} parse,
deduplicate, mask and tokenise candidates, which one process admits in input order; a group's candidates are then
tested in waves of its open quota, so a group with 10 slots and 30 candidates runs about 11 test jobs rather than 30,
and a held-out group with one slot about one. At most group_tries candidates per slot are ever admitted. The result
depends on the shuffled order and test outcomes, not on the number of workers; a test near its timeout can still
differ between machines, so the hashes in report.json rule. --cores sets the build's processes and test threads, for
a machine shared with other work {pin the process to that many cores}.
    python -m infilling.meta.benchmark.build [--limit N] [--out DIR] [--cores N]
Reference: https://github.com/DreamLM/DreamOn
Reference: https://arxiv.org/pdf/2505.18789
'''

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import random
from pathlib import Path

from common.files import read_jsonl, seal
from common.models import count_tokens, gemma
from infilling.meta.benchmark import panel, plus, sandbox
from infilling.meta.benchmark.loader import benchmarks, fetch_dataset, fetch_eval, revisions, train_sources
from infilling.meta.benchmark.masking import block_share, gap_counts, linked, linked_share, mask, max_share, min_chars, weights
from infilling.meta.benchmark.programs import NearCopies, as_script, catalogue, clean, copied, names, tidy
from infilling.meta.benchmark.sandbox import parse, run_many, safe, sound, test_runners
from infilling.meta.config.prompts import chat, solution
from infilling.meta.config.settings import data_dir, fits, max_answer, max_total

sides = {0: 'val', 1: 'test', 2: 'test', 3: 'test'}
held_out = {'val': 1000, 'test': 5000}
fenced = {'opencodeinstruct'}
group_caps = {'codecontests': 10}
group_tries = 3
copy_share = 0.5
spare_cores = 4
train_processes = 8
worker_env = {'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
context = {}

class Tokens:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(hf_hub_download(gemma, 'tokenizer.json'))
        self.all_special_tokens = [token.content for token in self.tokenizer.get_added_tokens_decoder().values() if token.special]

    def __call__(self, texts, add_special_tokens=False):
        return {'input_ids': [encoding.ids for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)]}

def tokenizer():
    if 'tokenizer' not in context:
        context['tokenizer'] = Tokens()

    return context['tokenizer']

def side(group, seed):
    return sides.get(int(hashlib.sha1(f'{seed}:{group}'.encode()).hexdigest()[:12], 16) % 100, 'train')

def quota(source, split):
    return group_caps.get(source, 1) if split == 'train' else 1

def bound(source, split):
    return quota(source, split) * group_tries

def tally(row):
    return ['rows', row['strategy'], row['source'], f'{row["gaps"]} gaps'] + ['linked'] * row['linked']

def record(uid, source, strategies, parts, middles, tests, task='', check='asserts', tied=False):
    hidden = round(sum(map(len, middles)) / sum(map(len, parts + middles)), 3)
    strategy = strategies[0] if len(set(strategies)) == 1 else 'mixed'

    return {'id': uid, 'source': source, 'strategy': strategy, 'gaps': len(middles), 'linked': tied, 'hidden': hidden, 'check': check, 'messages': chat(parts, middles, task), 'tests': tests}

def eval_rows(name):
    for uid, kind, prefix, middle, suffix, tests, check, task, group in fetch_eval(name):
        yield {'benchmark': name, 'group': group, **record(uid, name, [kind], [prefix, suffix], [middle], tests, task, check)}

def setup(held, known, seed):
    context.update(held=held, known=known, seed=seed, near=NearCopies())

def workers(held, known, seed, processes=None):
    os.environ.update(worker_env)

    return multiprocessing.get_context('spawn').Pool(processes or max(1, sandbox.cores - spare_cores), initializer=setup, initargs=(held, known, seed))

def marked(row, specials):
    return any(special in message['content'] for message in row['messages'] for special in specials)

def prepare(source, uid, text, tests, check, task, problem):
    try:
        if check == 'pytest':
            tests, check = as_script(tests), 'asserts'

        code, reason = (None, 'no_script_tests') if tests is None else clean(text) if source in fenced else tidy(text)

        if reason:
            return {'reason': reason}

        tree, script = parse(code), check == 'io'

        if not script and names(tree) & context['held']:
            return {'reason': 'benchmark_name'}

        keys = context['near'].keys(tree)
        group = problem or ' '.join(sorted(names(tree))) or keys[0]
        late = 'benchmark_copy' if copied(tree, context['known'], copy_share) else None if safe(code, script) and (script or safe(tests, extra=test_runners)) else 'unsafe'

        if late:
            return {'keys': keys, 'group': group, 'late': late}

        gaps = mask(code, tree, random.Random(f'{context["seed"]}:{uid}'), script=script)

        if not gaps:
            return {'keys': keys, 'group': group, 'late': 'no_gap'}

        edges = [0] + [edge for _, start, end in gaps for edge in (start, end)] + [len(code)]
        chunks = [code[start:end] for start, end in zip(edges, edges[1:])]
        row = record(f'{source}:{uid}', source, [strategy for strategy, _, _ in gaps], chunks[::2], chunks[1::2], tests, task, check, len(gaps) > 1 and linked(code, tree, gaps))
        count_tokens([row], tokenizer())
        late = 'special_token' if marked(row, tokenizer().all_special_tokens) else None if fits(row) else 'too_long'

        if late:
            return {'keys': keys, 'group': group, 'late': late}

        return {'keys': keys, 'group': group, 'late': None, 'code': code, 'row': {'group': group, **row}}

    except Exception as error:
        return {'reason': f'error: {type(error).__name__}'}

def admit(source, found, counts, seen, filled, seed):
    reason, split = found.get('reason'), None

    if not reason:
        split = side(found['group'], seed)
        reason = 'group_full' if filled[found['group']] >= bound(source, split) else seen.copy_of(found['keys']) or found['late']

    if reason:
        counts[reason] += 1
        return None

    filled[found['group']] += 1
    seen.add(found['keys'])

    return split, found['group'], found['code'], found['row']

def candidates(source, counts, seen, filled, pool, limit, seed, chunk=4096):
    records = fetch_dataset(source, limit, seed)

    while True:
        batch = []

        for fields in records:
            counts['read'] += 1
            problem = fields[-1]

            if problem and filled[problem] >= bound(source, side(problem, seed)):
                counts['group_full'] += 1
                continue

            batch.append(fields)

            if len(batch) == chunk:
                break

        if not batch:
            return

        for found in pool.starmap(prepare, [(source, *fields) for fields in batch], chunksize=16):
            made = admit(source, found, counts, seen, filled, seed)

            if made:
                yield made

def verified(pending, counts):
    train = [item for item in pending if item[0] == 'train']
    passed = run_many([(code, row['tests'], row['check']) for _, _, code, row in train])
    counts['fails_tests'] += len(passed) - sum(passed)
    good = {row['id'] for row in sound([row for split, _, _, row in pending if split != 'train'], counts)}

    return [(split, group, row) for (split, group, _, row), ok in zip(train, passed) if ok] + [(split, group, row) for split, group, _, row in pending if row['id'] in good]

def source_rows(source, counts, seen, filled, need, pool, limit, seed, batch=2000):
    waiting, queued = {}, 0

    def wave():
        nonlocal queued
        picked = []

        for group in list(waiting):
            picked += [waiting[group].popleft() for _ in range(min(need[group], len(waiting[group])))]

            if not waiting[group]:
                del waiting[group]

        queued -= len(picked)
        kept = verified(picked, counts)

        for split, group, _ in kept:
            need[group] -= 1
            counts[split] += 1

            if need[group] == 0:
                filled[group] = bound(source, split)
                spare = len(waiting.pop(group, ()))
                queued -= spare
                counts['group_full'] += spare

        return kept

    for split, group, code, row in candidates(source, counts, seen, filled, pool, limit, seed):
        need.setdefault(group, quota(source, split))
        waiting.setdefault(group, collections.deque()).append((split, group, code, row))
        queued += 1

        if queued >= batch:
            yield from wave()

    while waiting:
        yield from wave()

def draw_held(rows, size, rng):
    pools = collections.defaultdict(list)

    for row in rows:
        pools[row['source']].append(row)

    picked, rest = [], []

    for source, pool in pools.items():
        rng.shuffle(pool)
        share = size // len(pools)
        picked += pool[:share]
        rest += pool[share:]

    rng.shuffle(rest)

    return picked + rest[:size - len(picked)]

def held_copy(line):
    tree = parse(solution(json.loads(line)))

    return tree is not None and copied(tree, context['known'], copy_share)

def write_train(pool, out, seed):
    known = catalogue(read_jsonl(out / 'val.jsonl') + read_jsonl(out / 'test.jsonl') + [row for row in read_jsonl(out / 'eval.jsonl') if row['benchmark'] in benchmarks])
    kept, lines, dropped = collections.Counter(), [], collections.Counter()

    with open(pool, encoding='utf-8') as source:
        found = source.readlines()

    with workers(set(), known, seed, train_processes) as checking:
        for line, copy in zip(found, checking.imap(held_copy, found, chunksize=256)):
            if copy:
                dropped['held_copy'] += 1

            else:
                lines.append(line)
                kept.update(tally(json.loads(line)))

    random.Random(f'{seed}:train').shuffle(lines)

    with open(out / 'train.jsonl', 'w', encoding='utf-8', newline='\n') as target:
        target.writelines(lines)

    return {'train_dropped': dict(dropped), 'train': dict(kept)}

def build(limit=0, seed=0, out=data_dir):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    counts = collections.defaultdict(collections.Counter)
    pool = out / 'train.pool'
    files = {name: open(out / f'{name}.jsonl', 'w', encoding='utf-8', newline='\n') for name in ('val', 'test', 'eval')}
    files['train'] = open(pool, 'w', encoding='utf-8', newline='\n')
    pools = {split: [] for split in held_out}

    benchmark_rows = {name: list(eval_rows(name))[:limit or None] for name in benchmarks}
    known = catalogue([row for rows in benchmark_rows.values() for row in rows])
    held = set().union(*(names(tree) for tree in map(parse, {solution(row) for row in benchmark_rows['humaneval']}) if tree))
    seen, filled, need = NearCopies(), collections.Counter(), {}

    def to_train(row):
        files['train'].write(json.dumps({**row, 'tests': ''}) + '\n')

    with workers(held, known, seed) as preparing:
        for source in train_sources:
            for split, group, row in source_rows(source, counts[source], seen, filled, need, preparing, limit, seed):
                if split == 'train':
                    to_train(row)

                else:
                    pools[split].append(row)

            print(source, dict(counts[source]), flush=True)

    for split, rows in pools.items():
        drawn = draw_held(rows, held_out[split], random.Random(f'{seed}:{split}'))
        chosen = {row['id'] for row in drawn}

        for row in rows:
            if row['id'] in chosen:
                files[split].write(json.dumps(row) + '\n')
                counts[split].update(tally(row))

            else:
                to_train(row)

    for name, rows in benchmark_rows.items():
        rows = list(sound(rows, counts[name])) if name == 'safim' else rows
        count_tokens(rows, tokenizer())

        for row in rows:
            files['eval'].write(json.dumps(row) + '\n')
            counts[name].update(['rows', row['strategy']])
            counts[name]['over_canvas'] += row['tokens'] > max_answer

    for file in files.values():
        file.close()

    settings = {
        'seed': seed, 'group_caps': group_caps, 'group_tries': group_tries, 'copy_share': copy_share, 'held_out': held_out,
        'gap_counts': gap_counts, 'gap_weights': weights, 'min_chars': min_chars, 'linked_share': linked_share, 'block_share': block_share, 'max_share': max_share,
        'max_answer': max_answer, 'max_total': max_total, 'revisions': revisions,
    }
    report = {'settings': settings, **{name: dict(count) for name, count in counts.items()}, **write_train(pool, out, seed)}
    pool.unlink()
    seal(out, **report)
    seal(out, humanevalplus=plus.build(limit, out=out))

    return seal(out, panel=panel.build(out, tokenizer()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default=data_dir)
    parser.add_argument('--cores', type=int, default=os.cpu_count())
    args = parser.parse_args()
    sandbox.cores = args.cores

    print(json.dumps(build(args.limit, out=args.out), indent=1))