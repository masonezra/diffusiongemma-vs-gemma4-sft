'''
Rescores the HumanEval-Infilling rows under the stronger HumanEval+ tests, so a wrong fill that passes the original
tests is caught. Same row ids, so one set of rows is scored under two test strengths. Every row of a task rebuilds the
same reference, so each distinct program runs once and its verdict is shared {164 answers, not 8,488}. The answer runs
a second time and the empty gaps run only where the first run passed {a timed-out reference, e.g. the exponential fib
and fibfib under large inputs, would otherwise cost three timeouts for the same verdict}, and an empty gap runs once
with no retry, since a timeout already means the tests notice it {an emptied loop step never ends}.
Reference: https://arxiv.org/abs/2305.01210
Reference: https://huggingface.co/datasets/evalplus/humanevalplus
'''

import collections
from pathlib import Path

from common.files import iter_jsonl, merge_jsonl
from infilling.meta.benchmark.loader import load
from infilling.meta.benchmark import sandbox
from infilling.meta.benchmark.sandbox import run_many
from infilling.meta.config.prompts import blanks, solution
from infilling.meta.config.settings import data_dir, timeout

def plus_tests():
    return {row['task_id']: f"{row['test']}\ncheck({row['entry_point']})" for row in load('evalplus/humanevalplus', split='test')}

def verdicts(jobs, workers, retried=True):
    unique = list(dict.fromkeys(jobs))
    found = dict(zip(unique, run_many([job + (retried,) for job in unique], workers)))

    return [found[job] for job in jobs]

def build(limit=0, out=data_dir):
    out = Path(out)
    tests = plus_tests()
    rows = [{**row, 'benchmark': 'humanevalplus', 'tests': tests[row['group']]} for row in iter_jsonl(out / 'eval.jsonl') if row['benchmark'] == 'humaneval' and row['group'] in tests]
    rows = rows[:limit or None]
    workers = max(1, sandbox.cores // 2)
    answers = [(solution(row), row['tests'], row['check'], timeout(row)) for row in rows]
    once = verdicts(answers, workers)
    again = iter(verdicts([job for job, ok in zip(answers, once) if ok], workers))
    passed = [ok and next(again) for ok in once]
    emptied = [(at, program) for at, row in enumerate(rows) if passed[at] for program in blanks(row)]
    unnoticed = {at for (at, _), ok in zip(emptied, verdicts([(program, rows[at]['tests'], rows[at]['check'], timeout(rows[at])) for at, program in emptied], workers, retried=False)) if ok}
    counts, kept = collections.Counter(read=len(rows)), []

    for at, (row, fine) in enumerate(zip(rows, passed)):
        if not fine:
            counts['answer_fails'] += 1

        elif at in unnoticed:
            counts['tests_miss_gap'] += 1

        else:
            kept.append(row)

    counts['rows'] = len(kept)
    merge_jsonl(out / 'eval.jsonl', kept, 'benchmark', 'humanevalplus')

    return dict(counts)