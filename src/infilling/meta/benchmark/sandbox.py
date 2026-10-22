'''
Runs a program against its tests in a separate Python process with a timeout, and decides whether a held-out row's
tests are sound. A crash, or a timeout again on a second run at `retry` times the limit, is a failed test; the second
run keeps a program near the limit from failing only because every core is busy {SAFIM references take up to 4.5 s per
case alone}, so build and scoring give the same verdict on any load; retried=False skips it where a timeout already
gives the wanted verdict {an emptied gap that loops}. A static gate first allows only pure-computation
code. The child runs with a fixed hash seed and UTF-8 streams, sees only PATH, SYSTEMROOT, TEMP and TMP of the
environment {no token reaches tested code} and at most memory_limit of memory {address space by rlimit on Linux, as in
EvalPlus; committed memory by a job object on Windows}, so a runaway mutant or emptied loop cannot exhaust the machine; the
wrapper that sets the cap leaves sys.argv as a direct run would {fileinput reads sys.argv[1:] before stdin}. The gate is a filter, not a security boundary {sys.modules reaches os}: run
the build on a disposable machine that holds no credentials. Converted KodCode tests import pytest, so the interpreter
must have it.
A held-out row is sound when its answer passes twice, every gap left empty fails, and at least one small mutant of the
answer fails; kept rows may still accept some wrong fills.
Reference: https://github.com/evalplus/evalplus/blob/master/evalplus/eval/__init__.py
'''

import ast
import collections
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor

from infilling.meta.config.prompts import answer, blanks, fill, pieces, solution

assert importlib.util.find_spec('pytest'), 'the sandbox interpreter needs pytest: converted KodCode tests import it'

allowed = {
    'math', 'cmath', 'itertools', 'collections', 'functools', 'heapq', 'bisect', 're', 'string', 'typing', 'operator', 'fractions', 'decimal', 'statistics', 'copy', 'dataclasses', 'enum', 'json', 'abc', 'numbers', '__future__',
    'random', 'datetime', 'time', 'calendar', 'sys', 'array', 'queue', 'struct', 'hashlib', 'base64', 'binascii', 'uuid', 'secrets', 'textwrap', 'unicodedata', 'difflib', 'html', 'ipaddress', 'pprint', 'types', 'contextlib',
}
reading = {'io'}
test_runners = frozenset({'pytest', 'unittest'})
output_limit = 1_000_000
retry = 2
cores = os.cpu_count()
memory_limit = 4 << 30
environment = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'TEMP', 'TMP') if key in os.environ} | {'PYTHONHASHSEED': '0'}
flags = ['-s', '-P', '-X', 'utf8']
posix_cap = f'import resource; resource.setrlimit(resource.RLIMIT_AS, ({memory_limit}, {memory_limit}))'
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION is 144 bytes on x64: LimitFlags at offset 16 takes JOB_OBJECT_LIMIT_PROCESS_MEMORY
# (0x100) and ProcessMemoryLimit sits at offset 112; information class 9 is JobObjectExtendedLimitInformation
windows_cap = (
    'import ctypes, struct; kernel = ctypes.windll.kernel32; kernel.CreateJobObjectW.restype = kernel.GetCurrentProcess.restype = ctypes.c_void_p; '
    f'limits = bytearray(144); struct.pack_into("I", limits, 16, 0x100); struct.pack_into("Q", limits, 112, {memory_limit}); job = ctypes.c_void_p(kernel.CreateJobObjectW(None, None)); '
    'assert kernel.SetInformationJobObject(job, 9, (ctypes.c_char * 144).from_buffer(limits), 144) and kernel.AssignProcessToJobObject(job, ctypes.c_void_p(kernel.GetCurrentProcess()))'
)
limited = ['-c', f'{posix_cap if os.name == "posix" else windows_cap}; import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")']
banned = {'open', 'input', 'eval', 'exec', 'compile', 'exit', 'quit', 'breakpoint', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', 'help', 'hash'}
number = re.compile(r'(?<![\w.])\d+(?![\w.])')
swaps = (('<=', '<'), ('>=', '>'), ('==', '!='), ('!=', '=='), (' + ', ' - '), (' - ', ' + '), ('True', 'False'), ('False', 'True'), (' and ', ' or '), (' or ', ' and '))

def parse(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)

            return ast.parse(code)

    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None

def safe(code, script=False, extra=frozenset()):
    tree = parse(code)
    modules, words = (allowed | reading, banned - {'input', 'exit', 'quit'}) if script else (allowed | extra, banned)

    if tree is None:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split('.')[0] not in modules for alias in node.names):
            return False

        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').split('.')[0] not in modules):
            return False

        if isinstance(node, ast.Name) and (node.id in words or node.id.startswith('__')) and node.id != '__name__':
            return False

        if isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr.endswith('__'):
            return False

    return True

def execute(code, feed, timeout, capture=False):
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = os.path.join(folder, 'main.py')

            with open(path, 'w', encoding='utf-8') as file:
                file.write(code)

            with open(os.path.join(folder, 'output'), 'w+b') as output:
                done = subprocess.run([sys.executable, *flags, *limited, path], input=feed.encode(), stdout=output if capture else subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, cwd=folder, env=environment)
                output.seek(0)

                return done.returncode, output.read(output_limit)

    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None

def patient(code, feed, timeout, capture=False):
    return execute(code, feed, timeout, capture) or execute(code, feed, retry * timeout, capture)

def run(code, tests, check='asserts', timeout=10, retried=True):
    once = patient if retried else execute

    if check == 'asserts':
        done = once(f'{code}\n\n{tests}', '', timeout)

        return done is not None and done[0] == 0

    for case in json.loads(tests):
        done = once(code, case['input'], timeout, capture=True)

        if done is None or done[0] or done[1].decode(errors='replace').split() not in [output.split() for output in case['output']]:
            return False

    return True

def attempt(program):
    try:
        return run(*program)

    except Exception:
        return False

def run_many(programs, workers=None):
    with ThreadPoolExecutor(workers or cores) as pool:
        return list(pool.map(attempt, programs))

def mutants(text, rng, most=4):
    found = [text[:at] + new + text[at + len(old):] for old, new in swaps for at in range(len(text)) if text.startswith(old, at)]
    found += [text[:match.start()] + str(int(match.group()) + 1) + text[match.end():] for match in number.finditer(text)]
    lines = text.split('\n')

    if sum(bool(line.strip()) for line in lines) > 1:
        found += ['\n'.join(lines[:at] + lines[at + 1:]) for at, line in enumerate(lines) if line.strip()]

    found = [wrong for wrong in dict.fromkeys(found) if wrong != text]
    rng.shuffle(found)

    return found[:most]

def blind(rows, seed=0, most=4):
    wrong = []

    for at, row in enumerate(rows):
        rng, middles = random.Random(f'{seed}:{row["id"]}'), pieces(row['messages'][1]['content'])
        altered = [middles[:place] + [mutant] + middles[place + 1:] for place, middle in enumerate(middles) for mutant in mutants(middle, rng)]
        wrong += [(at, fill(row['messages'][0]['content'], answer(reply))) for reply in rng.sample(altered, min(most, len(altered)))]

    wrong = [(at, program) for at, program in wrong if parse(program) is not None]
    passed = run_many([(program, rows[at]['tests'], rows[at]['check']) for at, program in wrong])
    tried, caught = collections.Counter(), collections.Counter()

    for (at, _), ok in zip(wrong, passed):
        tried[at] += 1
        caught[at] += not ok

    return [bool(tried[at]) and not caught[at] for at in range(len(rows))]

def sound(rows, counts):
    twice = run_many([(solution(row), row['tests'], row['check']) for row in rows] * 2)
    passed = [once and again for once, again in zip(twice[:len(rows)], twice[len(rows):])]
    emptied = [(at, program) for at, row in enumerate(rows) for program in blanks(row)]
    unnoticed = {at for (at, _), ok in zip(emptied, run_many([(program, rows[at]['tests'], rows[at]['check']) for at, program in emptied])) if ok}

    for at, (row, fine, weak) in enumerate(zip(rows, passed, blind(rows))):
        if not fine:
            counts['answer_fails'] += 1

        elif at in unnoticed:
            counts['tests_miss_gap'] += 1

        elif weak:
            counts['tests_miss_errors'] += 1

        else:
            yield row