'''
Paired dependency panel: two statement holes in one test-program function, each of at least masking.min_chars
non-space characters, either dependent (the later reads a name the earlier binds, and that name occurs as a word
nowhere in the shown code nor in the task text) or independent and length-matched, each shown with both hidden,
first given and second given. Functions whose statements share a line are skipped, so every hole is whole lines.
Readout per model: the cost of filling both against one given, on dependent against independent pairs, paired within
a program.
Reference: https://arxiv.org/pdf/2204.05999
Reference: https://arxiv.org/pdf/2410.03103
'''

import ast
import collections
import re
from itertools import accumulate
from pathlib import Path

from common.files import merge_jsonl, read_jsonl
from common.models import count_tokens
from infilling.meta.benchmark.masking import min_chars
from infilling.meta.benchmark.sandbox import parse, run_many
from infilling.meta.config.prompts import chat, solution, task_of
from infilling.meta.config.settings import fits

cells = ('dependent', 'independent')
variants = ('both', 'first given', 'second given')

def functions(tree):
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

def statements(function):
    body = function.body[1:] if ast.get_docstring(function, clean=False) is not None else function.body
    body = [node for node in body if node.lineno > function.lineno]

    return body if all(first.end_lineno < second.lineno for first, second in zip(body, body[1:])) else []

def span(code, node):
    lines = code.split('\n')
    starts = list(accumulate((len(line) + 1 for line in lines), initial=0))

    return starts[node.lineno - 1], starts[node.end_lineno - 1] + len(lines[node.end_lineno - 1])

def size(code, nodes):
    return sum(end - start for start, end in (span(code, node) for node in nodes))

def stores(node):
    found = set()

    for inner in ast.walk(node):
        targets = getattr(inner, 'targets', None) or [getattr(inner, 'target', None)]

        if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.NamedExpr, ast.comprehension)):
            found |= {name.id for target in targets if target is not None for name in ast.walk(target) if isinstance(name, ast.Name)}

        if isinstance(inner, ast.withitem) and inner.optional_vars is not None:
            found |= {name.id for name in ast.walk(inner.optional_vars) if isinstance(name, ast.Name)}

    return found

def loads(node):
    return {inner.id for inner in ast.walk(node) if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)}

def parameters(function):
    arguments = function.args

    return {arg.arg for arg in arguments.args + arguments.kwonlyargs + arguments.posonlyargs} | {arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg}

def dependent_pairs(function, body):
    params = parameters(function)

    for i, first in enumerate(body):
        bound = stores(first) - params

        for j in range(i + 2, len(body)):
            linking = bound & loads(body[j]) - set().union(*(stores(body[k]) for k in range(i + 1, j)))

            if linking:
                yield i, j, linking

def hidden(code, nodes, linking, task):
    shown = '\n'.join(cut(code, nodes)[0] + [task])

    return any(not re.search(rf'\b{re.escape(name)}\b', shown) for name in linking)

def substantial(code, node):
    start, end = span(code, node)

    return len(''.join(code[start:end].split())) >= min_chars

def independent_pair(body, code, target, big):
    best = None

    for a, first in enumerate(body):
        for b in range(a + 2, len(body)):
            second = body[b]

            if not {a, b} <= big or stores(first) & (loads(second) | stores(second)) or stores(second) & loads(first):
                continue

            gap = abs(size(code, (first, second)) - target)

            if best is None or gap < best[0]:
                best = (gap, a, b)

    return best and best[1:]

def cut(code, nodes):
    edges = [0] + [edge for node in nodes for edge in span(code, node)] + [len(code)]
    chunks = [code[start:end] for start, end in zip(edges, edges[1:])]

    return chunks[::2], chunks[1::2]

def family_rows(row):
    try:
        return pairs(row)

    except Exception:
        return []

def pairs(row):
    code, task = solution(row), task_of(row['messages'][0]['content'])
    tree = parse(code)

    for function in functions(tree) if tree else []:
        body = statements(function)
        big = {at for at, node in enumerate(body) if substantial(code, node)}

        for i, j, linking in dependent_pairs(function, body):
            if not {i, j} <= big or not hidden(code, (body[i], body[j]), linking, task):
                continue

            other = independent_pair(body, code, size(code, (body[i], body[j])), big)

            if not other:
                break

            made = []

            for cell, (first, second) in zip(cells, ((i, j), other)):
                context, middles = cut(code, [body[first], body[second]])
                made.append((cell, 'both', context, middles))
                made.append((cell, 'first given', [context[0] + middles[0] + context[1], context[2]], [middles[1]]))
                made.append((cell, 'second given', [context[0], context[1] + middles[1] + context[2]], [middles[0]]))

            return made

    return []

def splice(context, middles):
    return ''.join(part + piece for part, piece in zip(context, middles + ['']))

def verified(families):
    programs, checks = [], []

    for family, (row, made) in families.items():
        for cell, variant, context, middles in made:
            emptied = [splice(context, middles[:at] + [''] + middles[at + 1:]) for at in range(len(middles))]
            checks.append((family, len(programs), len(emptied)))
            programs += [(program, row['tests'], row['check']) for program in [splice(context, middles)] + emptied]

    results = run_many(programs)
    failed = {family for family, at, emptied in checks if not results[at] or any(results[at + 1:at + 1 + emptied])}

    return {family: found for family, found in families.items() if family not in failed}, len(failed)

def build(out, tokenizer):
    out = Path(out)
    families = {}

    for row in read_jsonl(out / 'test.jsonl'):
        made = family_rows(row) if row['check'] == 'asserts' else []

        if made:
            families[row['id']] = (row, made)

    kept, unverified = verified(families)
    written = []

    for family, (row, made) in kept.items():
        task = task_of(row['messages'][0]['content'])

        for cell, variant, context, middles in made:
            hidden = round(sum(map(len, middles)) / sum(map(len, context + middles)), 3)
            written.append({
                'id': f'{family}/{cell}/{variant}', 'benchmark': 'panel', 'family': family, 'group': row['group'], 'source': row['source'], 'cell': cell, 'variant': variant,
                'strategy': 'statement', 'gaps': len(middles), 'linked': cell == 'dependent' and variant == 'both', 'hidden': hidden, 'check': row['check'],
                'messages': chat(context, middles, task), 'tests': row['tests'],
            })

    count_tokens(written, tokenizer)
    long ={made['family'] for made in written if not fits(made)}
    written = [made for made in written if made['family'] not in long]
    merge_jsonl(out / 'eval.jsonl', written, 'benchmark', 'panel')
    counts = collections.Counter(candidates=len(families), unverified=unverified, too_long=len(long), families=len(kept) - len(long), rows=len(written))

    return dict(counts)