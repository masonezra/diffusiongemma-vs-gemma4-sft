'''
Samples 1 to 3 disjoint gaps per program inside function bodies {or a script's top-level code}, each of one kind. I mix
three kinds, as in fill-in-the-middle training, because each benchmark family rewards a different one:
    line_span       whole lines: half of the draws the whole body of a compound statement {SAFIM block}, the rest two
                    uniform line cuts {HumanEval-Infilling single and multi line}; line-level training is best on
                    line benchmarks
    node_span       one multi-token expression: a condition, loop range, assigned or returned value, or call {SAFIM
                    control}; syntax-aware masks beat random character masks by up to 5 points {AST-FIM}
    random_span     two uniform character cuts, which may split a token {HumanEval-Infilling random span}; line-only
                    training fails almost completely on random spans {0.015 against 0.321 pass, Bavarian et al.}
Every gap hides at least min_chars non-space characters, so none is trivial. Marks gaps as linked when a later gap
reads a name an earlier gap binds and that name appears nowhere outside the gaps, so the gaps must agree with each
other rather than with the context.
Reference: https://arxiv.org/pdf/2207.14255
Reference: https://arxiv.org/pdf/2506.00204
Reference: https://arxiv.org/pdf/2403.04814
Reference: https://arxiv.org/pdf/2602.01326
'''

import ast
import re
from itertools import accumulate
from math import comb

weights = {'line_span': 40, 'node_span': 30, 'random_span': 30}
min_chars = 20
block_share = 0.5
gap_counts = {1: 40, 2: 35, 3: 25}
max_share = 0.5
linked_share = 0.5
targets = {
    ast.If: 'test', ast.While: 'test', ast.IfExp: 'test', ast.For: 'iter', ast.comprehension: 'iter',
    ast.Assign: 'value', ast.AugAssign: 'value', ast.AnnAssign: 'value', ast.Return: 'value',
}
block_fields = ('body', 'orelse', 'finalbody')

def functions(tree):
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

def bodies(tree, script):
    found = []

    for node in functions(tree):
        body = node.body[1:] if ast.get_docstring(node, clean=False) is not None else node.body

        if body and body[0].lineno > node.lineno:
            found.append((body[0].lineno - 1, node.end_lineno))

    top = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]

    if script and top:
        found.append((top[0].lineno - 1, top[-1].end_lineno))

    return found

def expressions(tree, script):
    for scope in [tree] if script else functions(tree):
        for node in ast.walk(scope):
            picked = node if isinstance(node, ast.Call) else getattr(node, targets.get(type(node), ''), None)

            if picked is not None and not isinstance(picked, (ast.Name, ast.Constant)):
                yield picked

def blocks(tree, script):
    for scope in [tree] if script else functions(tree):
        for node in ast.walk(scope):
            if isinstance(node, (ast.Module, ast.ClassDef)):
                continue

            for field in block_fields:
                block = getattr(node, field, None)

                if not isinstance(block, list) or not block or not isinstance(block[0], ast.stmt):
                    continue

                if field == 'body' and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False) is not None:
                    block = block[1:]

                if block and block[0].lineno > (node.pattern.lineno if isinstance(node, ast.match_case) else node.lineno):
                    yield block

def offsets(code):
    lines = code.split('\n')
    starts = [min(start, len(code)) for start in accumulate((len(line) + 1 for line in lines), initial=0)]

    # ast column offsets count UTF-8 bytes, so they are converted to character offsets
    return starts, lambda line, column: starts[line - 1] + len(lines[line - 1].encode()[:column].decode())

def by_length(spans, rng):
    spans = sorted(span for span in spans if span[1] > span[0])

    return (lambda: rng.choices(spans, weights=[end - start for start, end in spans])[0]) if spans else None

def sampler(code, tree, rng, strategy, script):
    starts, place = offsets(code)

    if strategy == 'node_span':
        return by_length({(place(node.lineno, node.col_offset), place(node.end_lineno, node.end_col_offset)) for node in expressions(tree, script)}, rng)

    whole = by_length({(starts[block[0].lineno - 1], starts[block[-1].end_lineno]) for block in blocks(tree, script)}, rng) if strategy == 'line_span' else None
    grids = [starts[first:last + 1] if strategy == 'line_span' else range(starts[first], starts[last] + 1) for first, last in bodies(tree, script)]
    grids = [grid for grid in grids if len(grid) > 1]

    if not grids:
        return None

    # Weighting each body by its number of cut pairs makes the pair uniform over the program, not per function
    ways = [comb(len(grid), 2) for grid in grids]

    def draw():
        if whole and rng.random() < block_share:
            return whole()

        return tuple(sorted(rng.sample(rng.choices(grids, weights=ways)[0], 2)))

    return draw

def uses(code, tree):
    _, place = offsets(code)

    return [(place(node.lineno, node.col_offset), node.id, isinstance(node.ctx, ast.Store)) for node in ast.walk(tree) if isinstance(node, ast.Name)]

def reads(first, second, found):
    bound = {name for at, name, stored in found if stored and first[1] <= at < first[2]}

    return {name for at, name, stored in found if not stored and name in bound and second[1] <= at < second[2]}

def outside(code, gaps):
    edges = [0] + [edge for _, start, end in gaps for edge in (start, end)] + [len(code)]

    return '\n'.join(code[start:end] for start, end in zip(edges[::2], edges[1::2]))

def linked(code, tree, gaps, found=None):
    found = uses(code, tree) if found is None else found
    gaps = sorted(gaps, key=lambda gap: gap[1])
    shown = outside(code, gaps)

    return any(not re.search(rf'\b{re.escape(name)}\b', shown) for at, first in enumerate(gaps) for second in gaps[at + 1:] for name in reads(first, second, found))

def dependence(code, tree, gaps):
    found, gaps = uses(code, tree), sorted(gaps, key=lambda gap: gap[1])
    inside = lambda at: any(start <= at < end for _, start, end in gaps)
    visible = {name for at, name, stored in found if not inside(at)} | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    bound = [{name for at, name, stored in found if stored and start <= at < end} for _, start, end in gaps]
    read = [{name for at, name, stored in found if not stored and start <= at < end} for _, start, end in gaps]
    forward = set().union(*(bound[first] & read[second] for first in range(len(gaps)) for second in range(first + 1, len(gaps))))
    backward = set().union(*(bound[second] & read[first] for first in range(len(gaps)) for second in range(first + 1, len(gaps))))

    return {'forward': bool(forward), 'backward': bool(backward), 'hidden': bool(forward - visible)}

def hard(text):
    return len(''.join(text.split())) >= min_chars

def mask(code, tree, rng, script=False, tries=20):
    count = rng.choices(list(gap_counts), weights=list(gap_counts.values()))[0]
    samplers = {strategy: sampler(code, tree, rng, strategy, script) for strategy in weights}
    tie = count > 1 and rng.random() < linked_share
    found = uses(code, tree) if tie else []
    gaps, room = [], max_share * len(code)

    for _ in range(count):
        strategy = rng.choices(list(weights), weights=list(weights.values()))[0]
        spare = None

        # A linked gap is rarer, so it gets three times the tries; if none links, the first valid gap is kept
        for _ in range((tries * (3 if tie and gaps else 1)) if samplers[strategy] else 0):
            start, end = samplers[strategy]()
            apart = all(end < first or last < start for _, first, last in gaps)

            # Half of the draws that run to the end of the code are rejected, so most gaps keep a suffix to condition on
            if not (apart and hard(code[start:end]) and end - start <= room and (code[end:].strip() or rng.random() < 0.5)):
                continue

            gap = (strategy, start, end)

            if not (tie and gaps) or linked(code, tree, gaps + [gap], found):
                spare = gap
                break

            spare = spare or gap

        if spare:
            gaps.append(spare)
            room -= spare[2] - spare[1]

    return sorted(gaps, key=lambda gap: gap[1])