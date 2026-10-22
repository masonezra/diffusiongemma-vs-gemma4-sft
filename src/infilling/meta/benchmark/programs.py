'''
Program normalisation and identity, on the docstring-free syntax tree with every name the program binds renamed by
order of first occurrence: renamed(tree) is the exact-copy key, shingles(tree) the 8-token windows of its source
{CRC-32, so every run hashes alike}. catalogue / copied keep held-out programs and their near copies out of training
{sorted shingle and owner arrays, a few MB, so every worker holds a copy}; NearCopies keeps one program of each
near-copy cluster: MinHash over the shingles, 20 bands of 5 rows {a pair at
Jaccard similarity s shares a band with probability 1 - (1 - s^5)^20, 99.96% at s = 0.8}; a pair sharing a band is a
near copy only when its signatures agree on at least `similarity` = 0.8 of the 100 values, the estimated Jaccard.
Reference: https://arxiv.org/pdf/2107.06499
Reference: https://arxiv.org/pdf/2305.06161
Reference: https://github.com/ekzhu/datasketch/blob/master/datasketch/minhash.py
'''

import ast
import copy
import hashlib
import re
import zlib
from typing import NamedTuple

import numpy as np

from infilling.meta.benchmark.sandbox import parse
from infilling.meta.config.prompts import gap, solution

fence = re.compile(r'```python\n(.*?)```', re.DOTALL)
word = re.compile(r'\w+|[^\w\s]')
definitions = (ast.FunctionDef, ast.ClassDef)
pytest_use = re.compile(r'pytest\s*\.\s*(\w+)|from\s+pytest\s+import\s+(\w+)')
pytest_plain = {'raises', 'approx'}
kept_nodes = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
named_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

class Known(NamedTuple):
    marks: np.ndarray
    owners: np.ndarray
    spans: np.ndarray
    twins: set

def layout(code):
    return '\n'.join(line.rstrip() for line in code.replace('\r\n', '\n').replace('\r', '\n').expandtabs(4).split('\n')).strip('\n')

def tidy(code, min_lines=3):
    code = layout(code)

    if parse(code) is None:
        return None, 'no_parse'

    if gap in code or '```' in code:
        return None, 'has_gap_marker'

    if code.count('\n') + 1 < min_lines:
        return None, 'size'

    return code, None

def clean(output, min_lines=3):
    found = fence.findall(output)

    if len(found) != 1:
        return None, 'no_single_fence'

    code, reason = tidy(found[0], 1)

    if reason:
        return None, reason

    tree = parse(code)

    if not any(isinstance(node, definitions) for node in tree.body):
        return None, 'no_function'

    used = {name.id for node in tree.body if isinstance(node, definitions) for name in ast.walk(node) if isinstance(name, ast.Name)}
    lines, kept = code.split('\n'), []

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]

            if not any(isinstance(name, ast.Name) and name.id in used for target in targets for name in ast.walk(target)):
                continue

        if isinstance(node, kept_nodes):
            start = min([node.lineno] + [decorator.lineno for decorator in getattr(node, 'decorator_list', [])])
            kept.append('\n'.join(lines[start - 1:node.end_lineno]))

    return tidy('\n\n'.join(kept), min_lines)

def as_script(test):
    tree = parse(test)

    if tree is None or {name for pair in pytest_use.findall(test) for name in pair if name} - pytest_plain:
        return None

    calls = [f'{node.name}()' for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith('test') and not node.args.args]

    if any(isinstance(node, ast.ClassDef) and any('TestCase' in ast.unparse(base) for base in node.bases) for node in tree.body):
        calls += ['import unittest', "unittest.main(argv=['tests'])"]
    kept = [line for line in test.split('\n') if not re.match(r'\s*(from solution import|import solution)', line)]

    return '\n'.join(kept + [''] + calls) if calls else None

def names(tree):
    return {node.name for node in tree.body if isinstance(node, definitions)}

def bare(tree):
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)

        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            node.body = body[1:] or [ast.Pass()]

    return tree

def normalised(tree):
    tree = bare(copy.deepcopy(tree))
    nodes = list(ast.walk(tree))
    bound = {node.name for node in nodes if isinstance(node, named_nodes)} | {node.arg for node in nodes if isinstance(node, ast.arg)}
    bound |= {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    order = {}

    for node in nodes:
        field = 'name' if isinstance(node, named_nodes) else 'arg' if isinstance(node, ast.arg) else 'id' if isinstance(node, ast.Name) else None

        if field and getattr(node, field) in bound:
            setattr(node, field, f'n{order.setdefault(getattr(node, field), len(order))}')

    return tree

def renamed(tree):
    return hashlib.sha1(ast.dump(normalised(tree)).encode()).hexdigest()

def shingles(tree, size=8):
    words = word.findall(ast.unparse(normalised(tree)))

    return {zlib.crc32(' '.join(words[i:i + size]).encode()) for i in range(len(words) - size + 1)}

class NearCopies:
    prime, width = np.uint64((1 << 61) - 1), np.uint64((1 << 32) - 1)

    def __init__(self, bands=20, rows=5, similarity=0.8, seed=0):
        rng = np.random.default_rng(seed)
        self.a, self.b = rng.integers(1, (1 << 61) - 1, (2, bands * rows), dtype=np.uint64)
        self.rows, self.similarity, self.exact, self.buckets, self.signatures = rows, similarity, set(), {}, []

    def keys(self, tree):
        marks = np.fromiter(shingles(tree), np.uint64)
        # MinHash as in datasketch: h(x) = ((a x + b) mod (2^61 - 1)) & (2^32 - 1), minimised over the shingles
        signature = ((np.outer(marks, self.a) + self.b) % self.prime & self.width).min(0) if len(marks) else None
        bands = [] if signature is None else [int.from_bytes(hashlib.blake2b(bytes([band]) + signature[band * self.rows:(band + 1) * self.rows].tobytes(), digest_size=8).digest(), 'big') for band in range(len(signature) // self.rows)]

        return renamed(tree), signature, bands

    def copy_of(self, keys):
        exact, signature, bands = keys

        if exact in self.exact:
            return 'duplicate'

        near = {at for band in bands for at in self.buckets.get(band, ())}

        return 'near_duplicate' if any((self.signatures[at] == signature).mean() >= self.similarity for at in near) else None

    def add(self, keys):
        exact, signature, bands = keys
        self.exact.add(exact)

        if signature is not None:
            for band in bands:
                self.buckets.setdefault(band, []).append(len(self.signatures))

            self.signatures.append(signature)

def catalogue(rows):
    seen, twins, marks, owners, spans = {}, set(), [], [], []

    for row in rows:
        uid, tree = row['id'].rsplit('/L', 1)[0], parse(solution(row))

        if tree is not None and uid not in seen:
            seen[uid] = len(spans)
            twins.add(renamed(tree))
            found = shingles(tree)
            spans.append(len(found))
            marks += found
            owners += [seen[uid]] * len(found)

    marks, owners = np.array(marks, dtype=np.int64), np.array(owners, dtype=np.int64)
    order = np.argsort(marks, kind='stable')

    return Known(marks[order], owners[order], np.array(spans, dtype=np.int64), twins)

def copied(tree, known, share):
    if renamed(tree) in known.twins:
        return True

    marks = np.fromiter(shingles(tree), np.int64)
    low, high = np.searchsorted(known.marks, marks), np.searchsorted(known.marks, marks, 'right')
    hits = np.bincount(np.concatenate([marks[:0]] + [known.owners[start:end] for start, end in zip(low, high)]), minlength=len(known.spans))

    return bool(((hits > 0) & (hits >= share * known.spans)).any())