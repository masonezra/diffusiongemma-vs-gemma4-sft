'''
Scores a reply against the edition and the carved input. Headline: recovered, the share of hidden spans matched
letter for letter (ground truth: the stone). Also expanded and restored (agreement with one editor), faithful (carved
letters copied unchanged), exact, error and unread.
Reference: https://www.nature.com/articles/s41586-025-09292-5
Reference: https://github.com/google-deepmind/predictingthepast
'''

import collections
import difflib
import re
import string
from statistics import fmean

from inscriptions.meta.benchmark.markup import alphabet, carved_text, edition_text, editor_additions, parse_inscription
from inscriptions.meta.config.prompts import prompt_text, target_text

max_span = 20
min_alignment = 0.9
abbreviation_bins = ((2, '0-2 abbreviations'), (5, '3-5 abbreviations'), (float('inf'), '6+ abbreviations'))
inserted_bins = ((10, '0-10 inserted'), (30, '11-30 inserted'), (60, '31-60 inserted'), (float('inf'), '61+ inserted'))
token_bins = ((32, '1-32 tokens'), (64, '33-64 tokens'), (128, '65-128 tokens'), (float('inf'), '129+ tokens'))
position_bins = ((1 / 3, 'gap at start'), (2 / 3, 'gap at centre'), (float('inf'), 'gap at end'))
span_bins = ((1, '1 letter'), (3, '2-3 letters'), (10, '4-10 letters'), (float('inf'), '11+ letters'))
special_token = re.compile(r'<[^<>=]*>')
notation = alphabet | set(string.digits + '/?!<>={}«»')
formula = re.compile(r'^(D M S?|H S E|S T T L|V S L M|B M F|H M H N S|I O M)\b')
ratios = (
    ('exact', 'rows'), ('faithful', 'rows'), ('unread', 'rows'), ('edits', 'length'), ('changed', 'carved'), ('dropped', 'carved'),
    ('replaced', 'carved'), ('added', 'carved'), ('expanded', 'abbreviations'), ('restored', 'gaps'), ('recovered', 'hidden'),
    ('length right', 'hidden'), ('cleaned', 'rows'), *((f'recovered, {name}', f'hidden, {name}') for _, name in span_bins),
)
binary = ('exact', 'faithful', 'unread', 'cleaned')

def edit_distance(a, b):
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    a, b = a[prefix:], b[prefix:]
    suffix = next((i for i, (x, y) in enumerate(zip(reversed(a), reversed(b))) if x != y), min(len(a), len(b)))
    a, b = a[:len(a) - suffix], b[:len(b) - suffix]
    row = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        diagonal, row[0] = row[0], i

        for j, y in enumerate(b, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))

    return row[-1]

def strip_wrapping(reply):
    lines = [line.split(':')[-1].strip(' "\'`.') for line in reply.replace('```', '\n').splitlines()]

    return ' '.join(' '.join(line for line in lines if line).split())

def read_reply(reply):
    text = strip_wrapping(special_token.sub(' ', reply))
    kept = ''.join(char for char in text if char in notation)

    return ' '.join(kept.split()), kept != text or bool(special_token.search(reply))

def carved_letters(marks):
    return [mark.char.upper() for mark in marks if mark.char != ' ' and not mark.restored and not mark.expanded]

def remove_spaces(text):
    return ''.join(text.split())

def binned(value, bins):
    return next(name for limit, name in bins if value <= limit)

def align_additions(reply_marks, prompt_marks):
    # Additions are keyed by carved-letter position, so the reply's carved letters are aligned to the prompt's and each
    # addition moves to its prompt position; below min_alignment the reply is too far off to align
    matcher = difflib.SequenceMatcher(None, carved_letters(reply_marks), carved_letters(prompt_marks), autojunk=False)
    position = {}

    if matcher.ratio() < min_alignment:
        return {}, {}

    for reply_start, prompt_start, size in matcher.get_matching_blocks():
        for step in range(size + 1):
            position.setdefault(reply_start + step, prompt_start + step)

    return ({position[at]: text for at, text in additions.items() if at in position} for additions in editor_additions(reply_marks))

def carved_changes(reply_marks, prompt_marks):
    before, after = carved_letters(prompt_marks), carved_letters(reply_marks) if reply_marks else []
    found = collections.Counter(carved=len(before))

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == 'delete':
            found['dropped'] += i2 - i1

        elif tag == 'insert':
            found['added'] += j2 - j1

        elif tag == 'replace':
            found['replaced'] += max(i2 - i1, j2 - j1)

    found['changed'] = found['dropped'] + found['replaced'] + found['added']

    return found

def score_row(row, reply):
    target, prompt = target_text(row), prompt_text(row)
    reply, cleaned = read_reply(reply)
    marks = parse_inscription(reply)
    parsed = marks is not None
    reply = edition_text(marks) if parsed else reply
    prompt_marks, target_marks = parse_inscription(prompt), parse_inscription(target)
    faithful = parsed and remove_spaces(carved_text(marks)) == remove_spaces(carved_text(prompt_marks))
    scores = collections.Counter({'rows': 1, 'exact': reply == target, 'faithful': faithful, 'unread': not parsed, 'cleaned': cleaned, 'edits': edit_distance(reply, target), 'length': len(target)})
    scores.update(carved_changes(marks, prompt_marks))
    expansions, restorations = editor_additions(target_marks)
    predicted_expansions, predicted_restorations = align_additions(marks, prompt_marks) if marks else ({}, {})
    span_errors = []

    for at, expansion in expansions.items():
        scores['abbreviations'] += 1
        scores['expanded'] += predicted_expansions.get(at) == expansion

    for at, letters in restorations.items():
        if not letters:
            continue

        predicted = predicted_restorations.get(at, '')

        if at in row['hidden']:
            size = binned(len(letters), span_bins)
            scores['hidden'] += 1
            scores['recovered'] += predicted == letters
            scores[f'hidden, {size}'] += 1
            scores[f'recovered, {size}'] += predicted == letters
            scores['length right'] += len(predicted) == len(letters)
            scores['length diff'] += len(predicted) - len(letters)
            span_errors.append((len(letters), min(1, edit_distance(predicted, letters) / len(letters))))

        else:
            scores['gaps'] += 1
            scores['restored'] += predicted == letters

    return scores, span_errors

def row_groups(row):
    prompt = prompt_text(row)
    inserted = sum(map(len, editor_additions(parse_inscription(target_text(row)))[0].values()))
    letters = sum(char.isalpha() for char in prompt)
    groups = [
        'all', binned(row['abbreviations'], abbreviation_bins), binned(inserted, inserted_bins), binned(row['tokens'], token_bins),
        'funerary' if row['funerary'] else 'not funerary',
        'formula opening' if formula.match(prompt) else 'other opening',
    ]

    if row['hidden']:
        groups.append(binned(row['hidden'][0] / max(1, letters), position_bins))

    return groups

def percent(count, part, whole):
    return round(100 * count[part] / count[whole], 1) if count[whole] else None

def score_rows(rows, replies):
    totals = collections.defaultdict(collections.Counter)
    errors_by_length = collections.defaultdict(lambda: collections.defaultdict(list))

    for row, reply in zip(rows, replies):
        scores, span_errors = score_row(row, reply)

        for group in row_groups(row):
            totals[group].update(scores)

            for length, error in span_errors:
                errors_by_length[group][length].append(error)

    result = {group: {'rows': count['rows'], **{'error' if part == 'edits' else part: percent(count, part, whole) for part, whole in ratios}} for group, count in totals.items()}

    for group, count in totals.items():
        if count['hidden']:
            result[group]['length error'] = round(count['length diff'] / count['hidden'], 2)

    for group, by_length in errors_by_length.items():
        result[group]['run error'] = round(100 * fmean([fmean(errors) for length, errors in by_length.items() if length <= max_span]), 1)

    return result