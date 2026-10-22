'''
Parses EDCS Leiden markup into letters marked carved, expanded or restored, and renders the model's input (the
carved text) and target (the edition). Carved letters are never changed, so any reply can be checked against its input.
Reference: https://zenodo.org/records/7072337
'''

import re
import string
from typing import NamedTuple

gap = '[---]'
lost = '-'
alphabet = set(string.ascii_letters + ' []()|' + lost)
erasure_marks = ('[[', ']]', '«', '»', '{', '}')

editor_remark = re.compile(r'\([?!]\)')
correction = re.compile(r'<([^<>=]*)=([^<>=]*)>')
lost_count = re.compile(r'[0-9-]+(?: +[0-9-]+)*')
empty_bracket = re.compile(r'\[ *\]')
line_break = re.compile(r' */+ +| +/+ *')
field_break = re.compile(r'//+')
bracket_padding = re.compile(r'[(\[] +| +[)\]]')

class Mark(NamedTuple):
    char: str
    restored: bool
    expanded: bool

space = Mark(' ', False, False)

def keep_carved(match):
    reading, carved = match.group(1).strip(), match.group(2).strip()

    if not reading[:1].isupper():
        return carved.lower()

    if reading[1:2].isupper():
        return carved.upper()

    return carved.capitalize()

def normalise(text):
    for mark in erasure_marks:
        text = text.replace(mark, '')

    text = correction.sub(keep_carved, editor_remark.sub('', text))
    text = line_break.sub(' ', text).replace('/', '').replace('?', '').replace('!', '')
    text = empty_bracket.sub(f'[{lost}]', lost_count.sub(lost, text))

    return bracket_padding.sub(lambda match: match.group().strip(), ' '.join(text.split()))

def letters_only(text):
    return ''.join(char for char in text.lower() if char.isalpha())

def parse_field(text):
    marks, restored, expanded, bracket_seen = [], False, False, False

    if set(text) - alphabet:
        return None

    for char in text:
        if char in '[]':
            if restored == (char == '[') and (restored or bracket_seen):
                return None

            # A ']' before any '[' means the field opened inside a restoration: everything so far was restored after a lost run
            if char == ']' and not restored:
                marks = [Mark(lost, True, False)] + [Mark(mark.char, True, mark.expanded) for mark in marks]

            restored, bracket_seen = char == '[', True

        elif char in '()':
            if expanded == (char == '('):
                return None

            expanded = char == '('

        elif char == lost and (expanded or not restored):
            return None

        elif char in (lost, ' ') and marks and marks[-1].char == char:
            continue

        else:
            marks.append(Mark(char, restored, expanded))

    if expanded:
        return None

    if restored and (not marks or marks[-1].char != lost):
        marks.append(Mark(lost, True, False))

    while marks and marks[0].char == ' ':
        marks.pop(0)

    while marks and marks[-1].char == ' ':
        marks.pop()

    return marks

def parse_inscription(text):
    fields = [parse_field(normalise(field)) for field in field_break.split(text)]

    if None in fields:
        return None

    return [mark for i, marks in enumerate(filter(None, fields)) for mark in [space] * bool(i) + marks]

def carved_text(marks):
    text, lost_run = [], False

    for mark in marks:
        if mark.restored:
            lost_run = True

        else:
            text.append(gap * lost_run + ('' if mark.expanded else mark.char.upper()))
            lost_run = False

    return ' '.join(''.join(text + [gap * lost_run]).split())

def edition_text(marks):
    text, restored, expanded = [], False, False

    for mark in marks:
        if (mark.restored, mark.expanded) != (restored, expanded):
            text.append(')' * expanded + ']' * (restored and not mark.restored) + '[' * (mark.restored and not restored) + '(' * mark.expanded)

        text.append(lost * 3 if mark.char == lost else mark.char)
        restored, expanded = mark.restored, mark.expanded

    return ''.join(text) + ')' * expanded + ']' * restored

def editor_additions(marks):
    expansions, restorations, position = {}, {}, 0

    for mark in marks:
        if mark.restored:
            letter = mark.char.lower() if mark.char.isalpha() and not mark.expanded else ''
            restorations[position] = restorations.get(position, '') + letter

        elif mark.expanded:
            expansions[position] = expansions.get(position, '') + mark.char.lower()

        else:
            position += mark.char != ' '

    return expansions, restorations

def align_words(marks):
    pairs, word = [], []

    for mark in marks + [space]:
        if mark == space:
            pairs.append((carved_text(word), edition_text(word)))
            word = []

        else:
            word.append(mark)

    return pairs