'''
Retrieval floors built from the training rows: lookup (each carved word's commonest edition), template (lookup plus
gaps filled with fitting frequent words) and retrieval (template plus words from the most similar training stone).
'''

import collections
import difflib
import re

from inscriptions.meta.benchmark.markup import align_words, letters_only, parse_inscription
from inscriptions.meta.config.prompts import prompt_text, target_text

names = ('lookup', 'template', 'retrieval')
gap_word = re.compile(r'\[(-+|\d+)\]')
expansion = re.compile(r'\([^()]*\)|\[[^\[\]]*[A-Za-z][^\[\]]*\]')

def context_keys(words, i):
    before, after = words[i - 1] if i else '', words[i + 1] if i + 1 < len(words) else ''

    return (before, words[i], after), (before, words[i], None), (None, words[i], after), words[i]

class Tables:
    def __init__(self, train):
        self.table, self.slots, self.vocabulary = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter()
        self.examples, self.postings = [], collections.defaultdict(list)

        for row in train:
            pairs = align_words(parse_inscription(target_text(row)))
            words = [carved for carved, _ in pairs]
            self.examples.append((words, [edition for _, edition in pairs]))

            for i, (carved, edition) in enumerate(pairs):
                for key in context_keys(words, i):
                    self.table[key][edition] += 1

                if '[' not in carved:
                    letters = letters_only(carved)
                    self.vocabulary[letters] += 1

                    for key in ((words[i - 1] if i else '', words[i + 1] if i + 1 < len(words) else ''), (words[i - 1] if i else '', None), (None, words[i + 1] if i + 1 < len(words) else '')):
                        self.slots[key][letters] += 1

            for word in set(words):
                self.postings[word].append(len(self.examples) - 1)

        self.best = {key: editions.most_common(1)[0][0] for key, editions in self.table.items()}
        self.slots = {key: [letters for letters, _ in counter.most_common()] for key, counter in self.slots.items()}
        self.common = [letters for letters, _ in self.vocabulary.most_common()]
        self.shaped = collections.defaultdict(list)

        for letters in self.common:
            for key in ((letters[:1], None), ('', len(letters)), (letters[:1], len(letters))):
                self.shaped[key].append(letters)

def pattern(word):
    found = gap_word.search(word)
    prefix, suffix = letters_only(word[:found.start()]), letters_only(gap_word.sub('', word[found.end():]))
    length = int(found.group(1)) if found.group(1).isdigit() else None

    return prefix, suffix, length

def fits(letters, prefix, suffix, length):
    return letters.startswith(prefix) and letters.endswith(suffix) and len(letters) > len(prefix) + len(suffix) and (length is None or len(letters) == len(prefix) + len(suffix) + length)

def filled(letters, prefix, suffix):
    return f'{prefix}[{letters[len(prefix):len(letters) - len(suffix)]}]{suffix}'

def template_word(tables, words, word, i):
    prefix, suffix, length = pattern(word)
    before, after = words[i - 1] if i else '', words[i + 1] if i + 1 < len(words) else ''

    for key in ((before, after), (before, None), (None, after)):
        for letters in tables.slots.get(key, ()):
            if fits(letters, prefix, suffix, length):
                return filled(letters, prefix, suffix)

    size = len(prefix) + len(suffix) + length if length else None

    for letters in tables.common if not prefix and not length else tables.shaped.get((prefix[:1], size), ()):
        if fits(letters, prefix, suffix, length):
            return filled(letters, prefix, suffix)

    return words[i].lower()

def counterpart(words, carved, i):
    aligned = {a + step: b + step for a, b, size in difflib.SequenceMatcher(None, words, carved, autojunk=False).get_matching_blocks() for step in range(size)}

    for at in (aligned.get(i - 1, -2) + 1, aligned.get(i + 1, len(carved) + 1) - 1):
        if 0 <= at < len(carved):
            return carved[at]

def retrieved_word(tables, words, word, i, neighbour):
    if neighbour is None:
        return None

    prefix, suffix, length = pattern(word)
    found = counterpart(words, neighbour[0], i)

    if found and '[' not in found and fits(letters_only(found), prefix, suffix, length):
        return filled(letters_only(found), prefix, suffix)

def nearest(tables, words):
    rare = sorted(set(words), key=lambda word: len(tables.postings.get(word, ())))[:3]
    scores = collections.Counter()

    for word in rare:
        for at in tables.postings.get(word, ())[:300]:
            if tables.examples[at][0] != words:
                scores[at] = len(set(words) & set(tables.examples[at][0]))

    return tables.examples[scores.most_common(1)[0][0]] if scores else None

def edition(tables, row, restore=False, retrieve=False):
    raw = prompt_text(row).split(' ')
    plain = raw
    neighbour = nearest(tables, plain) if retrieve else None
    found = []

    for i, word in enumerate(raw):
        if restore and len(gap_word.findall(word)) == 1 and not expansion.search(word) and word.count('[') == word.count(']'):
            found.append(retrieved_word(tables, plain, word, i, neighbour) or template_word(tables, plain, word, i))

        else:
            found.append(next((tables.best[key] for key in context_keys(plain, i) if key in tables.best), plain[i].lower()))

    return ' '.join(found)

def baseline(name, train, rows):
    tables = Tables(train)

    return [edition(tables, row, restore=name != 'lookup', retrieve=name == 'retrieval') for row in rows]