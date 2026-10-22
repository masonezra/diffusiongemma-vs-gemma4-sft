'''
Decipherment: c = E_K(p) in, plaintext p out, with neither the cipher nor the key K given. I chose it because one hidden
key constrains every answer position, so a correct answer has to be globally consistent, which is where a joint denoiser
should win. Seven classical-cipher levels on arXiv abstracts, held out by identifier month >= 2607; length from 2.5 x an
estimated unicity distance up to 254 symbols, one DiffusionGemma canvas (substitution levels graded at 16 to 254 symbols).
Reference: https://arxiv.org/pdf/2509.07282 {ALICE: error against ciphertext length}
Reference: https://arxiv.org/pdf/2012.15229 {Aldarrab and May: short ciphers, lengths 16 to 256}
'''

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from string import ascii_lowercase as letters, ascii_uppercase

from datasets import load_dataset

from common.models import canvas
from cryptanalysis.meta.config.prompts import chat, cipher_label, instruction, instruction_keyed, label
from cryptanalysis.meta.config.settings import lengths, lengths_keyed, margin, overlap, redundancy, release, revision, source, spare

room = canvas - 2
space = '_'
symbols = letters + ascii_uppercase + 'αβγδεζηθικλμνξοπρστυφχψω' + 'ΓΔΘΛΞΠΣΦΨΩ'
frequency = (8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.77, 4.0, 2.4, 6.7, 7.5, 1.9, 0.095, 6.0, 6.3, 9.1, 2.8, 0.98, 2.4, 0.15, 2.0, 0.074)
prose = re.compile(r'''[a-z\s.,;:!?'"()-]*''')
english = set('the of and in to is we for that this with on are by as an be it which from at or can our these has have not'.split())
foreign = set('le la les des une est dans nous et du sur pour el los las del para que por una der die und ein von mit ist os em um uma na da do di il che per'.split())


class Cipher:
    arbitrary = False

    def __init__(self, rng, **options):
        self.rng = rng
        self.key = self.draw(**options)

    def least(self):
        # L = ceil(margin * H(K) / D): Shannon's unicity distance H(K) / D at D bits of redundancy per letter, scaled by margin
        return math.ceil(margin * self.bits() / redundancy)

    def describe(self, plain, secret, names):
        pairs = sorted({names[symbol]: char for symbol, char in zip(secret, plain) if char in letters}.items())

        return 'Key: ' + ' '.join(f'{symbol}->{char}' for symbol, char in pairs)


class Substitution(Cipher):
    arbitrary = True

    def draw(self):
        return dict(zip(letters, self.rng.sample(letters, 26)))

    def bits(self):
        return math.log2(math.factorial(26))

    def encrypt(self, text):
        return ''.join(self.key.get(char, char) for char in text)

class Homophonic(Cipher):
    arbitrary = True

    def draw(self, size):
        extra = self.rng.choices(letters, weights=frequency, k=size - 26)
        pool = self.rng.sample(symbols, size)
        key = {}

        for letter in letters:
            count = 1 + extra.count(letter)
            key[letter], pool = pool[:count], pool[count:]

        return key

    def bits(self):
        return math.log2(26) * sum(map(len, self.key.values()))

    def encrypt(self, text):
        return ''.join(self.rng.choice(self.key[char]) if char in self.key else char for char in text)

class Vigenere(Cipher):
    def draw(self, period):
        return [self.rng.randrange(26) for _ in range(period)]

    def bits(self):
        return math.log2(26) * len(self.key)

    def encrypt(self, text):
        moved = lambda char, i: letters[(letters.index(char) + self.key[i % len(self.key)]) % 26]

        return ''.join(moved(char, i) if char in letters else char for i, char in enumerate(text))

    def describe(self, plain, secret, names):
        pairs = ' | '.join(f'{c} {letters[self.key[i % len(self.key)]]}' for i, c in enumerate(secret))

        return f'Key: each ciphertext symbol with its key letter after it; letters count a=0 to z=25 and each plaintext letter is the symbol minus its key letter, mod 26: {pairs}'


class Autokey(Vigenere):
    def encrypt(self, text):
        stream = self.key + [letters.index(char) for char in text]

        return ''.join(letters[(letters.index(char) + shift) % 26] for char, shift in zip(text, stream))

    def describe(self, plain, secret, names):
        return ('Key: the primer below, after which the key continues with the plaintext itself; letters count a=0 to z=25 and each plaintext '
                'letter is the symbol minus its key letter, mod 26: ' + ' '.join(letters[shift] for shift in self.key))


class Columnar(Cipher):
    def draw(self, width):
        return self.rng.sample(range(width), width)

    def bits(self):
        return math.log2(math.factorial(len(self.key)))

    def order(self, length):
        return [i for column in self.key for i in range(column, length, len(self.key))]

    def encrypt(self, text):
        return ''.join(text[i] for i in self.order(len(text)))

    def describe(self, plain, secret, names):
        width, taken, parts = len(self.key), 0, []

        for column in self.key:
            count = len(range(column, len(secret), width))
            parts.append(f'column {column + 1}: ' + ' '.join(secret[taken:taken + count]))
            taken += count

        return (f'Width: {width}; columns were read out in the order ' + ' '.join(str(column + 1) for column in self.key)
                + '; the ciphertext split into those columns: ' + ' | '.join(parts))


levels = {
    1: lambda rng: Substitution(rng),
    2: lambda rng: Substitution(rng),
    3: lambda rng: Vigenere(rng, period=rng.randint(3, 10)),
    4: lambda rng: Autokey(rng, period=rng.randint(3, 10)),
    5: lambda rng: Columnar(rng, width=rng.randint(4, 9)),
    6: lambda rng: Homophonic(rng, size=rng.randint(27, 48)),
    7: lambda rng: Homophonic(rng, size=rng.randint(49, 69)),
}
spaced = {1}
graded = {1, 2}
functional = {1, 2, 6, 7}
shortest = Substitution(random.Random(0)).least()

def naming(text):
    # Renaming by first appearance leaves only the pattern of repeats: a symbol's name says nothing about the key
    names = {space: space}

    for char in text:
        if char not in names:
            names[char] = symbols[len(names) - 1]

    return names

def relabel(text):
    names = naming(text)

    return ''.join(names[char] for char in text)

def clean(text):
    text = ' '.join(text.split()).replace('`', "'")

    if not text.isascii():
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if not unicodedata.combining(char))
        text = re.sub('[‘’]', "'", text)
        text = re.sub('[“”]', '"', text)
        text = re.sub('[‐‑‒–—]', '-', text)

    words = []

    for sentence in re.split(r'(?<=[.?!]) (?=[A-Z])', text):
        sentence = sentence.lower().replace("'", '')

        if prose.fullmatch(sentence):
            words += re.findall('[a-z]+', re.sub(r'\b([a-z])\.(?=[a-z]\b)', r'\1', sentence))

    return words

def pack(words):
    left, starts = 0, []

    for n in range(len(words) - 1, -1, -1):
        left += len(words[n])

        if left >= room:
            starts.append(n)

    if not starts:
        return []

    # Start at a hash-chosen word among those leaving at least `room` letters, so crops are not all abstract openings
    first = starts[int(hashlib.md5(' '.join(words).encode()).hexdigest(), 16) % len(starts)]
    used = 0

    for n in range(first, len(words)):
        used += len(words[n])

        if used >= room:
            return words[first:n + 1]

def fetch(limit=0):
    dataset = load_dataset(source, revision=revision, split='train', columns=['id', 'abstract'])
    dataset = dataset.select(range(limit)) if limit else dataset
    found, seen = {}, set()

    for batch in dataset.iter(20_000):
        for name, abstract in zip(batch['id'], batch['abstract']):
            words = pack(clean(abstract))
            text = ' '.join(words)

            if words and text not in seen and sum(word in foreign for word in words) <= sum(word in english for word in words):
                seen.add(text)
                found[name] = text

    return found

def shared(held, earlier, step=4):
    # Any shared run of `overlap` letters contains a window of overlap - step + 1 letters that starts at a multiple of
    # `step` in the earlier text, so sampling earlier windows every `step` letters still catches every overlap
    size = overlap - step + 1
    owner = defaultdict(list)

    for n, text in enumerate(held):
        text = text.replace(' ', '')

        for i in range(len(text) - size + 1):
            owner[text[i:i + size]].append(n)

    found = set()

    for text in earlier:
        text = text.replace(' ', '')

        for i in range(0, len(text) - size + 1, step):
            found.update(owner.get(text[i:i + size], ()))

    return found

def texts(limit=0, train=6, val=5):
    found = fetch(limit)
    order = lambda name: hashlib.md5(name.encode()).hexdigest()
    # New-style ids start with YYMM; old-style ids, the ones with '/', all predate 2007
    late = lambda name: '/' not in name and name[:4] >= release
    held = sorted(filter(late, found), key=order)[:spare]
    seen = shared([found[name] for name in held], [text for name, text in found.items() if not late(name)])
    held = [name for n, name in enumerate(held) if n not in seen]
    early = sorted((name for name in found if not late(name) and int(order(name), 16) % train == 0), key=order)
    parts = {'train': early, 'val': [name for name in held if int(order(name), 16) % val == 0]}
    parts['test'] = [name for name in held if name not in set(parts['val'])]

    return {part: [found[name] for name in names] for part, names in parts.items()}

def crop(text, size, spaces):
    return text.replace(' ', space)[:size].rstrip(space) if spaces else text.replace(' ', '')[:size]

def window(text, size, rng):
    words = text.split()
    first = rng.randrange(len(words))

    while sum(map(len, words[first:])) < size:
        first -= 1

    return ' '.join(words[first:])

def row(text, level, rng, size=0, keys=False, keyed=False):
    cipher = levels[level](rng)
    plain = crop(window(text, size, rng) if size else text, size or rng.randint(max(shortest, cipher.least()), room), level in spaced)
    secret = cipher.encrypt(plain)
    names = naming(secret) if cipher.arbitrary else {char: char for char in secret}
    shown = ' '.join(names[char] for char in secret)
    question = (f'{instruction_keyed}\n\n{cipher_label}{type(cipher).__name__.lower()}\n{cipher.describe(plain, secret, names)}\n{label}{shown}' if keyed
                else f'{instruction}\n\n{label}{shown}')
    reply = ' '.join(plain)
    made = {'id': hashlib.md5(question.encode()).hexdigest(), 'level': level, 'cipher': type(cipher).__name__.lower(),
            'length': len(plain), 'bucket': size, 'messages': chat(question, reply)}

    return {**made, 'key': json.dumps(cipher.key), 'ciphertext': secret} if keys else made

def rows(texts, seed, each=0, keys=False, keyed=False):
    rng = random.Random(seed)
    sizes, kinds = (lengths_keyed if keyed else lengths), tuple(levels)
    texts = texts[:each * len(kinds)] if each else texts
    order = [kinds[i % len(kinds)] for i in range(len(texts))]

    if not each:
        rng.shuffle(order)

    found, seen = [], set()

    for n, (text, level) in enumerate(zip(texts, order)):
        # Graded levels and every keyed row get a length bucket, round-robin in held-out splits so each cell fills evenly
        # and random in train; size 0 lets row() draw a length between the unicity bound and room
        size = 0 if level not in graded and not keyed else sizes[n // len(kinds) % len(sizes)] if each else rng.choice(sizes)
        made = row(text, level, rng, size, keys, keyed)

        if made['id'] not in seen:
            seen.add(made['id'])
            found.append(made)

    return found