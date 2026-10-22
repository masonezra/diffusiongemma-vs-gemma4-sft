'''
Splits by stone on the last digit of the EDCS id (3 = test, 4 = val, else train), as in Ithaca and Aeneas.
Drops training stones sharing half or more of their 30-letter shingles with held-out stones (carved and edition letters).
Reference: https://github.com/google-deepmind/predictingthepast/blob/main/train/dataloader.py
'''

import collections

split_by_digit = {3: 'test', 4: 'val'}
held_out = ('val', 'test')
window = 30
max_overlap = 0.5
views = ('letters', 'edition')

def assign_split(uid):
    return split_by_digit.get(int(uid.split('-')[1]) % 10, 'train')

def shingles(text, width=window):
    # hash() is salted per process, but both sides are hashed in this one, so the overlap test is unaffected
    return {hash(text[i:i + width]) for i in range(max(1, len(text) - width + 1))}

def split_items(items, counts):
    parts = collections.defaultdict(list)

    for item in items:
        parts[assign_split(item['id'])].append(item)

    held = {view: set().union(*(shingles(item[view]) for split in held_out for item in parts[split])) for view in views}
    clean = []

    for item in parts['train']:
        own = {view: shingles(item[view]) for view in views}

        if all(len(own[view] & held[view]) < max_overlap * len(own[view]) for view in views):
            clean.append(item)

    counts['held_copy'] = len(parts['train']) - len(clean)
    parts['train'] = clean

    return parts