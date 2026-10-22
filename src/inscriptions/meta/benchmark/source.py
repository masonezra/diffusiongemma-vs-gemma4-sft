'''
Downloads the EDCS dump of 2022-09-12 (537,262 texts) once into the cache, MD5-checked, and parses it.
Keeps Latin texts with parseable markup, at least 10 carved letters and no doubtful reading; drops exact duplicates.
Reference: https://zenodo.org/records/7072337
'''

import hashlib
import json

import requests

from inscriptions.meta.benchmark.markup import carved_text, edition_text, editor_additions, gap, letters_only, normalise, parse_inscription
from inscriptions.meta.config.settings import cache_dir

url = 'https://zenodo.org/api/records/7072337/files/{name}/content'
file, md5 = 'EDCS_text_cleaned_2022-09-12.json', '7ee22c65f442d54ba3537955603cc0ba'
min_letters = 10
funerary_type = 'tituli sepulcrales'

def file_md5(path):
    digest = hashlib.md5()

    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)

    return digest.hexdigest()

def fetch():
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / 'edcs' / file
    path.parent.mkdir(exist_ok=True)

    if not path.exists() or file_md5(path) != md5:
        part = path.with_name(path.name + '.part')

        with requests.get(url.format(name=file), stream=True, timeout=120) as response, open(part, 'wb') as out:
            response.raise_for_status()

            for chunk in response.iter_content(1 << 20):
                out.write(chunk)

        if file_md5(part) != md5:
            part.unlink()

            raise ValueError(f'{file}: checksum mismatch after download, run again')

        part.replace(path)

    return path

def records():
    with open(fetch(), encoding='utf-8') as source:
        for record in json.load(source):
            yield {'id': record['EDCS-ID'], 'text': record.get('inscription') or '', 'language': record.get('language') or '',
                   'province': record.get('province') or '', 'type': record.get('inscr_type') or []}

def parse_record(record):
    raw = record['text']

    if not raw:
        return None, 'no_text'

    if record['language']:
        return None, 'other_language'

    if '?' in raw:
        return None, 'doubtful'

    text, marks = normalise(raw), parse_inscription(raw)

    if '()' in text:
        return None, 'unresolved'

    if marks is None:
        return None, 'other_script' if any(ord(char) > 127 for char in text) else 'broken_marks'

    letters = letters_only(carved_text(marks).replace(gap, ''))

    if len(letters) < min_letters:
        return None, 'short'

    expansions, restorations = editor_additions(marks)
    types = record['type']
    item = {
        'id': record['id'], 'text': text, 'marks': marks, 'letters': letters, 'edition': letters_only(edition_text(marks)), 'province': record['province'],
        'funerary': funerary_type in ([types] if isinstance(types, str) else list(types)), 'has_additions': bool(expansions or any(restorations.values())),
    }

    return item, None

def load_items(records, counts):
    seen = set()

    for record in records:
        counts['read'] += 1
        item, reason = parse_record(record)

        if item and item['text'] in seen:
            reason = 'duplicate'

        if reason:
            counts[reason] += 1
            continue

        seen.add(item['text'])

        yield item