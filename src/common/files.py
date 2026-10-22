'''
JSONL read/write and SHA-256 sealing of built files into report.json. LF line ends on every system, so a build hashes
the same on Windows and Linux.
'''

import hashlib
import json
from pathlib import Path

def iter_jsonl(path):
    with open(path, encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

def read_jsonl(path):
    return list(iter_jsonl(path))

def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')

def merge_jsonl(path, rows, field, value):
    kept = [row for row in iter_jsonl(path) if row.get(field) != value] if Path(path).exists() else []
    write_jsonl(path, kept + list(rows))

def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)

    return digest.hexdigest()

def count_lines(path):
    with open(path, 'rb') as file:
        return sum(block.count(b'\n') for block in iter(lambda: file.read(8 * 1024 * 1024), b''))

def seal(folder, **entries):
    path = Path(folder) / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    report.update(entries)
    report['files'] = {file.name: {'rows': count_lines(file), 'sha256': file_sha256(file)} for file in sorted(Path(folder).glob('*.jsonl'))}

    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        json.dump(report, file, indent=1)

    return report