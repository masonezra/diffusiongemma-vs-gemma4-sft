'''
Runs one generation process per GPU and joins their reply files in shard order.
'''

import os
import subprocess
from pathlib import Path

def run_on_gpus(commands):
    processes = [subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}) for gpu, command in enumerate(commands)]
    failed = [command for command, process in zip(commands, processes) if process.wait()]

    if failed:
        raise RuntimeError(f'{len(failed)} of {len(commands)} shards failed: {failed[0]}')

def join(parts, out):
    with open(out, 'w', encoding='utf-8') as file:
        for part in map(Path, parts):
            file.write(part.read_text(encoding='utf-8'))
            part.unlink()

    return out