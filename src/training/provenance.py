'''
Provenance of a run, written beside its checkpoints on rank 0: host, time, package versions, the composed config, the LoRA parameter
count, dtype and modules, and SHA-256 of the chat template, the train and val files, the training code and recipe, and the first row's
token ids. check_template asserts that the training prefix equals the inference prompt and that the turn close is supervised;
check_lora asserts that every adapter weight is bfloat16 {NeMo's LoRA kernel needs adapter and base in one dtype}.
'''

import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

packages = ('nemo-automodel', 'transformers', 'torch')
turn_close = '<turn|>'

def sha256(path):
    digest = hashlib.sha256()

    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)

    return digest.hexdigest()

def version(package):
    try:
        return importlib.metadata.version(package)

    except importlib.metadata.PackageNotFoundError:
        return None

def check_template(config):
    from transformers import AutoTokenizer

    dataset = config['dataset']
    template = Path(dataset['chat_template']).read_text(encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(dataset['tokenizer']['pretrained_model_name_or_path'])

    with open(dataset['path_or_dataset_id'], encoding='utf-8') as file:
        messages = json.loads(next(file))['messages']

    rendered = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False)
    prompt = tokenizer.apply_chat_template(messages[:1], chat_template=template, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    close = tokenizer.convert_tokens_to_ids(turn_close)

    assert rendered.startswith(prompt), 'the training prefix differs from the inference prompt'
    assert any(token == close and kept for token, kept in zip(tokens['input_ids'], tokens['assistant_masks'])), 'the turn close is not supervised'

    return hashlib.sha256(json.dumps(tokens['input_ids']).encode()).hexdigest()

def check_lora(trainer):
    import torch

    found = {}

    for value in vars(trainer).values():
        for module in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(module, torch.nn.Module):
                found.update((name, weight) for name, weight in module.named_parameters() if 'lora_' in name)

    dtypes = sorted({str(weight.dtype) for weight in found.values()})

    if not found:
        raise ValueError('the trainer holds no LoRA parameters')

    if dtypes != ['torch.bfloat16']:
        raise ValueError(f'LoRA parameters are {dtypes}; both models train their adapters in bfloat16')

    return {'parameters': sum(weight.numel() for weight in found.values()), 'dtype': dtypes[0],
            'modules': sorted({re.sub(r'\.\d+\.', '.N.', name.split('.lora_')[0]) for name in found})}

def plain(value):
    if hasattr(value, '__qualname__'):
        return f'{value.__module__}.{value.__qualname__}'

    return str(value)

def record(config, task, token_ids_sha256, lora, recipe='recipe'):
    folder = Path(config['checkpoint']['checkpoint_dir'])
    here = Path(__file__).parent
    files = {key: config[key]['path_or_dataset_id'] for key in ('dataset', 'validation_dataset')}
    entry = {
        'host': platform.node(), 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'recipe': recipe,
        'versions': {package: version(package) for package in packages}, 'config': config, 'lora': lora,
        'template_sha256': sha256(config['dataset']['chat_template']), 'first_row_token_ids_sha256': token_ids_sha256,
        'dataset_sha256': {key: sha256(path) for key, path in files.items()},
        'code_sha256': {path.name: sha256(path) for path in sorted(here.iterdir()) if path.suffix in ('.py', '.yaml')} | {f'{recipe}.yaml': sha256(here.parent / task / f'{recipe}.yaml')},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"provenance.{entry['started'].replace(':', '')}.json").write_text(json.dumps(entry, indent=1, default=plain), encoding='utf-8')