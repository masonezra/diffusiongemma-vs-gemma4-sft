'''
Composes one NeMo config from shared.yaml + <model>.yaml + src/<task>/<recipe>.yaml and the arm's own keys {model, self_conditioning_grad
and any NeMo overrides}. Steps = rows / global batch for one pass, or max_steps when smaller; warmup 3% of the steps, checkpoint and
validation every tenth; one GPU gives expert-parallel size 1 with the torch dispatcher; the run folder is $ALPHA_CHECKPOINTS/<run>/<arm>
with <run> = <task> or <task>-<recipe>. Returns the config and whether the arm trains through the self-conditioning state.
'''

import copy
import os
from pathlib import Path

import yaml

from common.models import names, templates

here = Path(__file__).parent
src = here.parent
data = src.parent / 'dataset'
checkpoints = Path(os.environ.get('ALPHA_CHECKPOINTS', '/workspace/checkpoints'))
warmup_share = 0.03
evaluations = 10

def read(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)

        else:
            base[key] = copy.deepcopy(value)

    return base

def count_rows(path):
    with open(path, 'rb') as file:
        return sum(1 for _ in file)

def compose(task, arm, name='recipe'):
    recipe = read(src / task / f'{name}.yaml')
    extra = dict(recipe['arms'][arm])
    model, rule = extra.pop('model'), extra.pop('self_conditioning_grad', False)
    folder = data / task / recipe.get('dataset', '')
    train = folder / 'train.jsonl'
    epochs = recipe.get('epochs', 1)
    steps = min(recipe.get('max_steps', total := epochs * (count_rows(train) // recipe['global_batch_size'])), total)
    every = max(steps // evaluations, 1)
    run = task if name == 'recipe' else f'{task}-{name}'
    config = merge(read(here / 'shared.yaml'), read(here / f'{model}.yaml'))

    merge(config, {
        'step_scheduler': {'global_batch_size': recipe['global_batch_size'], 'local_batch_size': recipe['local_batch_size'], 'max_steps': steps,
                           'ckpt_every_steps': every, 'val_every_steps': every, 'num_epochs': epochs},
        'lr_scheduler': {'lr_warmup_steps': round(warmup_share * steps)},
        'model': {'pretrained_model_name_or_path': names[model]},
        'peft': {'dim': recipe['lora_rank'], 'alpha': recipe['lora_alpha']},
        'checkpoint': {'checkpoint_dir': str(checkpoints / run / arm)},
        'dataset': {'path_or_dataset_id': str(train), 'seq_length': recipe['seq_length'], 'chat_template': str(templates[model]),
                    'tokenizer': {'pretrained_model_name_or_path': names[model]}},
    })
    world = int(os.environ.get('WORLD_SIZE', '8'))
    merge(config, {'distributed': {'ep_size': min(8, world)}})

    if world == 1:
        merge(config, {'model': {'backend': {'dispatcher': 'torch'}}})

    if os.environ.get('WANDB_API_KEY'):
        merge(config, {'wandb': {'project': 'alpha', 'name': f'{task}.{arm}'}})

    if 'canvas' in recipe and model == 'diffusion':
        width = recipe['canvas']
        merge(config, {'model': {'canvas_length': width}, 'dllm': {'block_size': width, 'pad_block_size': width, 'pad_seq_len_divisible': width}})

    config['validation_dataset'] = merge(copy.deepcopy(config['dataset']), {'path_or_dataset_id': str(folder / recipe.get('validation', 'val.jsonl')), 'split': 'train'})

    return merge(config, extra), rule