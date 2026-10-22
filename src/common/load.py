'''
Loads either model in bfloat16 from Hugging Face transformers and merges a NeMo LoRA checkpoint: W <- W + (alpha / r) B A.
DiffusionGemma's encoder and decoder share tensors (asserted), so NeMo's one adapter per shared linear is merged once,
as NeMo's own generator does; an adapter key with no matching weight raises.
Reference: https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/dllm_generate/utils.py
'''

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from common.models import names, templates

key_maps = {'ar': {}, 'diffusion': {'model.layers.': 'model.decoder.layers.', 'model.self_conditioning.': 'model.decoder.self_conditioning.'}}

def adapter_dir(checkpoint):
    for folder in (Path(checkpoint) / 'model', Path(checkpoint)):
        if (folder / 'adapter_model.safetensors').is_file():
            return folder

    raise FileNotFoundError(f'no adapter_model.safetensors under {checkpoint}')

def merge(model, checkpoint, key_map):
    folder = adapter_dir(checkpoint)
    tensors = load_file(folder / 'adapter_model.safetensors')
    settings = json.loads((folder / 'adapter_config.json').read_text(encoding='utf-8'))
    weights, scale, merged = model.state_dict(), settings['lora_alpha'] / settings['r'], 0

    for key in tensors:
        if not key.endswith('.lora_A.weight'):
            continue

        base = key.removesuffix('.lora_A.weight').removeprefix('base_model.model.')

        for old, new in key_map.items():
            base = base.replace(old, new, 1)

        if f'{base}.weight' not in weights:
            raise KeyError(f'adapter key {key} has no weight {base}.weight in {type(model).__name__}')

        weight = weights[f'{base}.weight']
        delta = tensors[f'{key.removesuffix(".lora_A.weight")}.lora_B.weight'].float() @ tensors[key].float() * scale
        weight.add_(delta.to(weight.device, weight.dtype))
        merged += 1

    if not merged:
        raise ValueError(f'no LoRA pairs in {folder}')

    return merged

def load(kind, checkpoint=None, cls=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(names[kind])
    tokenizer.chat_template = templates[kind].read_text(encoding='utf-8')
    tokenizer.padding_side = 'left'

    if kind == 'diffusion':
        from transformers import DiffusionGemmaForBlockDiffusion

        model = (cls or DiffusionGemmaForBlockDiffusion).from_pretrained(names[kind], dtype=torch.bfloat16, device_map=device)
        weights = dict(model.named_parameters(remove_duplicate=False))
        shared = 'layers.0.self_attn.q_proj.weight'

        if weights[f'model.decoder.{shared}'].data_ptr() != weights[f'model.encoder.language_model.{shared}'].data_ptr():
            raise ValueError('encoder and decoder do not share weights in this checkpoint: the merge would reach one pass only')

    else:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(names[kind], dtype=torch.bfloat16, device_map=device)

    if checkpoint:
        merge(model, checkpoint, key_maps[kind])

    return model.eval(), tokenizer