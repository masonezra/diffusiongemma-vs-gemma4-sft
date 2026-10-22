'''
The gradient rule of the diffusion_gradient arm. NeMo's released forward computes the self-conditioning logits of pass 1 under no_grad
and detaches them; install replaces DiffusionGemmaForBlockDiffusion.forward with one that runs pass 1 through NeMo's own forward with
self-conditioning off, hands its logits to pass 2 without detaching, and lets the loss of pass 2 reach pass 1. The gradient norm at the
handed logits is logged every 50 forwards to channel_gradient.jsonl in the run folder.
Reference: https://github.com/NVIDIA-NeMo/Automodel {components/models/diffusion_gemma/model.py}
'''

import json
import os
from pathlib import Path

gradient_log, forwards, log_every, original = None, 0, 50, None

def forward_gradient(self, input_ids=None, *, canvas_ids=None, self_conditioning_logits=None, do_self_conditioning=None, **kwargs):
    if self_conditioning_logits is None and self.self_conditioning:
        self_conditioning_logits = original(self, input_ids, canvas_ids=canvas_ids, self_conditioning_logits=None, do_self_conditioning=False, **kwargs).logits

        if self_conditioning_logits.requires_grad:
            self_conditioning_logits.register_hook(note_gradient)

        if do_self_conditioning is None:
            do_self_conditioning = True

    return original(self, input_ids, canvas_ids=canvas_ids, self_conditioning_logits=self_conditioning_logits, do_self_conditioning=do_self_conditioning, **kwargs)

def note_gradient(gradient):
    global forwards
    forwards += 1

    if gradient_log and forwards % log_every == 0 and int(os.environ.get('RANK', '0')) == 0:
        norms = gradient.float().flatten(1).norm(dim=1)

        with open(gradient_log, 'a', encoding='utf-8') as file:
            file.write(json.dumps({'forward': forwards, 'norm_mean': float(norms.mean()), 'norm_max': float(norms.max())}) + '\n')

def install(folder):
    global gradient_log, original
    from nemo_automodel.components.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion

    gradient_log = Path(folder) / 'channel_gradient.jsonl'
    gradient_log.parent.mkdir(parents=True, exist_ok=True)
    original = DiffusionGemmaForBlockDiffusion.forward
    DiffusionGemmaForBlockDiffusion.forward = forward_gradient