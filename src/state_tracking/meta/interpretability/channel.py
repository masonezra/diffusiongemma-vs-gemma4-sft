'''
Interventions on the self-conditioning state S = s softmax(z) W_E that one denoising step hands to the next, and the
record Traced keeps. With p = softmax(z), K the k likeliest tokens, m = 1 - sum_K p and l = min_K p:
    top k        p on K, m spread uniformly elsewhere {arXiv 2606.20560, Algorithm 1: the monitor's re-embedding}
    resample k   p on K, the rest taken from the donor's p, rescaled to mass m and capped at l; the share of m placed
                 by headroom where the cap binds is logged as spread {over the whole canvas}
    swap         the donor's whole state
    zero         S = 0, the state the model trains with on half of its examples {in-distribution ablation}
    quantise     at every reply position, p uniform on the carrier's candidate set of that position {sudoku: the solver's
                 candidates after round 1, the given digit on givens}: a reading that holds the solver's state and nothing else
    edit j       at one reply position of depth j, p put on another admissible symbol {the counterfactual of the interchange test};
                 the canvas token at that position is rewritten too unless the canvas is held
Interventions apply to the state handed on, so not after the last step. The donor of row n is row n + 1 of its batch {one cell}. At T = 2 the donor's state comes from step 1, which no
intervention reaches; at T > 2 donors are un-intervened copies appended to the batch. hold keeps the starting noise
canvas at every step, so at T = 2 the run is the training forward: pass 1 with S = 0, pass 2 with S, same canvas. Reply
positions are located on the step's argmax canvas as the first run of width symbol tokens at the carrier's stride.
The record holds per step the top-8 ids and probabilities, the canvas and the probabilities of the alphabet tokens;
with states, also S handed from step 1 to step 2 and the monitor's re-embedding of its top-8 reading.
Reference: https://arxiv.org/pdf/2606.20560 {Section 3.1, Algorithm 1}
Reference: https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y/p/JvZhhzycHu2Yd57RN {resampling ablation}
'''

import inspect

import torch
from transformers import DiffusionGemmaForBlockDiffusion

top = 8

def parse(name):
    words = name.split()

    return (words[0], int(words[1])) if len(words) == 2 else (words[0], None)

def donors_of(size, device, donors):
    if donors:
        # Row n < half reads its donor copy at half + n; the copies read themselves and are left alone
        half = size // 2
        index = torch.cat([torch.arange(half, size), torch.arange(half, size)])

        return index.to(device), torch.arange(size, device=device) < half

    return torch.roll(torch.arange(size, device=device), -1), torch.ones(size, dtype=torch.bool, device=device)

def kept_mask(p, amount):
    return torch.zeros_like(p, dtype=torch.bool).scatter_(-1, p.topk(amount, -1).indices, True)

def squash(logits, kept):
    # Each dropped logit is set to c with |V \ K| e^c = sum_{V \ K} e^z, so the dropped mass is spread uniformly over V \ K
    rest = logits.float().masked_fill(kept, float('-inf'))
    count = (~kept).sum(-1, keepdim=True).clamp(min=1).float()
    constant = torch.logsumexp(rest, -1, keepdim=True) - count.log()

    return torch.where(kept, logits.float(), constant).to(logits.dtype)

def resampled(p, kept, donor):
    # q = p on K; off K the dropped mass m is refilled from the donor's p, capped at min_K p so K stays the top k, and what
    # the cap leaves over is spread over the headroom under it
    own = torch.where(kept, p, 0.0)
    mass = 1 - own.sum(-1, keepdim=True)
    least = torch.where(kept, p, 1.0).min(-1, keepdim=True).values
    other = torch.where(kept, 0.0, donor)
    other = (other * (mass / other.sum(-1, keepdim=True).clamp(min=1e-9))).minimum(least)
    room = torch.where(kept, 0.0, least - other)
    left = (mass - other.sum(-1, keepdim=True)).clamp(min=0)
    spread = (left / mass.clamp(min=1e-9)).squeeze(-1).mean(-1)

    return own + other + room * (left / room.sum(-1, keepdim=True).clamp(min=1e-9)), spread

def first_of(canvas, alphabet_ids, width, stride):
    symbol = torch.isin(canvas, alphabet_ids)
    found = torch.full((canvas.shape[0],), -1, dtype=torch.long, device=canvas.device)

    for first in range(canvas.shape[1] - stride * (width - 1)):
        hit = symbol[:, first:first + stride * width:stride].all(-1) & (found < 0)
        found = torch.where(hit, torch.full_like(found, first), found)

    return found

class Traced(DiffusionGemmaForBlockDiffusion):
    intervention, hold, donors, record, states, monitor, spread, finished, step, watch = 'full', False, False, None, None, None, None, None, 0, None
    alphabet_ids, width, stride, candidates, edit = None, 0, 1, None, None

    def __init__(self, config):
        super().__init__(config)
        assert 'current_canvas' in inspect.signature(super()._denoising_step).parameters, 'transformers changed the denoising step'
        self.model.decoder.self_conditioning.register_forward_pre_hook(self.capture)

    def begin(self, intervention='full', hold=False, donors=False, record=False, states=False, candidates=None, edit=None):
        self.intervention, self.hold, self.donors, self.step = intervention, hold, donors, 0
        self.record, self.finished, self.spread = ([] if record else None), [], []
        self.states, self.monitor = ([] if states else None), ([] if states else None)
        self.candidates, self.edit = candidates, edit

    def capture(self, module, args):
        if self.states is not None:
            self.states.append(args[1].detach())

        if self.intervention == 'zero':
            return (args[0], torch.zeros_like(args[1]), *args[2:])

    def embedded(self, logits):
        embed = self.model.decoder.embed_tokens

        return torch.matmul(logits.softmax(-1, dtype=torch.float32).to(embed.weight.dtype), embed.weight) * embed.embed_scale.to(embed.weight.dtype)

    def positions(self, argmax):
        first = first_of(argmax, self.alphabet_ids, self.width, self.stride)

        return first, first[:, None] + self.stride * torch.arange(self.width, device=argmax.device)[None, :]

    def quantised(self, logits, argmax):
        first, where = self.positions(argmax)
        lowest = torch.finfo(logits.dtype).min
        over = torch.full((logits.shape[0], self.width, logits.shape[-1]), lowest, dtype=logits.dtype, device=logits.device)
        over.scatter_(-1, self.alphabet_ids[None, None, :].expand(logits.shape[0], self.width, -1), torch.where(self.candidates, 0.0, float(lowest)).to(logits.dtype))
        changed = logits.clone()
        rows = (first >= 0).nonzero().squeeze(-1)

        for b in rows.tolist():
            changed[b, where[b]] = over[b]

        return changed

    def edited(self, logits, argmax, canvas):
        first, where = self.positions(argmax)
        lowest = torch.finfo(logits.dtype).min
        changed, canvas, argmax = logits.clone(), canvas.clone(), argmax.clone()

        for b, (position, token) in enumerate(self.edit):
            if first[b] >= 0 and position >= 0:
                at = int(where[b, position])
                changed[b, at] = lowest
                changed[b, at, token] = 0.0
                argmax[b, at] = token

                if not self.hold:
                    canvas[b, at] = token

        return changed, canvas, argmax

    def _denoising_step(self, *args, **kwargs):
        canvas, argmax, logits, done = super()._denoising_step(*args, **kwargs)
        kind, amount = parse(self.intervention)
        p = logits.float().softmax(-1)
        self.step += 1
        self.finished.append(done.cpu().numpy().copy())

        if self.hold:
            canvas = kwargs['current_canvas']

        if self.record is not None:
            found = p.topk(top, -1)
            self.record.append({'ids': found.indices.int().cpu().numpy(), 'probs': found.values.half().cpu().numpy(), 'canvas': canvas.int().cpu().numpy()})

            if self.watch is not None:
                self.record[-1]['watch'] = p[..., self.watch].half().cpu().numpy()

        if self.monitor is not None and self.step == 1:
            self.monitor.append(self.embedded(squash(logits, kept_mask(p, top))))

        # cur_step counts down, so step 1 is the last, whose state is never handed on
        if kind not in ('full', 'zero') and kwargs['cur_step'] > 1:
            if kind == 'quantise':
                logits = self.quantised(logits, argmax)

            elif kind == 'edit':
                logits, canvas, argmax = self.edited(logits, argmax, canvas)

            else:
                assert p.shape[0] > 1, kind
                index, mine = donors_of(p.shape[0], p.device, self.donors)

                if kind == 'swap':
                    changed = logits[index]

                elif kind == 'top':
                    changed = squash(logits, kept_mask(p, amount))

                else:
                    q, spread = resampled(p, kept_mask(p, amount), p[index])
                    changed = torch.where(q > 0, q.log(), torch.finfo(logits.dtype).min).to(logits.dtype)
                    self.spread.append(spread.cpu().numpy())

                logits = torch.where(mine[:, None, None], changed, logits)

        return canvas, argmax, logits, done