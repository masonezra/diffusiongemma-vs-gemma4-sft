'''
The shift_register panel: a nonlinear feedback shift register on Z_16^16 {16 letters a..p}, driven by k input letters; the reply
is the state x^(k) after k steps, none of them written. Step t with input a in Z_16 is F_a = S o rho_a: rho_a turns the ring left by
a places, then S replaces two cells from their left neighbours, x_0 <- R[x_15][x_0] and x_8 <- R[x_7][x_8], with R one fixed table of
16 permutations of Z_16 {one per neighbour value, the same in every episode, printed in every prompt}. rho_a is a permutation of the
positions and S a bijection of Z_16^16, so every F_a is a permutation of the state space and x^(k) = F_{a_k} o ... o F_{a_1}(x^(0)) is a
word in the group the F_a generate: the state-tracking setting of arXiv 2404.08819, with the intermediate states unwritten. One step
changes two letters, so a model is rewarded for copying from the first example on and learns the lookup on top, while the k-step
composition depends on the row at every step and has no associative shortcut.
The control shift_register_single has the same prompt but x^(t) = F_{a_t}(x^(0)), so its answer needs one step; the two are paired
by episode and split by MD5(start row) mod 8. It ran as recipe1 of this benchmark. Both DiffusionGemma rules and Gemma 4 stayed
at chance beyond one step {see note.md}, so I replaced it with the sudoku carrier {recipe2} and kept this code.
Reference: Golomb 1967, Shift Register Sequences {nonlinear feedback shift registers}
Reference: https://arxiv.org/pdf/2404.08819 {The Illusion of State: state tracking as a word problem}
Reference: https://arxiv.org/pdf/2606.20560 {Section 2.2, serial depth}
'''

import hashlib
import json
import random

from state_tracking.meta.config.prompts import chat
from state_tracking.meta.config.settings import letters, shift_register_horizons, shift_register_rungs, width

nibbles = 'abcdefghijklmnop'
table = ''.join(''.join(random.Random(2026 + n).sample(nibbles, 16)) for n in range(16))
machines = ('shift_register', 'shift_register_single')
reads = {
    'shift_register': 'Each step starts from the row the step before left.',
    'shift_register_single': 'Each step starts from the starting row, not from the row the step before left.',
}

def opening(machine, width):
    return (f'Run the ring. The row holds {width} letters, each a to p, in a ring: the letter after the last is the first, and the '
            'letter before the first is the last. Table has one row for each letter a to p in order, and each row lists 16 letters, one '
            'for each letter a to p in order. Step t uses input t. A step has two parts. First the row is turned: the input of the step '
            'says how many places, a none, b one, c two and so on; every letter moves that many places toward the front, and letters '
            'pushed past the front reappear at the end. Then the 1st letter and the 9th letter are replaced, each the same way: take the '
            'Table row of the letter before it, and in that row take the entry in the place of the letter itself, a the 1st, b the 2nd '
            f'and so on; the 9th letter uses the 8th letter as it was before this replacement. {reads[machine]} '
            f'Spare letters are not used. After the last step reply with the row, {width} letters separated by spaces.')

def value(letter):
    return nibbles.index(letter)

def turn(row, letter):
    shift = value(letter) % len(row)

    return row[shift:] + row[:shift]

def step(row, letter, table):
    row = turn(row, letter)
    first, ninth = table[16 * value(row[-1]) + value(row[0])], table[16 * value(row[7]) + value(row[8])]

    return first + row[1:8] + ninth + row[9:]

def advance(row, inputs, table):
    for letter in inputs:
        row = step(row, letter, table)

    return row

def trajectory(case):
    row, stages = case['start'], []

    for letter in case['inputs']:
        row = step(row if case['machine'] == 'shift_register' else case['start'], letter, case['table'])
        stages.append(row)

    return stages

def episode(horizon, rng):
    return {'width': width, 'horizon': horizon, 'table': table,
            'start': ''.join(rng.choices(nibbles, k=width)), 'inputs': ''.join(rng.choices(nibbles, k=horizon)),
            'spare': ''.join(rng.choices(nibbles, k=letters - horizon))}

def start_key(start):
    return hashlib.md5(start.encode()).hexdigest()

def partition(start):
    residue = int(start_key(start), 16) % 8

    return 'val' if residue == 0 else 'test' if residue == 1 else 'train'

def episode_key(case):
    return hashlib.md5(f"{case['start']} {case['table']} {case['inputs']} {case['spare']}".encode()).hexdigest()

def lines(case):
    return ['Table:', *(f'{a}: {" ".join(case["table"][16 * n:16 * n + 16])}' for n, a in enumerate(nibbles)),
            f'Inputs: {" ".join(case["inputs"])}', f'Spare: {" ".join(case["spare"])}',
            f'Steps: {case["horizon"]}', f'Row: {" ".join(case["start"])}']

def render(case, machine, held=False):
    case = dict(case, machine=machine)
    question = f'{opening(machine, case["width"])}\n\n' + '\n'.join(lines(case))
    stages = trajectory(case)
    row = {
        'id': hashlib.md5(question.encode()).hexdigest(), 'machine': machine, 'horizon': case['horizon'], 'split': partition(case['start']),
        'cluster': start_key(case['start']), 'episode': episode_key(case), 'messages': chat(question, ' '.join(stages[-1])),
    }

    if held:
        row['truth'] = json.dumps({'start': case['start'], 'table': case['table'], 'inputs': case['inputs'], 'spare': case['spare'], 'stages': stages})

    return row

def draw(horizon, split, rng, seen):
    while True:
        case = episode(horizon, rng)

        if partition(case['start']) == split and episode_key(case) not in seen:
            seen.add(episode_key(case))

            return case

def shares(episodes):
    weights = [1 / horizon for horizon in shift_register_horizons]
    counts = [int(episodes * weight / sum(weights)) for weight in weights]
    counts[0] += episodes - sum(counts)

    return counts

def rows(count, single, seed):
    rng, seen, found, every = random.Random(seed), set(), [], count // single

    for horizon, episodes in zip(shift_register_horizons, shares(count)):
        for _ in range(episodes):
            case = draw(horizon, 'train', rng, seen)
            found += [render(case, machine) for machine in machines[:1 + (len(seen) % every == 0)]]

    return found

def panel(each, seed, split):
    rng, seen, found = random.Random(seed), set(), []

    for horizon in shift_register_rungs:
        for _ in range(each):
            case = draw(horizon, split, rng, seen)
            found += [render(case, machine, held=True) for machine in machines]

    return found

def mark(reply, row):
    truth = row['messages'][1]['content'].split()
    guess = reply.strip().split()

    return {'exact': guess == truth, 'symbol': sum(a == b for a, b in zip(guess, truth)) / len(truth)}

def cell(row):
    return f"{row['machine']} k={row['horizon']}"