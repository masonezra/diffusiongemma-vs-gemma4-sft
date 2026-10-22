'''
The whole evaluation, as specified in note.md, from one adapters file {ar and every DiffusionGemma arm by name}:
    M0, M1  step use on test: exact match and symbol accuracy per rung at the T budgets of settings on every panel and T = 48
            on the carrier {128 episodes per rung}, Gemma 4 once; learned by the carrier's criterion {sudoku: R1 puzzle
            accuracy >= 50%; shift_register: k = 1 letter accuracy >= 50%}; k*(T) shifts with paired bootstrap intervals:
            one = k*(2) - k*(1) of the reference arm {diffusion}, steps = k*(48) - k*(2)
    M2      the training forward at T = 2 {hold}: symbol accuracy against the reference arm at T = 1 above its k*(1), and
            the loss from hold to the sampled T = 2
    M3-M6   for a learned arm, the T = 2 ladder on the carrier, val then test: use, sufficiency {P1}, interchange {P2},
            legibility; rounds chosen on val, tested on test
    M7      for a learned arm, the T = 48 ladder {donors, 128 episodes per rung; full, zero, top 1, top 3, resample 3, quantise} on
            the four rungs nearest k*(48), and {full, zero, top 3} inside Google's shipped sampler; commit order against depth from the traced run; the probe of
            the unread state on every arm with a ladder and on the released checkpoint; effective steps per arm from one
            shipped-sampler pass
    chain   for every DiffusionGemma arm, the T = 2 ladder {use, sufficiency, interchange by exact match} on the chain panel, val then test,
            whether or not the carrier was learned: the same carrier test on a published task whose state the models already track
Rounds per step: k*(T) / T, which counts the rounds of a single pass, and beside it the rounds added per step between
consecutive budgets, (k*(T') - k*(T)) / (T' - T), with k*(1) of the reference arm as the single-pass rounds.
--skip-chain leaves the chain ladder out. Runs already on disk with the same arguments are reused. --reanalyse rescores the
runs on disk without generating, keeping legibility, commit order and the probe from the previous evaluation.json where their
traces are absent. Output: results/state_tracking/<recipe>/evaluation.json; every reply file, its argument sidecar and the traces
under results/state_tracking/<recipe>/runs/.
    python -m state_tracking.meta.evaluate.evaluate --adapters adapters.json --gpus 1
'''

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from common.files import read_jsonl
from common.shards import join, run_on_gpus
from state_tracking.meta.benchmark import carrier, chain, config_patch, nocot
from state_tracking.meta.config.settings import budgets, data_dir, handoff, interventions, long, long_episodes, long_kinds, margin, results_dir, runs_dir, shipped_kinds
from state_tracking.meta.evaluate.crossing import crossing, shift
from state_tracking.meta.evaluate.generate import select, shard_suffix
from state_tracking.meta.evaluate.score import mark
from state_tracking.meta.evaluate.stats import above, summary
from state_tracking.meta.interpretability import chain_ladder, commit, ladder, legibility, probe

previous = None

def generate(split, model, run, checkpoint, args, steps=0, channel='full', hold=False, donors=False, trace=False, states=False, machines=(), rungs=(), limit=None, sampler='design'):
    limit = args.limit if limit is None else min(limit, args.limit or limit)
    path = runs_dir / f'{split}.{model}.{run}.jsonl'
    sidecar = path.with_name(path.name + '.args.json')
    manifest = {'checkpoint': checkpoint, 'steps': steps, 'channel': channel, 'hold': hold, 'donors': donors, 'trace': trace, 'states': states,
                'machines': list(machines), 'rungs': list(rungs), 'limit': limit, 'batch': args.batch, 'gpus': args.gpus, 'sampler': sampler}

    # A run on disk is reused only when its sidecar matches these arguments; under --reanalyse nothing is generated
    if path.exists() and (previous is not None or sidecar.exists() and json.loads(sidecar.read_text(encoding='utf-8')) == manifest):
        return path

    if previous is not None:
        raise FileNotFoundError(path)

    command = [sys.executable, '-m', 'state_tracking.meta.evaluate.generate', split, model, '--run', run, '--steps', str(steps), '--channel', channel,
               '--batch', str(args.batch), '--limit', str(limit), '--sampler', sampler]
    command += ['--checkpoint', checkpoint] if checkpoint else []
    command += [flag for flag, on in (('--hold', hold), ('--donors', donors), ('--trace', trace), ('--states', states)) if on]
    command += ['--machines', *machines] if machines else []
    command += ['--rungs', *map(str, rungs)] if rungs else []
    parts = [f'{gpu}/{args.gpus}' for gpu in range(args.gpus)]
    run_on_gpus([command + ['--part', part] for part in parts])

    if args.gpus > 1:
        join([runs_dir / f'{split}.{model}.{run}{shard_suffix(part)}.jsonl' for part in parts], path)

    sidecar.write_text(json.dumps(manifest), encoding='utf-8')

    return path

def effective_steps(path):
    used = [reply['steps_used'] for reply in read_jsonl(path) if 'steps_used' in reply]

    return round(float(np.mean(used)), 2) if used else None

def outcomes(rows, path):
    replies = {reply['id']: reply['reply'] for reply in read_jsonl(path)}
    exact, symbols = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))

    for row in rows:
        if row['id'] in replies:
            found = mark(replies[row['id']], row)
            exact[row['machine']][row['horizon']][row['id']] = float(found['exact'])

            if 'symbol' in found:
                symbols[row['machine']][row['horizon']][row['id']] = found['symbol']

    return exact, symbols

def listed(by_rung):
    return {k: list(v.values()) for k, v in by_rung.items()}

def paired(first, second):
    common = {k: sorted(set(first[k]) & set(second[k])) for k in first if k in second}

    return {k: [first[k][uid] for uid in ids] for k, ids in common.items() if ids}, {k: [second[k][uid] for uid in ids] for k, ids in common.items() if ids}

def described(exact, symbols, floors):
    found = {}

    for machine, by_rung in exact.items():
        found[machine] = {'exact': {k: round(100 * np.mean(list(v.values())), 1) for k, v in sorted(by_rung.items())}, 'episodes': {k: len(v) for k, v in sorted(by_rung.items())}}
        found[machine].update({'symbols': {k: round(100 * np.mean(list(v.values())), 1) for k, v in sorted(symbols[machine].items())}} if machine in symbols else {})

        if machine == carrier.machine or machine in (chain.machine, config_patch.machine):
            found[machine]['crossing'] = crossing(listed(by_rung), floors.get(machine))

    return found

def step_use(rows, adapters, labels, reference_label, args, summary_):
    floors = {chain.machine: chain.blind_floor([row for row in rows if row['machine'] == chain.machine]),
              config_patch.machine: nocot.floor([row for row in rows if row['machine'] == config_patch.machine])}
    runs = {'ar': generate('test', 'ar', 'tuned', adapters['ar'], args)} if adapters.get('ar') else {}

    for label in labels:
        for steps in budgets:
            if steps > 1 or label == reference_label:
                extra = {'machines': (carrier.machine,), 'limit': long_episodes} if steps == long else {}
                runs[f'{label}.T{steps}'] = generate('test', 'diffusion', f'{label}.T{steps}', adapters[label], args, steps=steps, **extra)

    for label in labels:
        runs[f'{label}.shipped'] = generate('test', 'diffusion', f'{label}.shipped', adapters[label], args, steps=long, sampler='shipped')

    marks = {name: outcomes(rows, path) for name, path in runs.items()}
    summary_['step_use'] = {name: described(*found, floors) for name, found in marks.items()}
    summary_['effective_steps'] = {label: effective_steps(runs[f'{label}.shipped']) for label in labels}
    reference = marks.get(f'{reference_label}.T1', (None,))[0]

    def shifted(first, second, machine):
        return shift(*paired(first[machine], second[machine]), floors.get(machine))

    for label in labels:
        own = {steps: marks[f'{label}.T{steps}'][0] for steps in budgets if f'{label}.T{steps}' in marks}
        arm = summary_['arms'][label] = {'learned': any(carrier.learned(marks[f'{label}.T{steps}'][0][carrier.machine], rows) for steps in own)}

        for machine in (carrier.machine, chain.machine, config_patch.machine):
            if reference and handoff in own:
                arm.setdefault('one', {})[machine] = shifted(reference, own[handoff], machine)

            if 'ar' in marks and handoff in own:
                arm.setdefault('against_ar', {})[machine] = shifted(marks['ar'][0], own[handoff], machine)

        if handoff in own and long in own:
            arm['steps'] = {carrier.machine: shifted(own[handoff], own[long], carrier.machine)}

        k50 = {steps: summary_['step_use'][f'{label}.T{steps}'][carrier.machine]['crossing']['k50'] for steps in own}
        single = summary_['step_use'].get(f'{reference_label}.T1', {}).get(carrier.machine, {}).get('crossing', {}).get('k50')
        k50 = {1: single, **k50} if single is not None else k50
        budgets_ = sorted(k50)
        arm['rounds_per_step'] = {steps: round(k50[steps] / steps, 3) for steps in own}
        arm['single_pass_rounds'] = single
        arm['rounds_added_per_step'] = {f'{a}-{b}': round((k50[b] - k50[a]) / (b - a), 3) for a, b in zip(budgets_, budgets_[1:])}
        arm['uses_a_second_step'] = bool(arm.get('one', {}).get(carrier.machine, {}).get('low', 0) > 0)
        arm['iterates'] = bool(arm.get('steps', {}).get(carrier.machine, {}).get('low', 0) > 0)

    if 'ar' in marks:
        summary_['arms']['ar'] = {'learned': carrier.learned(marks['ar'][0][carrier.machine], rows)}

def transition(rows, adapters, labels, reference_label, args, summary_):
    held = [row for row in rows if row['machine'] == carrier.machine]
    reference = summary_['step_use'].get(f'{reference_label}.T1', {}).get(carrier.machine, {}).get('crossing', {}).get('k50')
    replies = lambda path: {reply['id']: reply['reply'] for reply in read_jsonl(path)}
    one = replies(runs_dir / f'test.diffusion.{reference_label}.T1.jsonl') if reference is not None else {}

    for label in labels:
        hold = replies(generate('test', 'diffusion', f'{label}.T{handoff}.hold', adapters[label], args, steps=handoff, hold=True, machines=(carrier.machine,)))
        sampled = replies(runs_dir / f'test.diffusion.{label}.T{handoff}.jsonl')
        accuracy = lambda found, row: carrier.mark(found.get(row['id'], ''), row)['symbol']
        above_one = [row for row in held if reference is not None and row['horizon'] > reference]
        summary_['arms'][label]['transition'] = {
            'symbols_by_rung': {k: round(100 * np.mean([accuracy(hold, row) for row in held if row['horizon'] == k]), 1) for k in sorted({row['horizon'] for row in held})},
            'over_one_pass': summary([accuracy(hold, row) - accuracy(one, row) for row in above_one]) if one else None,
            'sampled_minus_hold': summary([accuracy(sampled, row) - accuracy(hold, row) for row in held])}
        found = summary_['arms'][label]['transition']
        found['learned_a_transition'] = bool(found['over_one_pass'] and above(found['over_one_pass'], 0))
        found['transfers_to_sampling'] = bool(found['sampled_minus_hold']['low'] is not None and found['sampled_minus_hold']['low'] >= -margin)

def ladder_runs(split, label, checkpoint, args, steps, rungs=(), donors=False, machines=None, run=None, trace=True, kinds=interventions, sampler='design'):
    machines = machines or (carrier.machine,)
    run = run or f'{label}.T{steps}.ladder'

    for kind in kinds:
        generate(split, 'diffusion', run + ladder.tag(kind), checkpoint, args, steps=steps, channel=kind, donors=donors, trace=trace and kind == 'full',
                 states=trace and kind == 'full' and steps == handoff, machines=machines, rungs=rungs, limit=long_episodes if steps == long else None, sampler=sampler)

    return run

def carrier_ladders(rows, adapters, labels, args, summary_):
    present = sorted({row['horizon'] for row in rows if row['machine'] == carrier.machine})

    for label in labels:
        arm = summary_['arms'].get(label)

        if not arm or not arm['learned']:
            continue

        run = ladder_runs('val', label, adapters[label], args, handoff)
        loaded = ladder.load('val', run)
        _, table = ladder.compare(*loaded)
        chosen = ladder.choose(table, loaded[0])
        readable = legibility.load('val', run) if previous is None else None
        read_round = legibility.choose(readable[0], readable[2]) if readable else None

        ladder_runs('test', label, adapters[label], args, handoff)
        arm['ladder'] = ladder.compare(*ladder.load('test', run), chosen)[0]
        kept = previous['arms'][label] if previous is not None else {}
        arm['legibility'] = kept['legibility'] if kept else legibility.summarise(*legibility.load('test', run), read_round)

        k50 = summary_['step_use'][f'{label}.T{long}'][carrier.machine]['crossing']['k50']
        rungs = tuple(sorted(sorted(present, key=lambda k: abs(k - k50))[:4]))
        run = ladder_runs('test', label, adapters[label], args, long, rungs, donors=True, kinds=long_kinds)
        arm['long_ladder'] = ladder.compare(*ladder.load('test', run), chosen)[0]
        arm['long_legibility'] = kept['long_legibility'] if kept else legibility.summarise(*legibility.load('test', run))
        arm['commit_order'] = kept['commit_order'] if kept else commit.analyse('test', run)
        shipped = ladder_runs('test', label, adapters[label], args, long, rungs, donors=True, run=f'{label}.shipped.ladder', trace=False,
                              kinds=shipped_kinds, sampler='shipped')
        arm['shipped_ladder'] = ladder.compare(*ladder.load('test', shipped), chosen)[0]

def carrier_chain(adapters, labels, args, summary_):
    kinds = ('full', 'swap', 'top 8', 'resample 8', 'resample 1')

    for label in labels:
        arm = summary_['arms'].get(label)

        if not arm:
            continue

        run = ladder_runs('val', label, adapters[label], args, handoff, machines=(chain_ladder.machine,), run=f'{label}.T{handoff}.chain', trace=False, kinds=kinds)
        loaded = chain_ladder.load('val', run)
        _, table = chain_ladder.compare(*loaded)
        chosen = chain_ladder.choose(table, loaded[0])
        ladder_runs('test', label, adapters[label], args, handoff, machines=(chain_ladder.machine,), run=run, trace=False, kinds=kinds)
        arm['chain_ladder'] = chain_ladder.compare(*chain_ladder.load('test', run), chosen)[0]

def unread(adapters, labels, args, summary_):
    arms = [label for label in labels if summary_['arms'].get(label, {}).get('ladder')]

    if not arms:
        return

    generate('test', 'diffusion', f'base.T{handoff}.ladder', None, args, steps=handoff, trace=True, states=True, machines=(carrier.machine,))
    names = [name for name in (*arms, 'base') if previous is None or (runs_dir / f'test.diffusion.{name}.T{handoff}.ladder').is_dir()]
    summary_['probe'] = {name: probe.probe('test', f'{name}.T{handoff}.ladder') for name in names}
    summary_['probe'].update({name: {'status': 'traces not on disk; the earlier estimate was invalid'} for name in (*arms, 'base') if name not in names})

def outcome(arm):
    found = arm.get('ladder', {})

    if not arm['learned']:
        return 'O1 not learned'

    if not (arm['uses_a_second_step'] or arm['iterates']):
        return 'O2 no serial use of steps'

    if found.get('P1') == 'insufficient':
        return 'O4 hidden carrier: state used outside the decisive reading'

    if found.get('P1') == 'sufficient' and (found.get('carried') or found.get('moved')):
        how = '' if found.get('carried') else ' {by the shift; the strict criterion is not met}'
        return ('O3 legible carrier' if arm.get('legibility', {}).get('state_legible') else 'O5 coded reading: sufficient, intermediate round carried, not legible') + how

    if found.get('P1') == 'sufficient' or not found.get('informative'):
        return 'O6 steps help, no intermediate round carried by the handed state'

    return 'undetermined'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--skip-chain', action='store_true')
    parser.add_argument('--reanalyse', action='store_true')
    args = parser.parse_args()
    global previous
    previous = json.loads((results_dir / 'evaluation.json').read_text(encoding='utf-8')) if args.reanalyse else None
    # An intervention needs a donor, so a limit keeps at least 2 episodes per cell
    args.limit = max(args.limit, 2) if args.limit else 0
    adapters = json.loads(open(args.adapters, encoding='utf-8').read())
    labels = [label for label in adapters if label != 'ar' and adapters[label]]
    reference_label = 'diffusion' if 'diffusion' in labels else (labels[0] if labels else None)
    rows = select(read_jsonl(data_dir / 'test.jsonl'), limit=args.limit)
    summary_ = {'arms': {}, 'reference': reference_label}
    step_use(rows, adapters, labels, reference_label, args, summary_)
    transition(rows, adapters, labels, reference_label, args, summary_)
    carrier_ladders(rows, adapters, labels, args, summary_)
    if not args.skip_chain:
        carrier_chain(adapters, labels, args, summary_)

    unread(adapters, labels, args, summary_)

    for label in labels:
        summary_['arms'][label]['outcome'] = outcome(summary_['arms'][label])

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'evaluation.json').write_text(json.dumps(summary_, indent=1, default=str), encoding='utf-8')
    print(json.dumps({label: {key: arm.get(key) for key in ('learned', 'uses_a_second_step', 'iterates', 'rounds_per_step', 'outcome')} for label, arm in summary_['arms'].items()}, indent=1))

if __name__ == '__main__':
    main()