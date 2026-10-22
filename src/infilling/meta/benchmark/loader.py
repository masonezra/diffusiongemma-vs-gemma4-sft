'''
Loads the training sources and public infilling benchmarks from Hugging Face, each pinned to one revision. The
benchmarks match the three training gap kinds: HumanEval-Infilling single line, multi line and random span; SAFIM
block and control.
Every record carries tests and a group (a contest problem, a HumanEval problem): the unit of the split and of the
clustered intervals. A training source is read in a seeded shuffled order, so a quota or --limit takes a uniform
sample of it rather than the head of the file {KodCode is stored by subset, OpenCodeInstruct by domain}.
Reference: https://huggingface.co/datasets/deepmind/code_contests
Reference: https://huggingface.co/datasets/KodCode/KodCode-V1
Reference: https://huggingface.co/datasets/nvidia/OpenCodeInstruct
Reference: https://huggingface.co/datasets/loubnabnl/humaneval_infilling
Reference: https://huggingface.co/datasets/gonglinyuan/safim
Reference: https://huggingface.co/datasets/evalplus/humanevalplus
'''

import hashlib
import json

revisions = {
    'deepmind/code_contests': '802411c3010cb00d1b05bad57ca77365a3c699d6',
    'KodCode/KodCode-V1': '97d81f0695ddd7c5c02bfe16436db9a17a4a21e5',
    'nvidia/OpenCodeInstruct': '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d',
    'loubnabnl/humaneval_infilling': 'ad46002f24b153968a3d0949e6fa9576780530ba',
    'gonglinyuan/safim': 'be132cc15372e90b6f03a608e77f2d940e384edb',
    'evalplus/humanevalplus': 'd32357cf319e50e9c8d8dab5ea876c72b0fd321b',
}

python3_label = 3
max_io_tests = 10

def load(repository, config=None, split='train', data_file=None):
    from datasets import load_dataset

    if data_file:
        return load_dataset('json', data_files=f'hf://datasets/{repository}@{revisions[repository]}/{data_file}', split='train')

    return load_dataset(repository, config, split=split, revision=revisions[repository])

def codecontests_records(row):
    tests = [{'input': given, 'output': [wanted]} for part in ('public_tests', 'private_tests', 'generated_tests') for given, wanted in zip(row[part]['input'], row[part]['output'])]
    solutions = [solution for solution, label in zip(row['solutions']['solution'], row['solutions']['language']) if label == python3_label]

    for index, solution in enumerate(solutions):
        yield f'{row["name"]}:{index}', solution, json.dumps(tests[:max_io_tests]), 'io', row['description'].strip(), row['name']

def kodcode_records(row):
    yield row['question_id'], row['solution'], row['test'], 'pytest', row['question'].strip(), ''

def opencodeinstruct_records(row):
    if float(row['average_test_score']) == 1.0:
        tests = '\n'.join(test.strip() for test in json.loads(row['unit_tests']))

        yield row['id'], row['output'], tests, 'asserts', row['input'].strip(), ''

train_sources = {
    'codecontests': ({'repository': 'deepmind/code_contests'}, codecontests_records),
    'kodcode': ({'repository': 'KodCode/KodCode-V1'}, kodcode_records),
    'opencodeinstruct': ({'repository': 'nvidia/OpenCodeInstruct'}, opencodeinstruct_records),
}

def fetch_dataset(name, limit=0, seed=0):
    location, to_records = train_sources[name]
    rows = load(**location).shuffle(seed=seed)

    for row in rows.select(range(min(limit, len(rows)))) if limit else rows:
        yield from to_records(row)

def humaneval_records():
    for subset in ('SingleLine', 'MultiLine', 'RandomSpan'):
        gap_kind = 'random_span' if subset == 'RandomSpan' else 'line_span'

        for row in load('loubnabnl/humaneval_infilling', data_file=f'data/HumanEval-{subset}Infilling.jsonl'):
            tests = f'{row["test"]}\ncheck({row["entry_point"]})'

            yield row['task_id'], gap_kind, row['prompt'], row['canonical_solution'], row['suffix'], tests, 'asserts', '', '/'.join(row['task_id'].split('/')[1:3])

def safim_records():
    for subset, gap_kind in (('block_v2', 'block'), ('control_fixed', 'control')):
        rows = load('gonglinyuan/safim', subset, split='test')

        for row in rows.filter(lambda language: language == 'python', input_columns='lang'):
            prefix, suffix = row['eval_prompt'].split('{{completion}}')
            task = row['prompt'].split('\n\n', 1)[1].rsplit('\n\nCode:\n\n', 1)[0]
            group = row.get('problem_source') or hashlib.sha1((task or row['task_id']).encode()).hexdigest()[:12]

            yield row['task_id'], gap_kind, prefix, row['ground_truth'], suffix, row['unit_tests'], 'io', task, group

benchmarks = {'humaneval': humaneval_records, 'safim': safim_records}

def fetch_eval(name):
    return benchmarks[name]()