'''
Chat format: the program with [MASK] at each gap in, the missing code out as one python fence per gap, in order.
Reference: https://huggingface.co/datasets/gonglinyuan/safim
'''

import re

gap = '[MASK]'
opening, closing = '```python\n', '\n```'
instruction = f'Fill in the code missing at each {gap}. Reply with only the missing code, one code block for each {gap} in order.'
heading = '\n\nTask:\n'
block = re.compile(r'```python\n(.*?)\n```', re.DOTALL)

def answer(middles):
    return '\n'.join(f'{opening}{middle}{closing}' for middle in middles)

def chat(context, middles, task=''):
    code = f'{opening}{gap.join(context)}{closing}'
    question = f'{instruction}{heading}{task}\n\n{code}' if task else f'{instruction}\n\n{code}'

    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer(middles)}]

def task_of(question):
    return question.split(heading, 1)[1].rsplit('\n\n' + opening, 1)[0] if heading in question else ''

def pieces(reply):
    found = block.findall(reply)

    if found:
        return found

    return [reply.split(opening, 1)[1].removesuffix('```') if opening in reply else reply]

def shown(question):
    return question[question.rindex(opening) + len(opening):question.rindex(closing)].split(gap)

def fill(question, reply):
    context = shown(question)
    given = pieces(reply) + [''] * len(context)

    return ''.join(part + piece for part, piece in zip(context[:-1], given)) + context[-1]

def solution(row):
    return fill(row['messages'][0]['content'], row['messages'][1]['content'])

def blanks(row):
    middles = pieces(row['messages'][1]['content'])

    return [fill(row['messages'][0]['content'], answer(middles[:at] + [''] + middles[at + 1:])) for at in range(len(middles))]