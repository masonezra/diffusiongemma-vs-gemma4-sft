'''
Chat format: one user turn (instruction and carved text), one assistant turn (the edition).
The instruction's notation examples occur in no row, so the prompt gives away no answer.
'''

from inscriptions.meta.benchmark.markup import carved_text, edition_text, gap

instruction = f'Write the edition of this Latin inscription. It is given as cut on the stone, and {gap} marks lost letters. Expand each abbreviation in round brackets, as in arch(itectus). Restore lost letters in square brackets, as in medi[cus], and keep {gap} where they cannot be restored. Reply with only the edition.'
prompt_label = 'Inscription: '

def to_messages(marks):
    return [{'role': 'user', 'content': f'{instruction}\n\n{prompt_label}{carved_text(marks)}'}, {'role': 'assistant', 'content': edition_text(marks)}]

def prompt_text(row):
    return row['messages'][0]['content'].rsplit(prompt_label, 1)[1]

def target_text(row):
    return row['messages'][1]['content']