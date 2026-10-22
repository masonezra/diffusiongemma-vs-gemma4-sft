'''
The chat format of a row: one user turn holding the question and one assistant turn holding the answer alone.
'''

def chat(question, reply):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}]