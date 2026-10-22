'''
Prompt format shared by both models: one instruction, the ciphertext, and a single user/assistant turn; the reply is the
plaintext alone in both recipes. Recipe 1 names neither the cipher nor the key; recipe 2 prints the cipher name and the
full key before the ciphertext, in the same symbols the ciphertext uses, so every level is the application of a given key.
'''

instruction = 'Decipher the ciphertext. The plaintext is English in the letters a to z; the cipher type and the key are not given. Where the ciphertext has _ it marks a gap between words and stays as it is. Reply with only the plaintext: as many characters as the ciphertext has symbols, separated by spaces.'
label = 'Ciphertext: '
instruction_keyed = ('Decipher the ciphertext with the cipher and key given. The plaintext is English in the letters a to z. Where the ciphertext '
                     'has _ it marks a gap between words and stays as it is. Reply with only the plaintext: as many characters as the '
                     'ciphertext has symbols, separated by spaces.')
cipher_label = 'Cipher: '

def chat(question, reply):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}]