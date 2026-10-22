'''
Model ids, the vendored chat template of each, the shared 256-token answer budget (one DiffusionGemma canvas), and the
end tokens a reply stops on.
'''

from pathlib import Path

gemma = 'google/gemma-4-26B-A4B-it'
diffusion_gemma = 'google/diffusiongemma-26B-A4B-it'
names = {'ar': gemma, 'diffusion': diffusion_gemma}
templates = {'ar': Path(__file__).with_name('gemma4_chat_template.jinja'), 'diffusion': Path(__file__).with_name('diffusion_gemma_chat_template.jinja')}
canvas = 256

def count_tokens(rows, tokenizer):
    prompts = tokenizer([row['messages'][0]['content'] for row in rows], add_special_tokens=False)['input_ids']
    answers = tokenizer([row['messages'][1]['content'] for row in rows], add_special_tokens=False)['input_ids']

    for row, prompt, answer in zip(rows, prompts, answers):
        row['tokens'] = len(answer)
        row['total_tokens'] = len(prompt) + len(answer)

def end_tokens(tokenizer):
    ends = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<turn|>'), tokenizer.convert_tokens_to_ids('<|fim_suffix|>')}
    ends.discard(None)
    ends.discard(tokenizer.unk_token_id)

    return ends