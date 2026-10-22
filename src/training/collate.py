'''
Collater of the Gemma 4 arm: NeMo's default collater behind the (examples, processor) signature the VLM recipe calls.
'''

from nemo_automodel.components.datasets.utils import default_collater

def padded(examples, processor=None, **kwargs):
    return default_collater(examples)