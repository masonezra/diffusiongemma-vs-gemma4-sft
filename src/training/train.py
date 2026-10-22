'''
Trains one arm of one benchmark with NeMo Automodel: composes the config {recipe.compose}, installs the gradient-through-S forward for
arms with self_conditioning_grad {self_conditioning.install}, checks the chat template and the LoRA dtype, records provenance on rank 0,
and runs NeMo's train and validation loop. Extra arguments are NeMo dotted overrides.
    torchrun --nproc-per-node 8 -m training.train inscriptions diffusion
    torchrun --nproc-per-node 1 -m training.train cryptanalysis ar --recipe recipe2
Reference: https://github.com/NVIDIA-NeMo/Automodel {recipes dllm.train_ft and vlm.finetune}
'''

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

import yaml

from training.provenance import check_lora, check_template, record
from training.recipe import compose
from training.self_conditioning import install

recipes = {
    'DiffusionGemmaSFTRecipe': ('nemo_automodel.recipes.dllm.train_ft', 'DiffusionGemmaSFTRecipe'),
    'FinetuneRecipeForVLM': ('nemo_automodel.recipes.vlm.finetune', 'FinetuneRecipeForVLM'),
}

def main():
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('arm')
    parser.add_argument('--recipe', default='recipe')
    args, options = parser.parse_known_args()
    rank = int(os.environ.get('RANK', '0'))
    config, rule = compose(args.task, args.arm, args.recipe)
    path = Path(tempfile.gettempdir()) / f'alpha.{args.task}.{args.recipe}.{args.arm}.{rank}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    sys.argv = [sys.argv[0], '-c', str(path), *options]
    cfg = parse_args_and_load_config()
    settings = cfg.to_dict()

    if rule:
        install(settings['checkpoint']['checkpoint_dir'])

    token_ids_sha256 = check_template(settings) if rank == 0 else None
    module, name = recipes[settings['recipe']]
    trainer = getattr(importlib.import_module(module), name)(cfg)
    trainer.setup()
    lora = check_lora(trainer)

    if rank == 0:
        record(settings, args.task, token_ids_sha256, lora, args.recipe)

    trainer.run_train_validation_loop()

if __name__ == '__main__':
    main()