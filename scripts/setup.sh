#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq && apt-get install -y -qq tmux rsync >/dev/null
pip install -q hf_xet

grep -q ALPHA_ENV ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/hf
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH=/workspace/alpha/src
export ALPHA_CHECKPOINTS=/workspace/checkpoints
EOF
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 PYTHONPATH=/workspace/alpha/src ALPHA_CHECKPOINTS=/workspace/checkpoints

hf download google/gemma-4-26B-A4B-it
hf download google/diffusiongemma-26B-A4B-it

python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('nemo-automodel', 'transformers', 'torch')})"
python -c "from transformers import DiffusionGemmaForBlockDiffusion, AutoModelForImageTextToText; print('transformers classes ok')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true