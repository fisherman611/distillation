#! /usr/bin/env bash

set -euo pipefail

export NCCL_DEBUG="${NCCL_DEBUG:-}"

python -m pip install --upgrade pip

# Versions required by TSD-KD.
python -m pip install torch==2.5.1
python -m pip install transformers==4.57.3
python -m pip install trl==0.21.0

# Shared dependencies used by TSD-KD and the existing distillation scripts.
python -m pip install \
  accelerate \
  datasets \
  deepspeed \
  nltk \
  numerize \
  peft \
  protobuf \
  rich \
  rouge-score \
  sentencepiece \
  torchtyping \
  wandb
