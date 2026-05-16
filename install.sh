#! /usr/bin/env bash

set -euo pipefail

export NCCL_DEBUG="${NCCL_DEBUG:-}"

python -m pip install --upgrade pip

# RTX 50/Blackwell GPUs such as RTX 5090 require PyTorch CUDA 12.8+ wheels
# with sm_120 kernels. The original TSD-KD torch==2.5.1 wheel only supports
# up to sm_90.
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url "${PYTORCH_CUDA_INDEX_URL}"

# Versions required by TSD-KD.
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
