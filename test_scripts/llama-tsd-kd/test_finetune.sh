#! /usr/bin/env bash

set -euo pipefail

beta="${BETA:-0.9}"
lmbda="${LAMBDA:-1.0}"
threshold="${THRESHOLD:-0.1}"
model_name="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
indirect_kd_alpha="${INDIRECT_KD_ALPHA:-0.1}"
teacher_model_name="${TEACHER_MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
seq_kd="${SEQ_KD:-0}"

max_train_samples="${MAX_TRAIN_SAMPLES:-32}"
max_eval_samples="${MAX_EVAL_SAMPLES:-8}"
max_steps="${MAX_STEPS:-5}"
batch_size="${BATCH_SIZE:-1}"
eval_batch_size="${EVAL_BATCH_SIZE:-1}"
grad_acc="${GRAD_ACC:-1}"
output_dir="${OUTPUT_DIR:-results/tsd-kd-smoke/llama}"

if [[ -n "${RUN_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RUN_GPUS}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS_PER_NODE="${#GPUS[@]}"
else
  GPUS_PER_NODE=1
fi
MASTER_PORT="${RUN_MASTER_PORT:-29500}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "Missing dependency: accelerate is not installed or not on PATH." >&2
  echo "Install project dependencies in the active environment: bash install.sh" >&2
  echo "Or minimally install accelerate: python -m pip install accelerate" >&2
  exit 127
fi

cmd=(
  accelerate
  launch
  --config_file TSD-KD/accelerate_ddp_config.yaml
  --num_processes "${GPUS_PER_NODE}"
  --main_process_port "${MASTER_PORT}"
  TSD-KD/train.py
  --beta "${beta}"
  --lmbda "${lmbda}"
  --threshold "${threshold}"
  --model-name "${model_name}"
  --indirect-kd-alpha "${indirect_kd_alpha}"
  --teacher-model-name "${teacher_model_name}"
  --output-dir "${output_dir}"
  --max-train-samples "${max_train_samples}"
  --max-eval-samples "${max_eval_samples}"
  --max-steps "${max_steps}"
  --num-train-epochs 1
  --per-device-train-batch-size "${batch_size}"
  --per-device-eval-batch-size "${eval_batch_size}"
  --gradient-accumulation-steps "${grad_acc}"
  --logging-steps 1
  --no-load-best-model-at-end
)

if [[ "${seq_kd}" == "1" || "${seq_kd}" == "true" ]]; then
  cmd+=(--seq-kd)
fi

echo "${cmd[*]}"
"${cmd[@]}"

if [[ ! -d "${output_dir}" ]]; then
  echo "Smoke train failed: output dir was not created: ${output_dir}" >&2
  exit 1
fi

echo "Smoke train finished: ${output_dir}"
