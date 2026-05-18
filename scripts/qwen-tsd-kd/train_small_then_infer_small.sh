#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

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

if ! command -v accelerate >/dev/null 2>&1; then
  echo "Missing dependency: accelerate is not installed or not on PATH." >&2
  echo "Install project dependencies in the active environment: bash install.sh" >&2
  echo "Or minimally install accelerate: python -m pip install accelerate" >&2
  exit 127
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://fisherman611/text-to-cypher-models/e5-bs2-lr1e-05-G8-N2-NN1-lora-32-64-0.1/1065}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tsd-kd-smoke/qwen/train_small_then_infer_small}"

BETA="${BETA:-0.9}"
LMBDA="${LMBDA:-1.0}"
THRESHOLD="${THRESHOLD:-0.1}"
INDIRECT_KD_ALPHA="${INDIRECT_KD_ALPHA:-0.1}"
SEQ_KD="${SEQ_KD:-0}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-4}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-1}"
RUN_MASTER_PORT="${RUN_MASTER_PORT:-29500}"

INFER_BENCHMARK="${INFER_BENCHMARK:-Cypherbench}"
INFER_DB="${INFER_DB:-full}"
INFER_DATA_SOURCE="${INFER_DATA_SOURCE:-auto}"
INFER_LIMIT="${INFER_LIMIT:-20}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
INFER_OUTPUT_PATH="${INFER_OUTPUT_PATH:-${OUTPUT_DIR}/infer/${INFER_BENCHMARK}/${INFER_DB}_cyphers_result.json}"

infer_max_length_for() {
  case "$1" in
    Cypherbench) printf '1034' ;;
    Neo4j_Text2Cypher) printf '3092' ;;
    Mind_the_query) printf '2427' ;;
    *) printf '1034' ;;
  esac
}

INFER_MAX_LENGTH="${INFER_MAX_LENGTH:-$(infer_max_length_for "${INFER_BENCHMARK}")}"

train_cmd=(
  accelerate launch
  --config_file TSD-KD/accelerate_ddp_config.yaml
  --num_processes "${GPUS_PER_NODE}"
  --main_process_port "${RUN_MASTER_PORT}"
  TSD-KD/train.py
  --beta "${BETA}"
  --lmbda "${LMBDA}"
  --threshold "${THRESHOLD}"
  --model-name "${MODEL_NAME}"
  --indirect-kd-alpha "${INDIRECT_KD_ALPHA}"
  --teacher-model-name "${TEACHER_MODEL_NAME}"
  --teacher-peft-path "${TEACHER_PEFT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --max-steps "${MAX_STEPS}"
  --num-train-epochs 1
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACC}"
  --logging-steps 1
  --no-load-best-model-at-end
)

if [[ "${SEQ_KD}" == "1" || "${SEQ_KD}" == "true" ]]; then
  train_cmd+=(--seq-kd)
fi

echo "[train] ${train_cmd[*]}"
"${train_cmd[@]}"

ckpt_path="$(
  python - "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
markers = {
    "adapter_config.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
}
candidates = []
if root.is_dir():
    for child in (root, *root.rglob("*")):
        if child.is_dir() and any((child / marker).exists() for marker in markers):
            step = int(child.name.split("-")[-1]) if child.name.startswith("checkpoint-") and child.name.split("-")[-1].isdigit() else -1
            candidates.append((step, child.stat().st_mtime, child))
if candidates:
    print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
)"

if [[ -z "${ckpt_path}" ]]; then
  echo "No checkpoint found under ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${INFER_OUTPUT_PATH}")"

infer_cmd=(
  python infer.py
  --benchmark "${INFER_BENCHMARK}"
  --db "${INFER_DB}"
  --data_source "${INFER_DATA_SOURCE}"
  --model "${MODEL_NAME}"
  --ckpt_path "${ckpt_path}"
  --device cuda
  --batch-size "${INFER_BATCH_SIZE}"
  --max-length "${INFER_MAX_LENGTH}"
  --limit "${INFER_LIMIT}"
  --output_path "${INFER_OUTPUT_PATH}"
)

echo "[infer] checkpoint=${ckpt_path}"
echo "[infer] ${infer_cmd[*]}"
"${infer_cmd[@]}"

echo "Done."
echo "Checkpoint: ${ckpt_path}"
echo "Inference output: ${INFER_OUTPUT_PATH}"
