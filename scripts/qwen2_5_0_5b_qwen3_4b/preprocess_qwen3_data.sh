#! /usr/bin/env bash

set -euo pipefail

BASE_PATH="${BASE_PATH:-.}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
RAW_DATA_DIR="${RAW_DATA_DIR:-}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${BASE_PATH}/processed_data/qwen3_4b_instruct_2507}"
DATA_PROCESS_WORKERS="${DATA_PROCESS_WORKERS:-8}"
MAX_LENGTH="${MAX_LENGTH:-892}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-797}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-797}"

if [[ -z "${RAW_DATA_DIR}" ]]; then
  echo "Set RAW_DATA_DIR to the directory containing train.jsonl, dev.jsonl, and test.jsonl." >&2
  exit 1
fi

export PYTHONPATH="${BASE_PATH}"
export TOKENIZERS_PARALLELISM=false

for SPLIT in train valid test; do
  CMD=(
    python "${BASE_PATH}/process_data.py"
    --model-path "${MODEL_PATH}"
    --model-type "${MODEL_TYPE}"
    --data-dir "${RAW_DATA_DIR}"
    --processed-data-dir "${PROCESSED_DATA_DIR}"
    --split "${SPLIT}"
    --data-process-workers "${DATA_PROCESS_WORKERS}"
    --max-length "${MAX_LENGTH}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --t-max-prompt-length "${T_MAX_PROMPT_LENGTH}"
  )

  echo "${CMD[*]}"
  "${CMD[@]}"
done

