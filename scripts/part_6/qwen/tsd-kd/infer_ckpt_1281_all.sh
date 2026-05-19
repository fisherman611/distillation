#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

CKPT_PATH="${CKPT_PATH:-tsd-kd-qwen-ckpt/tsd-kd/qwen/no_seq_kd/20260518_145716/checkpoint-1281}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/infer/tsd-kd/qwen/no_seq_kd/20260518_145716/checkpoint-1281}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FLUSH_EVERY="${FLUSH_EVERY:-${BATCH_SIZE}}"
DB="${DB:-full}"
DATA_SOURCE="${DATA_SOURCE:-auto}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARKS="${BENCHMARKS:-Cypherbench Mind_the_query Neo4j_Text2Cypher}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

infer_max_length_for() {
  local benchmark="$1"

  case "${benchmark}" in
    Cypherbench) printf '1034' ;;
    Neo4j_Text2Cypher) printf '3092' ;;
    Mind_the_query) printf '2427' ;;
    *) printf '1024' ;;
  esac
}

echo "Checkpoint: ${CKPT_PATH}"
echo "Model: ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Flush every: ${FLUSH_EVERY}"
echo "DB: ${DB}"

for benchmark in ${BENCHMARKS}; do
  max_length="$(infer_max_length_for "${benchmark}")"
  output_dir="${OUTPUT_ROOT}/${benchmark}"
  output_path="${output_dir}/${DB}_cyphers_result.json"
  mkdir -p "${output_dir}"

  cmd=(
    "${PYTHON_BIN}" infer.py
    --benchmark "${benchmark}"
    --db "${DB}"
    --data_source "${DATA_SOURCE}"
    --model "${MODEL_NAME}"
    --ckpt_path "${CKPT_PATH}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --flush-every "${FLUSH_EVERY}"
    --max-length "${max_length}"
    --output_path "${output_path}"
  )

  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  echo "[infer] benchmark=${benchmark} max_length=${max_length} output=${output_path}"
  "${cmd[@]}"
done

echo "All inference jobs finished: ${OUTPUT_ROOT}"
