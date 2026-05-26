#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
HF_ROOT="${HF_ROOT:-hf://fisherman611/text-to-cypher-llama-model/qwen3}"
CKPT_STEP="${CKPT_STEP:-2130}"
BENCHMARK="${BENCHMARK:-Mind_the_query}"
OUTPUT_DIR="${OUTPUT_DIR:-results/Mind_the_query}"
LOG_DIR="${LOG_DIR:-run_logs/inferent_tok_rel}"
GPU_LIST="${RUNNER_GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-2427}"

EXPERIMENTS=(
  "token_rel_loss_0.6B_4B_Cypherbench_csd"
  "token_rel_loss_0.6B_4B_Cypherbench_csd_kd0.4"
  "token_rel_loss_0.6B_4B_Cypherbench_csd_kd0.5"
  "token_rel_loss_0.6B_4B_Cypherbench_csd_kd0.6"
  "token_rel_loss_0.6B_4B_Cypherbench_adaptive_srkl_kd0.3"
  "token_rel_loss_0.6B_4B_Cypherbench_adaptive_srkl_kd0.4"
  "token_rel_loss_0.6B_4B_Cypherbench_adaptive_srkl_kd0.5"
  "token_rel_loss_0.6B_4B_Cypherbench_adaptive_srkl_kd0.6"
)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=', ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set RUNNER_GPUS, for example: RUNNER_GPUS=0,1,2,3" >&2
  exit 1
fi

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local idx
  local experiment
  local ckpt_path
  local output_path
  local log_path

  for ((idx = worker_idx; idx < ${#EXPERIMENTS[@]}; idx += ${#GPUS[@]})); do
    experiment="${EXPERIMENTS[$idx]}"
    ckpt_path="${HF_ROOT}/${experiment}/${CKPT_STEP}"
    output_path="${OUTPUT_DIR}/full_cyphers_result_Qwen3-0.6B_${experiment}_ckpt${CKPT_STEP}.json"
    log_path="${LOG_DIR}/${experiment}_ckpt${CKPT_STEP}.log"

    echo "[launch] gpu=${gpu} experiment=${experiment}"
    echo "         ckpt=${ckpt_path}"
    echo "         out=${output_path}"
    echo "         log=${log_path}"

    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON_BIN}" infer.py \
        --benchmark "${BENCHMARK}" \
        --model "${MODEL_NAME}" \
        --ckpt_path "${ckpt_path}" \
        --output_path "${output_path}" \
        --batch-size "${BATCH_SIZE}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --device "${DEVICE}" \
        --max-length "${MAX_LENGTH}"
    ) > "${log_path}" 2>&1
  done
}

pids=()
for idx in "${!GPUS[@]}"; do
  run_worker "${idx}" "${GPUS[$idx]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "Finished with ${failures} failed inference job(s). Check logs in ${LOG_DIR}." >&2
  exit 1
fi

echo "All inference jobs finished. Results: ${OUTPUT_DIR}"
