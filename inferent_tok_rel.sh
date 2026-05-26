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
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
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

if ! [[ "${GPUS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_JOB must be a positive integer. Got: ${GPUS_PER_JOB}" >&2
  exit 1
fi

GPU_CHUNKS=()
for ((idx = 0; idx < ${#GPUS[@]}; idx += GPUS_PER_JOB)); do
  gpu_chunk="${GPUS[$idx]}"
  for ((offset = 1; offset < GPUS_PER_JOB && idx + offset < ${#GPUS[@]}; offset += 1)); do
    gpu_chunk="${gpu_chunk},${GPUS[$((idx + offset))]}"
  done
  GPU_CHUNKS+=("${gpu_chunk}")
done

QUEUE_DIR="${LOG_DIR}/queue_${$}"
QUEUE_PENDING_DIR="${QUEUE_DIR}/pending"
QUEUE_CLAIMED_DIR="${QUEUE_DIR}/claimed"
QUEUE_FAILED_DIR="${QUEUE_DIR}/failed"

mkdir -p "${QUEUE_PENDING_DIR}" "${QUEUE_CLAIMED_DIR}" "${QUEUE_FAILED_DIR}"

for idx in "${!EXPERIMENTS[@]}"; do
  printf -v queue_name "%06d" "${idx}"
  : > "${QUEUE_PENDING_DIR}/${queue_name}"
done

claim_next_job() {
  local worker_idx="$1"
  local queue_file
  local claimed_file
  local job_name

  while true; do
    queue_file="$(find "${QUEUE_PENDING_DIR}" -maxdepth 1 -type f | sort | head -n 1)"
    if [[ -z "${queue_file}" ]]; then
      return 1
    fi

    job_name="$(basename "${queue_file}")"
    claimed_file="${QUEUE_CLAIMED_DIR}/${job_name}.worker${worker_idx}"
    if mv "${queue_file}" "${claimed_file}" 2>/dev/null; then
      echo "${job_name}"
      return 0
    fi
  done
}

run_worker() {
  local worker_idx="$1"
  local gpu_chunk="$2"
  local idx
  local job_name
  local experiment
  local ckpt_path
  local output_path
  local log_path
  local status

  while job_name="$(claim_next_job "${worker_idx}")"; do
    idx=$((10#${job_name}))
    experiment="${EXPERIMENTS[$idx]}"
    ckpt_path="${HF_ROOT}/${experiment}/${CKPT_STEP}"
    output_path="${OUTPUT_DIR}/full_cyphers_result_Qwen3-0.6B_${experiment}_ckpt${CKPT_STEP}.json"
    log_path="${LOG_DIR}/${experiment}_ckpt${CKPT_STEP}.log"

    echo "[launch] gpus=${gpu_chunk} experiment=${experiment}"
    echo "         ckpt=${ckpt_path}"
    echo "         out=${output_path}"
    echo "         log=${log_path}"

    if (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu_chunk}"
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
    ) > "${log_path}" 2>&1; then
      echo "[done] gpus=${gpu_chunk} experiment=${experiment}"
    else
      status="$?"
      echo "[fail] gpus=${gpu_chunk} experiment=${experiment} exit=${status}" >&2
      echo "exit=${status} experiment=${experiment} log=${log_path}" > "${QUEUE_FAILED_DIR}/${job_name}"
    fi
  done
}

pids=()
for idx in "${!GPU_CHUNKS[@]}"; do
  run_worker "${idx}" "${GPU_CHUNKS[$idx]}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

job_failures="$(find "${QUEUE_FAILED_DIR}" -maxdepth 1 -type f | wc -l | tr -d '[:space:]')"
failures=$((failures + job_failures))

if [[ "${failures}" -gt 0 ]]; then
  echo "Finished with ${failures} failed inference job(s). Check logs in ${LOG_DIR}." >&2
  exit 1
fi

echo "All inference jobs finished. Results: ${OUTPUT_DIR}"
