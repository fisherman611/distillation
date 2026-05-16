#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="${ROOT_DIR}/scripts"

MODE="parallel"
GPU_LIST="${RUNNER_GPUS:-0,1}"
GPUS_PER_JOB=1
FILTER=""
MAX_RETRIES=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
LOG_DIR=""
INFER_AFTER_TRAIN=0
INFER_SCRIPT=""
INFER_BENCHMARKS="Cypherbench,Mind_the_query,Neo4j_Text2Cypher"
INFER_DB="full"
INFER_DATA_SOURCE="auto"
INFER_BATCH_SIZE=1
INFER_MAX_LENGTH="auto"
INFER_LIMIT=""
INFER_EXTRA_ARGS=""
INFER_OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage: ./running.sh [options]

Run all .sh training scripts under ./scripts. Each script file is treated as
a complete experiment config. The runner passes RUN_GPUS and RUN_MASTER_PORT
to each child script.

Options:
  --mode <parallel|sequential>   Run with a dynamic GPU queue or one by one. Default: parallel
  --gpus <list>                  Comma-separated GPU ids. Default: RUNNER_GPUS or 0
  --gpus-per-job <n>             GPUs used by each script. Default: 1
  --filter <pattern>             Only run scripts whose path contains this substring.
  --max-retries <n>              Retry a failed script up to n times. Default: 0
  --log-dir <path>               Directory for runner logs. Default: ./run_logs/<timestamp>
  --continue-on-error            Keep scheduling after a script exhausts retries.
  --dry-run                      Print what would run without launching jobs.
  --infer-after-train            Run infer.py on the latest checkpoint after each successful train script.
  --infer-script <path>          Inference script. Default: ./infer.py
  --infer-benchmark <name>       Single benchmark passed to infer.py.
  --infer-benchmarks <list>      Comma-separated benchmarks. Default: all supported benchmarks
  --infer-db <name|full>         DB/subset passed to infer.py. Default: full
  --infer-data-source <source>   Data source passed to infer.py. Default: auto
  --infer-batch-size <n>         Batch size passed to infer.py. Default: 1
  --infer-max-length <n|auto>    Max generation length. Default: auto by model family and benchmark
  --infer-limit <n>              Optional sample limit passed to infer.py.
  --infer-extra-args <string>    Extra raw arguments appended to infer.py.
  --infer-output-root <path>     Output root for inference JSON files. Default: ./results/infer
  -h, --help                     Show this message.

Auto infer max length:
  Qwen:  Cypherbench=1034, Neo4j_Text2Cypher=3092, Mind_the_query=2427
  Llama: Cypherbench=1053, Neo4j_Text2Cypher=2884, Mind_the_query=2445

Examples:
  ./running.sh --dry-run
  ./running.sh --filter updated_span_schema_query_new_data/qwen/csd
  ./running.sh --filter kd0.7 --gpus 0,1,2,3 --gpus-per-job 1
  ./running.sh --mode sequential --filter finetune/qwen/sft
  ./running.sh --max-retries 1 --continue-on-error
  ./running.sh --filter kd0.8 --infer-after-train
  ./running.sh --filter kd0.8 --infer-after-train --infer-benchmark Cypherbench --infer-db movie
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --gpus-per-job)
      GPUS_PER_JOB="$2"
      shift 2
      ;;
    --filter)
      FILTER="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --infer-after-train)
      INFER_AFTER_TRAIN=1
      shift
      ;;
    --infer-script)
      INFER_SCRIPT="$2"
      shift 2
      ;;
    --infer-benchmark)
      INFER_BENCHMARKS="$2"
      shift 2
      ;;
    --infer-benchmarks)
      INFER_BENCHMARKS="$2"
      shift 2
      ;;
    --infer-db)
      INFER_DB="$2"
      shift 2
      ;;
    --infer-data-source)
      INFER_DATA_SOURCE="$2"
      shift 2
      ;;
    --infer-batch-size)
      INFER_BATCH_SIZE="$2"
      shift 2
      ;;
    --infer-max-length)
      INFER_MAX_LENGTH="$2"
      shift 2
      ;;
    --infer-limit)
      INFER_LIMIT="$2"
      shift 2
      ;;
    --infer-extra-args)
      INFER_EXTRA_ARGS="$2"
      shift 2
      ;;
    --infer-output-root)
      INFER_OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${SCRIPT_ROOT}" ]]; then
  echo "Script root does not exist: ${SCRIPT_ROOT}" >&2
  exit 1
fi

if [[ "${MODE}" != "parallel" && "${MODE}" != "sequential" ]]; then
  echo "--mode must be 'parallel' or 'sequential'" >&2
  exit 1
fi

if ! [[ "${GPUS_PER_JOB}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_JOB}" -le 0 ]]; then
  echo "--gpus-per-job must be a positive integer" >&2
  exit 1
fi

if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "--max-retries must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "${INFER_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${INFER_BATCH_SIZE}" -le 0 ]]; then
  echo "--infer-batch-size must be a positive integer" >&2
  exit 1
fi

if [[ "${INFER_MAX_LENGTH}" != "auto" ]] && { ! [[ "${INFER_MAX_LENGTH}" =~ ^[0-9]+$ ]] || [[ "${INFER_MAX_LENGTH}" -le 0 ]]; }; then
  echo "--infer-max-length must be 'auto' or a positive integer" >&2
  exit 1
fi

if [[ -n "${INFER_LIMIT}" ]] && ! [[ "${INFER_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "--infer-limit must be a non-negative integer" >&2
  exit 1
fi

if [[ "${MODE}" == "parallel" && "${DRY_RUN}" -ne 1 ]]; then
  if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Parallel mode requires Bash >= 5.1 for 'wait -n -p'. Use --mode sequential or upgrade Bash." >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON:-python}"
if [[ -z "${INFER_SCRIPT}" ]]; then
  INFER_SCRIPT="${ROOT_DIR}/infer.py"
elif [[ "${INFER_SCRIPT}" != /* ]]; then
  INFER_SCRIPT="${ROOT_DIR}/${INFER_SCRIPT}"
fi

if [[ "${INFER_AFTER_TRAIN}" -eq 1 && ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script does not exist: ${INFER_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${INFER_OUTPUT_ROOT}" ]]; then
  INFER_OUTPUT_ROOT="${ROOT_DIR}/results/infer"
elif [[ "${INFER_OUTPUT_ROOT}" != /* ]]; then
  INFER_OUTPUT_ROOT="${ROOT_DIR}/${INFER_OUTPUT_ROOT#./}"
fi

case "${INFER_BENCHMARKS}" in
  all|All|ALL)
    INFER_BENCHMARKS="Cypherbench,Mind_the_query,Neo4j_Text2Cypher"
    ;;
esac

IFS=', ' read -r -a RAW_INFER_BENCHMARKS <<< "${INFER_BENCHMARKS}"
INFER_BENCHMARK_LIST=()
for benchmark in "${RAW_INFER_BENCHMARKS[@]}"; do
  if [[ "${benchmark}" == "Neo4j_Text2ypher" ]]; then
    benchmark="Neo4j_Text2Cypher"
  fi
  if [[ -n "${benchmark}" ]]; then
    INFER_BENCHMARK_LIST+=("${benchmark}")
  fi
done

if [[ "${INFER_AFTER_TRAIN}" -eq 1 && "${#INFER_BENCHMARK_LIST[@]}" -eq 0 ]]; then
  echo "No inference benchmarks configured." >&2
  exit 1
fi
INFER_BENCHMARK_DISPLAY="$(IFS=,; echo "${INFER_BENCHMARK_LIST[*]}")"

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${ROOT_DIR}/run_logs/${timestamp}"
fi
JOB_LOG_DIR="${LOG_DIR}/jobs"
INFER_LOG_DIR="${LOG_DIR}/infer"
mkdir -p "${JOB_LOG_DIR}"
if [[ "${INFER_AFTER_TRAIN}" -eq 1 ]]; then
  mkdir -p "${INFER_LOG_DIR}"
  mkdir -p "${INFER_OUTPUT_ROOT}"
fi

RUN_LOG="${LOG_DIR}/run.log"
SUCCESS_LOG="${LOG_DIR}/success.log"
FAIL_LOG="${LOG_DIR}/failed.log"
RETRY_LOG="${LOG_DIR}/retry.log"
INFER_SUCCESS_LOG="${LOG_DIR}/infer_success.log"
INFER_FAIL_LOG="${LOG_DIR}/infer_failed.log"

: > "${RUN_LOG}"
: > "${SUCCESS_LOG}"
: > "${FAIL_LOG}"
: > "${RETRY_LOG}"
: > "${INFER_SUCCESS_LOG}"
: > "${INFER_FAIL_LOG}"

append_log() {
  local log_file="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" >> "${log_file}"
}

rel_path() {
  local path="$1"
  printf '%s' "${path#${ROOT_DIR}/}"
}

script_id() {
  local rel
  rel="$(rel_path "$1")"
  rel="${rel//\//__}"
  rel="${rel//\\/__}"
  printf '%s' "${rel%.sh}"
}

script_root_name() {
  local script="$1"
  local inside

  inside="${script#${SCRIPT_ROOT}/}"
  inside="${inside#scripts/}"
  printf '%s' "${inside%%/*}"
}

script_experiment_id() {
  local script="$1"
  local inside

  inside="${script#${SCRIPT_ROOT}/}"
  inside="${inside#scripts/}"
  if [[ "${inside}" == */* ]]; then
    inside="${inside#*/}"
  fi
  inside="${inside//\//__}"
  inside="${inside//\\/__}"
  printf '%s' "${inside%.sh}"
}

safe_name() {
  local raw="$1"
  raw="${raw//[^[:alnum:]._-]/_}"
  printf '%s' "${raw}"
}

extract_train_arg_from_log() {
  local job_log="$1"
  local arg_name="$2"

  "${PYTHON_BIN}" - "$job_log" "$arg_name" <<'PY'
import shlex
import sys

log_path, arg_name = sys.argv[1], sys.argv[2]
flag = f"--{arg_name}"
value = ""

try:
    with open(log_path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if flag not in line:
                continue
            try:
                parts = shlex.split(line)
            except ValueError:
                continue
            for idx, part in enumerate(parts[:-1]):
                if part == flag:
                    value = parts[idx + 1]
except FileNotFoundError:
    pass

print(value)
PY
}

read_args_json_value() {
  local args_json="$1"
  local key="$2"

  "${PYTHON_BIN}" - "$args_json" "$key" <<'PY'
import json
import sys

args_json, key = sys.argv[1], sys.argv[2]
try:
    with open(args_json, "r", encoding="utf-8") as fin:
        value = json.load(fin).get(key, "")
except Exception:
    value = ""

print("" if value is None else value)
PY
}

normalize_model_family() {
  local raw="$1"

  case "${raw}" in
    *[Qq][Ww][Ee][Nn]*|qwen|Qwen)
      printf 'qwen'
      ;;
    *[Ll][Ll][Aa][Mm][Aa]*|llama|Llama)
      printf 'llama'
      ;;
    *)
      printf 'unknown'
      ;;
  esac
}

infer_max_length_for() {
  local model_family="$1"
  local benchmark="$2"

  if [[ "${INFER_MAX_LENGTH}" != "auto" ]]; then
    printf '%s' "${INFER_MAX_LENGTH}"
    return
  fi

  case "${model_family}:${benchmark}" in
    qwen:Cypherbench)
      printf '1034'
      ;;
    qwen:Neo4j_Text2Cypher)
      printf '3092'
      ;;
    qwen:Mind_the_query)
      printf '2427'
      ;;
    llama:Cypherbench)
      printf '1053'
      ;;
    llama:Neo4j_Text2Cypher)
      printf '2884'
      ;;
    llama:Mind_the_query)
      printf '2445'
      ;;
    *)
      printf '1024'
      ;;
  esac
}

abs_or_root_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${ROOT_DIR}" "${path#./}"
  fi
}

latest_checkpoint_dir() {
  local save_path="$1"

  "${PYTHON_BIN}" - "$save_path" <<'PY'
import sys
from pathlib import Path

save_path = Path(sys.argv[1])
files = {
    "adapter_config.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
}
candidates = []

if save_path.is_dir():
    for child in (save_path, *save_path.rglob("*")):
        if not child.is_dir():
            continue
        if any((child / name).exists() for name in files):
            step = int(child.name) if child.name.isdigit() else -1
            candidates.append((step, child.stat().st_mtime, child))

if candidates:
    print(str(max(candidates, key=lambda item: (item[0], item[1]))[2]))
PY
}

run_infer_after_train() {
  local script="$1"
  local gpu_chunk="$2"
  local attempt="$3"
  local job_log="$4"
  local rel_script
  local job_id
  local output_id
  local save_arg
  local save_path
  local args_json
  local model_path
  local model_type
  local model_family
  local ckpt_path
  local ckpt_step
  local script_root
  local output_path
  local output_dir
  local infer_log
  local infer_status
  local infer_max_length
  local overall_status
  local db_name
  local benchmark_name
  local infer_benchmark
  local -a cmd
  local -a extra_args

  rel_script="$(rel_path "${script}")"
  job_id="$(script_id "${script}")"
  output_id="$(safe_name "$(script_experiment_id "${script}")")"

  save_arg="$(extract_train_arg_from_log "${job_log}" "save")"
  if [[ -z "${save_arg}" ]]; then
    echo "[infer-skip] Cannot find --save in train log for ${rel_script}" | tee -a "${job_log}" >&2
    append_log "${INFER_FAIL_LOG}" "missing_save script=${rel_script} train_log=${job_log}"
    return 1
  fi

  save_path="$(abs_or_root_path "${save_arg}")"
  args_json="${save_path}/args.json"

  model_path=""
  if [[ -f "${args_json}" ]]; then
    model_path="$(read_args_json_value "${args_json}" "model_path")"
  fi
  if [[ -z "${model_path}" ]]; then
    model_path="$(extract_train_arg_from_log "${job_log}" "model-path")"
  fi
  if [[ -z "${model_path}" ]]; then
    echo "[infer-skip] Cannot find model path for ${rel_script}" | tee -a "${job_log}" >&2
    append_log "${INFER_FAIL_LOG}" "missing_model script=${rel_script} train_log=${job_log}"
    return 1
  fi

  model_type=""
  if [[ -f "${args_json}" ]]; then
    model_type="$(read_args_json_value "${args_json}" "model_type")"
  fi
  if [[ -z "${model_type}" ]]; then
    model_type="$(extract_train_arg_from_log "${job_log}" "model-type")"
  fi
  model_family="$(normalize_model_family "${model_type}")"
  if [[ "${model_family}" == "unknown" ]]; then
    model_family="$(normalize_model_family "${model_path}")"
  fi

  ckpt_path="$(latest_checkpoint_dir "${save_path}")"
  if [[ -z "${ckpt_path}" ]]; then
    echo "[infer-skip] No numeric checkpoint directory found under ${save_path}" | tee -a "${job_log}" >&2
    append_log "${INFER_FAIL_LOG}" "missing_ckpt script=${rel_script} save=${save_path} train_log=${job_log}"
    return 1
  fi

  ckpt_step="$(basename "${ckpt_path}")"
  db_name="$(safe_name "${INFER_DB}")"
  script_root="$(safe_name "$(script_root_name "${script}")")"
  overall_status=0

  for infer_benchmark in "${INFER_BENCHMARK_LIST[@]}"; do
    infer_max_length="$(infer_max_length_for "${model_family}" "${infer_benchmark}")"
    benchmark_name="$(safe_name "${infer_benchmark}")"
    output_dir="${INFER_OUTPUT_ROOT}/${script_root}/${benchmark_name}"
    output_path="${output_dir}/${output_id}__ckpt${ckpt_step}__${db_name}_cyphers_result.json"
    infer_log="${INFER_LOG_DIR}/${job_id}__ckpt${ckpt_step}__${benchmark_name}__${db_name}.log"
    mkdir -p "${output_dir}"

    echo "[infer] ${rel_script}"
    echo "        benchmark: ${infer_benchmark}"
    echo "        model    : ${model_family}"
    echo "        max len  : ${infer_max_length}"
    echo "        ckpt     : ${ckpt_path}"
    echo "        out      : ${output_path}"
    append_log "${RUN_LOG}" "INFER_START attempt=${attempt} gpus=${gpu_chunk} script=${rel_script} benchmark=${infer_benchmark} model_family=${model_family} max_length=${infer_max_length} ckpt=${ckpt_path} output=${output_path} log=${infer_log}"

    cmd=(
      "${PYTHON_BIN}" "${INFER_SCRIPT}"
      --benchmark "${infer_benchmark}"
      --db "${INFER_DB}"
      --data_source "${INFER_DATA_SOURCE}"
      --model "${model_path}"
      --ckpt_path "${ckpt_path}"
      --device cuda
      --batch-size "${INFER_BATCH_SIZE}"
      --max-length "${infer_max_length}"
      --output_path "${output_path}"
    )
    if [[ -n "${INFER_LIMIT}" ]]; then
      cmd+=(--limit "${INFER_LIMIT}")
    fi
    if [[ -n "${INFER_EXTRA_ARGS}" ]]; then
      # Intentional word-splitting: this option is a raw convenience escape hatch.
      extra_args=( ${INFER_EXTRA_ARGS} )
      cmd+=("${extra_args[@]}")
    fi

    if {
      echo "${cmd[*]}"
      CUDA_VISIBLE_DEVICES="${gpu_chunk}" PYTHONPATH="${ROOT_DIR}" "${cmd[@]}"
    } > "${infer_log}" 2>&1; then
      infer_status=0
    else
      infer_status=$?
    fi

    if [[ "${infer_status}" -eq 0 ]]; then
      echo "[infer-done] ${rel_script} ${infer_benchmark} -> ${output_path}"
      append_log "${RUN_LOG}" "INFER_DONE attempt=${attempt} gpus=${gpu_chunk} script=${rel_script} benchmark=${infer_benchmark} model_family=${model_family} max_length=${infer_max_length} ckpt=${ckpt_path} output=${output_path} log=${infer_log}"
      append_log "${INFER_SUCCESS_LOG}" "attempt=${attempt} gpus=${gpu_chunk} script=${rel_script} benchmark=${infer_benchmark} model_family=${model_family} max_length=${infer_max_length} ckpt=${ckpt_path} output=${output_path} log=${infer_log}"
    else
      echo "[infer-fail] ${rel_script} ${infer_benchmark} exited with ${infer_status}" >&2
      append_log "${RUN_LOG}" "INFER_FAIL attempt=${attempt} exit_code=${infer_status} gpus=${gpu_chunk} script=${rel_script} benchmark=${infer_benchmark} model_family=${model_family} max_length=${infer_max_length} ckpt=${ckpt_path} output=${output_path} log=${infer_log}"
      append_log "${INFER_FAIL_LOG}" "attempt=${attempt} exit_code=${infer_status} script=${rel_script} benchmark=${infer_benchmark} model_family=${model_family} max_length=${infer_max_length} ckpt=${ckpt_path} output=${output_path} log=${infer_log}"
      overall_status="${infer_status}"
    fi
  done

  return "${overall_status}"
}

IFS=', ' read -r -a ALL_GPUS <<< "${GPU_LIST}"
GPU_COUNT="${#ALL_GPUS[@]}"
if [[ "${GPU_COUNT}" -lt "${GPUS_PER_JOB}" ]]; then
  echo "Need at least ${GPUS_PER_JOB} GPUs, but got ${GPU_COUNT}: ${GPU_LIST}" >&2
  exit 1
fi

chunks=()
for ((i=0; i<GPU_COUNT; i+=GPUS_PER_JOB)); do
  if (( i + GPUS_PER_JOB <= GPU_COUNT )); then
    chunk="$(IFS=,; echo "${ALL_GPUS[*]:i:GPUS_PER_JOB}")"
    chunks+=("${chunk}")
  fi
done

if [[ "${#chunks[@]}" -eq 0 ]]; then
  echo "Failed to form GPU chunks from ${GPU_LIST}" >&2
  exit 1
fi

if (( GPU_COUNT % GPUS_PER_JOB != 0 )); then
  leftover=$((GPU_COUNT % GPUS_PER_JOB))
  echo "Warning: ${leftover} GPU(s) will be idle because ${GPU_COUNT} is not divisible by ${GPUS_PER_JOB}." >&2
  append_log "${RUN_LOG}" "Warning: ${leftover} GPU(s) idle because ${GPU_COUNT} is not divisible by ${GPUS_PER_JOB}."
fi

mapfile -t SCRIPTS < <(find "${SCRIPT_ROOT}" -type f -name "*.sh" | sort)
if [[ -n "${FILTER}" ]]; then
  FILTERED=()
  for script in "${SCRIPTS[@]}"; do
    if [[ "$(rel_path "${script}")" == *"${FILTER}"* ]]; then
      FILTERED+=("${script}")
    fi
  done
  SCRIPTS=("${FILTERED[@]}")
fi

if [[ "${#SCRIPTS[@]}" -eq 0 ]]; then
  echo "No scripts found under ${SCRIPT_ROOT} matching filter '${FILTER}'" >&2
  exit 1
fi

append_log "${RUN_LOG}" "Started run: mode=${MODE}, gpus=${GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, max_retries=${MAX_RETRIES}, continue_on_error=${CONTINUE_ON_ERROR}, filter=${FILTER:-<none>}"
append_log "${RUN_LOG}" "Script root: ${SCRIPT_ROOT}"
append_log "${RUN_LOG}" "Logs directory: ${LOG_DIR}"
append_log "${RUN_LOG}" "Scheduled scripts: ${#SCRIPTS[@]}"
if [[ "${INFER_AFTER_TRAIN}" -eq 1 ]]; then
  append_log "${RUN_LOG}" "Infer after train: script=${INFER_SCRIPT}, benchmarks=${INFER_BENCHMARK_DISPLAY}, db=${INFER_DB}, data_source=${INFER_DATA_SOURCE}, batch_size=${INFER_BATCH_SIZE}, max_length=${INFER_MAX_LENGTH}, limit=${INFER_LIMIT:-<none>}, output_root=${INFER_OUTPUT_ROOT}"
fi

failures=0

random_master_port() {
  # Keep ports in unprivileged range and away from very low ephemeral values.
  echo $((10000 + (RANDOM * 32768 + RANDOM) % 50000))
}

run_script_once() {
  local script="$1"
  local gpu_chunk="$2"
  local attempt="$3"
  local port
  local rel_script
  local job_id
  local job_log

  rel_script="$(rel_path "${script}")"
  job_id="$(script_id "${script}")"
  job_log="${JOB_LOG_DIR}/${job_id}.attempt${attempt}.log"
  port="$(random_master_port)"

  echo "[launch] ${rel_script}"
  echo "         GPUs: ${gpu_chunk} | port: ${port} | attempt: ${attempt}"
  echo "         log : ${job_log}"
  append_log "${RUN_LOG}" "DISPATCH attempt=${attempt} gpus=${gpu_chunk} port=${port} script=${rel_script} log=${job_log}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${INFER_AFTER_TRAIN}" -eq 1 ]]; then
      echo "         infer after train: ${INFER_SCRIPT} (${INFER_BENCHMARK_DISPLAY}/${INFER_DB}) -> ${INFER_OUTPUT_ROOT}/<script_root>/<benchmark>/"
    fi
    return 0
  fi

  RUN_GPUS="${gpu_chunk}" RUN_MASTER_PORT="${port}" bash "${script}" > "${job_log}" 2>&1
  local train_status=$?
  if [[ "${train_status}" -ne 0 ]]; then
    return "${train_status}"
  fi

  if [[ "${INFER_AFTER_TRAIN}" -eq 1 ]]; then
    run_infer_after_train "${script}" "${gpu_chunk}" "${attempt}" "${job_log}"
  fi
}

if [[ "${MODE}" == "sequential" ]]; then
  gpu_chunk="${chunks[0]}"
  for script in "${SCRIPTS[@]}"; do
    rel_script="$(rel_path "${script}")"
    attempt=1
    while true; do
      if run_script_once "${script}" "${gpu_chunk}" "${attempt}"; then
        append_log "${RUN_LOG}" "DONE attempt=${attempt} gpus=${gpu_chunk} script=${rel_script}"
        append_log "${SUCCESS_LOG}" "attempt=${attempt} gpus=${gpu_chunk} script=${rel_script}"
        break
      fi

      append_log "${RUN_LOG}" "FAIL attempt=${attempt} gpus=${gpu_chunk} script=${rel_script}"
      if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
        attempt=$((attempt + 1))
        append_log "${RETRY_LOG}" "next_attempt=${attempt} script=${rel_script}"
        continue
      fi

      failures=$((failures + 1))
      append_log "${FAIL_LOG}" "attempt=${attempt} gpus=${gpu_chunk} script=${rel_script}"
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        append_log "${RUN_LOG}" "Stopped sequential mode after exhausted retries for ${rel_script}."
        echo "Stopping after failure: ${rel_script}" >&2
        exit 1
      fi
      break
    done
  done
else
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    for idx in "${!SCRIPTS[@]}"; do
      chunk_idx=$((idx % ${#chunks[@]}))
      run_script_once "${SCRIPTS[$idx]}" "${chunks[$chunk_idx]}" 1
    done
  else
    declare -a JOB_ATTEMPTS=()
    declare -a PENDING_RUNS=()
    declare -a ACTIVE_PIDS=()
    declare -a CHUNK_BUSY=()
    declare -A PID_TO_RUN_IDX=()
    declare -A PID_TO_CHUNK_IDX=()
    declare -A PID_TO_ATTEMPT=()
    declare -A PID_TO_LOG=()

    for idx in "${!SCRIPTS[@]}"; do
      JOB_ATTEMPTS[idx]=0
      PENDING_RUNS+=("${idx}")
    done

    for idx in "${!chunks[@]}"; do
      CHUNK_BUSY[idx]=0
    done

    queue_head=0
    stop_scheduling=0

    remove_active_pid() {
      local target_pid="$1"
      local next_active=()
      local active_pid

      for active_pid in "${ACTIVE_PIDS[@]}"; do
        if [[ "${active_pid}" != "${target_pid}" ]]; then
          next_active+=("${active_pid}")
        fi
      done
      ACTIVE_PIDS=("${next_active[@]}")
    }

    start_job_on_chunk() {
      local run_idx="$1"
      local chunk_idx="$2"
      local script="${SCRIPTS[$run_idx]}"
      local attempt=$((JOB_ATTEMPTS[run_idx] + 1))
      local port
      local rel_script
      local job_id
      local job_log
      local pid

      JOB_ATTEMPTS[run_idx]="${attempt}"
      port="$(random_master_port)"
      rel_script="$(rel_path "${script}")"
      job_id="$(script_id "${script}")"
      job_log="${JOB_LOG_DIR}/${job_id}.attempt${attempt}.log"

      echo "[queue] dispatch ${rel_script}"
      echo "        GPUs: ${chunks[$chunk_idx]} | port: ${port} | attempt: ${attempt}"
      echo "        log : ${job_log}"
      append_log "${RUN_LOG}" "DISPATCH attempt=${attempt} gpus=${chunks[$chunk_idx]} port=${port} script=${rel_script} log=${job_log}"

      (
        set +e
        RUN_GPUS="${chunks[$chunk_idx]}" RUN_MASTER_PORT="${port}" bash "${script}" > "${job_log}" 2>&1
        train_status=$?
        if [[ "${train_status}" -ne 0 ]]; then
          exit "${train_status}"
        fi
        if [[ "${INFER_AFTER_TRAIN}" -eq 1 ]]; then
          run_infer_after_train "${script}" "${chunks[$chunk_idx]}" "${attempt}" "${job_log}"
          exit "$?"
        fi
        exit 0
      ) &
      pid="$!"

      ACTIVE_PIDS+=("${pid}")
      CHUNK_BUSY[chunk_idx]=1
      PID_TO_RUN_IDX["${pid}"]="${run_idx}"
      PID_TO_CHUNK_IDX["${pid}"]="${chunk_idx}"
      PID_TO_ATTEMPT["${pid}"]="${attempt}"
      PID_TO_LOG["${pid}"]="${job_log}"
    }

    schedule_ready_jobs() {
      local chunk_idx
      while [[ "${stop_scheduling}" -eq 0 ]]; do
        chunk_idx=""
        for idx in "${!chunks[@]}"; do
          if [[ "${CHUNK_BUSY[idx]}" -eq 0 ]]; then
            chunk_idx="${idx}"
            break
          fi
        done

        if [[ -z "${chunk_idx}" || "${queue_head}" -ge "${#PENDING_RUNS[@]}" ]]; then
          break
        fi

        start_job_on_chunk "${PENDING_RUNS[$queue_head]}" "${chunk_idx}"
        queue_head=$((queue_head + 1))
      done
    }

    schedule_ready_jobs

    while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
      finished_pid=""
      if wait -n -p finished_pid; then
        wait_status=0
      else
        wait_status=$?
      fi

      run_idx="${PID_TO_RUN_IDX[${finished_pid}]}"
      chunk_idx="${PID_TO_CHUNK_IDX[${finished_pid}]}"
      attempt="${PID_TO_ATTEMPT[${finished_pid}]}"
      script="${SCRIPTS[$run_idx]}"
      rel_script="$(rel_path "${script}")"
      job_log="${PID_TO_LOG[${finished_pid}]}"

      CHUNK_BUSY[chunk_idx]=0
      remove_active_pid "${finished_pid}"
      unset "PID_TO_RUN_IDX[$finished_pid]"
      unset "PID_TO_CHUNK_IDX[$finished_pid]"
      unset "PID_TO_ATTEMPT[$finished_pid]"
      unset "PID_TO_LOG[$finished_pid]"

      if [[ "${wait_status}" -eq 0 ]]; then
        echo "[done] ${rel_script}"
        append_log "${RUN_LOG}" "DONE attempt=${attempt} gpus=${chunks[$chunk_idx]} script=${rel_script} log=${job_log}"
        append_log "${SUCCESS_LOG}" "attempt=${attempt} gpus=${chunks[$chunk_idx]} script=${rel_script} log=${job_log}"
      else
        echo "[fail] ${rel_script} exited with ${wait_status}" >&2
        append_log "${RUN_LOG}" "FAIL attempt=${attempt} exit_code=${wait_status} gpus=${chunks[$chunk_idx]} script=${rel_script} log=${job_log}"
        if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
          next_attempt=$((attempt + 1))
          append_log "${RETRY_LOG}" "next_attempt=${next_attempt} script=${rel_script}"
          PENDING_RUNS+=("${run_idx}")
        else
          failures=$((failures + 1))
          append_log "${FAIL_LOG}" "attempt=${attempt} exit_code=${wait_status} script=${rel_script} log=${job_log}"
          if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
            stop_scheduling=1
            append_log "${RUN_LOG}" "Stopped scheduling after exhausted retries for ${rel_script}."
          fi
        fi
      fi

      schedule_ready_jobs
    done

    if [[ "${stop_scheduling}" -eq 1 && "${queue_head}" -lt "${#PENDING_RUNS[@]}" ]]; then
      skipped_jobs=$(( ${#PENDING_RUNS[@]} - queue_head ))
      append_log "${RUN_LOG}" "Stopped with ${skipped_jobs} queued script(s) not started."
      echo "Stopped with ${skipped_jobs} queued script(s) not started." >&2
    fi
  fi
fi

if [[ "${failures}" -gt 0 ]]; then
  append_log "${RUN_LOG}" "Finished with ${failures} failed script(s)."
  echo "Finished with ${failures} failed script(s)." >&2
  exit 1
fi

UPLOAD_SCRIPT="${ROOT_DIR}/upload_to_hf.py"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would upload results with ${UPLOAD_SCRIPT}"
  append_log "${RUN_LOG}" "DRY_RUN_UPLOAD script=${UPLOAD_SCRIPT}"
elif [[ -f "${UPLOAD_SCRIPT}" ]]; then
  echo "[upload] Uploading results to Hugging Face..."
  append_log "${RUN_LOG}" "UPLOAD_START script=${UPLOAD_SCRIPT}"
  "${PYTHON_BIN}" "${UPLOAD_SCRIPT}"
  append_log "${RUN_LOG}" "UPLOAD_DONE script=${UPLOAD_SCRIPT}"
  echo "[upload-done] Results uploaded to Hugging Face."
else
  echo "Upload script does not exist: ${UPLOAD_SCRIPT}" >&2
  append_log "${RUN_LOG}" "UPLOAD_FAIL missing_script=${UPLOAD_SCRIPT}"
  exit 1
fi

append_log "${RUN_LOG}" "All requested scripts finished successfully."
echo "All requested scripts finished successfully."
