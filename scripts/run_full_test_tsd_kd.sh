#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-sequential}"
VARIANTS="${VARIANTS:-no_seq}"
QWEN_GPUS="${QWEN_GPUS:-${RUN_GPUS:-0}}"
LLAMA_GPUS="${LLAMA_GPUS:-${RUN_GPUS:-0}}"
LOG_ROOT="${LOG_ROOT:-run_logs/tsd-kd-test/$(date +%Y%m%d_%H%M%S)}"
TEST_SCRIPT_ROOT="${TEST_SCRIPT_ROOT:-test_scripts}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-80}"

mkdir -p "$LOG_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_test_tsd_kd.sh

Environment:
  MODE=sequential|parallel       Default: sequential
  VARIANTS=no_seq|with_seq|both  Default: no_seq
  RUN_GPUS=0                    Default GPU list for both test jobs
  QWEN_GPUS=0                   GPU list for Qwen test job
  LLAMA_GPUS=1                  GPU list for Llama test job
  MAX_TRAIN_SAMPLES=32          Smoke train samples
  MAX_EVAL_SAMPLES=8            Smoke eval samples
  MAX_STEPS=5                   Smoke train steps
  LOG_ROOT=run_logs/...         Log directory
  LOG_TAIL_LINES=80             Lines to print from a failed job log
  TEST_SCRIPT_ROOT=test_scripts Directory containing qwen-tsd-kd/test_finetune.sh and llama-tsd-kd/test_finetune.sh

Examples:
  RUN_GPUS=0 bash scripts/run_full_test_tsd_kd.sh
  VARIANTS=both RUN_GPUS=0 bash scripts/run_full_test_tsd_kd.sh
  MODE=parallel QWEN_GPUS=0 LLAMA_GPUS=1 bash scripts/run_full_test_tsd_kd.sh
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

case "$VARIANTS" in
  no_seq)
    seq_values=(0)
    ;;
  with_seq)
    seq_values=(1)
    ;;
  both)
    seq_values=(0 1)
    ;;
  *)
    echo "VARIANTS must be one of: no_seq, with_seq, both" >&2
    exit 1
    ;;
esac

if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
  echo "MODE must be sequential or parallel" >&2
  exit 1
fi

test_script_for() {
  local name="$1"
  local path="${TEST_SCRIPT_ROOT}/${name}-tsd-kd/test_finetune.sh"
  if [[ ! -f "$path" ]]; then
    echo "Missing test script: ${path}" >&2
    exit 1
  fi
  printf '%s' "$path"
}

variant_name() {
  if [[ "$1" == "1" ]]; then
    printf 'with_seq_kd'
  else
    printf 'no_seq_kd'
  fi
}

run_job() {
  local name="$1"
  local gpus="$2"
  local seq_value="$3"
  local script
  local variant
  local log_file
  local output_dir

  script="$(test_script_for "$name")"
  variant="$(variant_name "$seq_value")"
  log_file="$LOG_ROOT/${name}__${variant}.log"
  output_dir="${OUTPUT_ROOT:-results/tsd-kd-smoke}/${name}/${variant}"

  echo "[start] ${name}/${variant} GPUs=${gpus}"
  echo "        script=${script}"
  echo "        log=${log_file}"
  local status=0
  RUN_GPUS="$gpus" \
    RUN_MASTER_PORT="$((10000 + RANDOM % 50000))" \
    SEQ_KD="$seq_value" \
    OUTPUT_DIR="$output_dir" \
    bash "$script" > "$log_file" 2>&1 || status="$?"

  if [[ "$status" -eq 0 ]]; then
    echo "[done] ${name}/${variant}"
    return 0
  fi

  echo "[fail] ${name}/${variant} exit=${status}"
  echo "        log=${log_file}"
  if [[ -s "$log_file" ]]; then
    echo "-------- last ${LOG_TAIL_LINES} log lines --------"
    tail -n "$LOG_TAIL_LINES" "$log_file"
    echo "----------------------------------------"
  else
    echo "        log file is empty"
  fi
  return "$status"
}

if [[ "$MODE" == "parallel" ]]; then
  pids=()
  for seq_value in "${seq_values[@]}"; do
    run_job qwen "$QWEN_GPUS" "$seq_value" &
    pids+=("$!")
    run_job llama "$LLAMA_GPUS" "$seq_value" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  exit "$status"
fi

for seq_value in "${seq_values[@]}"; do
  run_job qwen "$QWEN_GPUS" "$seq_value"
  run_job llama "$LLAMA_GPUS" "$seq_value"
done

echo "All TSD-KD smoke tests finished. Logs: ${LOG_ROOT}"
