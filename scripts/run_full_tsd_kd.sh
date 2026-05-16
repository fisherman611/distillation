#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-sequential}"
VARIANTS="${VARIANTS:-no_seq}"
QWEN_GPUS="${QWEN_GPUS:-${RUN_GPUS:-0}}"
LLAMA_GPUS="${LLAMA_GPUS:-${RUN_GPUS:-0}}"
LOG_ROOT="${LOG_ROOT:-run_logs/tsd-kd-full/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_tsd_kd.sh

Environment:
  MODE=sequential|parallel       Default: sequential
  VARIANTS=no_seq|with_seq|both  Default: no_seq
  RUN_GPUS=0                    Default GPU list for both jobs
  QWEN_GPUS=0                   GPU list for Qwen job
  LLAMA_GPUS=1                  GPU list for Llama job
  LOG_ROOT=run_logs/...         Log directory

Examples:
  RUN_GPUS=0 bash scripts/run_full_tsd_kd.sh
  VARIANTS=both RUN_GPUS=0 bash scripts/run_full_tsd_kd.sh
  MODE=parallel QWEN_GPUS=0 LLAMA_GPUS=1 bash scripts/run_full_tsd_kd.sh
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
    variant_scripts=(train_no_seq_kd.sh)
    ;;
  with_seq)
    variant_scripts=(train_with_seq_kd.sh)
    ;;
  both)
    variant_scripts=(train_no_seq_kd.sh train_with_seq_kd.sh)
    ;;
  *)
    echo "VARIANTS must be one of: no_seq, with_seq, both" >&2
    exit 1
    ;;
esac

run_job() {
  local name="$1"
  local gpus="$2"
  local script="$3"
  local log_file="$LOG_ROOT/${name}__${script%.sh}.log"

  echo "[start] ${name}/${script} GPUs=${gpus}"
  echo "        log=${log_file}"
  RUN_GPUS="$gpus" RUN_MASTER_PORT="$((10000 + RANDOM % 50000))" bash "scripts/${name}-tsd-kd/${script}" \
    > "$log_file" 2>&1
  echo "[done] ${name}/${script}"
}

if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
  echo "MODE must be sequential or parallel" >&2
  exit 1
fi

if [[ "$MODE" == "parallel" ]]; then
  pids=()
  for script in "${variant_scripts[@]}"; do
    run_job qwen "$QWEN_GPUS" "$script" &
    pids+=("$!")
    run_job llama "$LLAMA_GPUS" "$script" &
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

for script in "${variant_scripts[@]}"; do
  run_job qwen "$QWEN_GPUS" "$script"
  run_job llama "$LLAMA_GPUS" "$script"
done

echo "All full TSD-KD jobs finished. Logs: ${LOG_ROOT}"
