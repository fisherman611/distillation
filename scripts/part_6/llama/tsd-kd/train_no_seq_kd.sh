#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tsd-kd/llama/no_seq_kd}"
if [[ -n "${DATA_DIR:-}" && -z "${DATASET_ROOT:-}" ]]; then
  export DATASET_ROOT="${DATA_DIR}"
fi
export OUTPUT_DIR

# Emit normalized metadata so running.sh can parse --model-path/--save for post-train infer.
echo "tsd_kd_meta --model-path ${MODEL_PATH} --save ${OUTPUT_DIR}"

exec bash "${REPO_ROOT}/scripts/llama-tsd-kd/train_no_seq_kd.sh" "$@"
