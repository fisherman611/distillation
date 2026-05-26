#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DISTILL_TYPE="${DISTILL_TYPE:-adaptive-srkl}"
export SAVE_PATH="${SAVE_PATH:-./results/qwen3/token_rel_loss_0.6B_4B_Cypherbench_adaptive_srkl_kd0.6}"

bash "${SCRIPT_DIR}/train_0.6B_4B_csd_kd0.6.sh" "$@"
