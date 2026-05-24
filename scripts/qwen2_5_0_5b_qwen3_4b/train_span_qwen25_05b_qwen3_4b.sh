#! /usr/bin/env bash

set -euo pipefail

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0)
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

BASE_PATH="${BASE_PATH:-.}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${RUN_MASTER_PORT:-66$(($RANDOM % 90 + 10))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${#GPUS[@]}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

STUDENT_CKPT="${STUDENT_CKPT:-Qwen/Qwen2.5-0.5B}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen3-4B-Instruct-2507}"
STUDENT_CKPT_NAME="${STUDENT_CKPT_NAME:-qwen2.5-0.5B}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen3-4B-Instruct-2507}"

# This can be a local processed-data directory or an hf:// path resolved by the dataset loader.
DATA_DIR="${DATA_DIR:-hf://fisherman611/text-to-cypher-processed-data/Cypherbench/qwen}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/qwen2.5/span_qwen25_05b_student_qwen3_4b_teacher}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-1}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-892}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-797}"
KD_RATIO="${KD_RATIO:-0.6}"
W_SPAN_LOSS="${W_SPAN_LOSS:-0.7}"
DISTIL_TYPE="${DISTIL_TYPE:-adaptive-srkl}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

STUDENT_LAYER_MAPPING="${STUDENT_LAYER_MAPPING:--1}"
TEACHER_LAYER_MAPPING="${TEACHER_LAYER_MAPPING:--1}"

PEFT_LORA_R="${PEFT_LORA_R:-32}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

OPTS=(
  --base-path "${BASE_PATH}"
  --model-path "${STUDENT_CKPT}"
  --teacher-model-path "${TEACHER_CKPT}"
  --ckpt-name "${STUDENT_CKPT_NAME}"
  --teacher-ckpt-name "${TEACHER_CKPT_NAME}"
  --model-type qwen
  --teacher-model-type qwen
  --n-gpu "${GPUS_PER_NODE}"
  --data-dir "${DATA_DIR}"
  --num-workers 1
  --dev-num -1
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACC}"
  --warmup-iters 0
  --lr-decay-style cosine
  --weight-decay 1e-2
  --clip-grad 1.0
  --epochs "${EPOCHS}"
  --kd-ratio "${KD_RATIO}"
  --w-span-loss "${W_SPAN_LOSS}"
  --student_layer_mapping ${STUDENT_LAYER_MAPPING}
  --teacher_layer_mapping ${TEACHER_LAYER_MAPPING}
  --max-length "${MAX_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --do-train
  --do-valid
  --eval-gen
  --save-interval -1
  --eval-interval -1
  --log-interval "${LOG_INTERVAL}"
  --mid-log-num -1
  --save "${SAVE_PATH}"
  --seed "${SEED}"
  --deepspeed
  --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_fp16.json"
  --type "${DISTIL_TYPE}"
  --do-sample
  --top-k 0
  --top-p 0.95
  --temperature 0.5
  --student-gen
  --gen-num-beams 1
  --gen-top-p 1.0
  --init-threshold 0.0
  --loss-eps 0.1
  --capacity 1000
  --peft lora
  --peft-lora-r "${PEFT_LORA_R}"
  --peft-lora-alpha "${PEFT_LORA_ALPHA}"
  --peft-lora-dropout "${PEFT_LORA_DROPOUT}"
)

if [[ -n "${TEACHER_PEFT_PATH:-}" ]]; then
  OPTS+=(--teacher-peft-path "${TEACHER_PEFT_PATH}")
fi

export NCCL_DEBUG="${NCCL_DEBUG:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONPATH="${BASE_PATH}"

mkdir -p "${SAVE_PATH}"

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${BASE_PATH}/finetuning/span_finetune.py" "${OPTS[@]}" "$@")

echo "${CMD[*]}"
echo "PYTHONPATH=${PYTHONPATH}"
CODE_BASE=HF "${CMD[@]}"

