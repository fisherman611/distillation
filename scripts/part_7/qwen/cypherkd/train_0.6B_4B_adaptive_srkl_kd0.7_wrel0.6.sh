#! /bin/bash

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0 1)
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${RUN_MASTER_PORT:-66$(($RANDOM%90+10))}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=.
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
SCRIPT_GROUP="$(basename "$(dirname "${BASH_SOURCE[0]}")")"
SAVE_TAG="updated_span_question_schema_2_update_span_weight_${SCRIPT_GROUP}_${SCRIPT_NAME}"
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
TEACHER_CKPT_NAME="qwen3-4B"
TEACHER_CKPT="Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR="hf://fisherman611/text-to-cypher-processed-data/Cypherbench/qwen"
BATCH_SIZE=8
LR=0.0001
GRAD_ACC=1
EVAL_BATCH_SIZE=16
EPOCHS=5
MAX_LENGTH=892
MAX_PROMPT_LENGTH=797
SAVE_PATH="${BASE_PATH}/results/qwen3/${SAVE_TAG}"
SEED=42
W_REL_LOSS=0.6
KD_RATIO=0.7
STUDENT_LAYER_MAPPING=(-1)
TEACHER_LAYER_MAPPING=(-1)

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --teacher-peft-path hf://fisherman611/text-to-cypher-models/e5-bs2-lr1e-05-G8-N2-NN1-lora-32-64-0.1/1065"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num -1"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio ${KD_RATIO}"
OPTS+=" --w-rel-loss ${W_REL_LOSS}"
OPTS+=" --student_layer_mapping ${STUDENT_LAYER_MAPPING[*]}"
OPTS+=" --teacher_layer_mapping ${TEACHER_LAYER_MAPPING[*]}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp16.json"
OPTS+=" --type adaptive-srkl"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 32"
OPTS+=" --peft-lora-alpha 64"
OPTS+=" --peft-lora-dropout 0.1"
OPTS+=" --no-span-length-weight"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetuning/updated_span_finetune_question_schema_2_update_span_weight.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}

