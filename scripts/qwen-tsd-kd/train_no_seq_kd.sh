beta=0.9
lmbda=1.0
threshold=0.1
model_name=Qwen/Qwen3-0.6B
indirect_kd_alpha=0.1
teacher_model_name=Qwen/Qwen3-4B-Instruct-2507
teacher_peft_path="${TEACHER_PEFT_PATH:-hf://fisherman611/text-to-cypher-models/e5-bs2-lr1e-05-G8-N2-NN1-lora-32-64-0.1/1065}"
output_dir="${OUTPUT_DIR:-results/tsd-kd/qwen/no_seq_kd}"
train_batch_size="${TRAIN_BATCH_SIZE:-8}"
eval_batch_size="${EVAL_BATCH_SIZE:-16}"
grad_acc="${GRAD_ACC:-2}"

if [[ -n "${RUN_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RUN_GPUS}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS_PER_NODE="${#GPUS[@]}"
else
  GPUS_PER_NODE=1
fi
MASTER_PORT="${RUN_MASTER_PORT:-29500}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "Missing dependency: accelerate is not installed or not on PATH." >&2
  echo "Install project dependencies in the active environment: bash install.sh" >&2
  echo "Or minimally install accelerate: python -m pip install accelerate" >&2
  exit 127
fi

extra_args=()
if [[ -n "${teacher_peft_path}" ]]; then
  extra_args+=(--teacher-peft-path "$teacher_peft_path")
fi

accelerate launch \
  --config_file TSD-KD/accelerate_ddp_config.yaml \
  --num_processes "${GPUS_PER_NODE}" \
  --main_process_port "${MASTER_PORT}" \
  TSD-KD/train.py \
  --beta $beta \
  --lmbda $lmbda \
  --threshold $threshold \
  --model-name $model_name \
  --indirect-kd-alpha $indirect_kd_alpha \
  --teacher-model-name $teacher_model_name \
  --output-dir "$output_dir" \
  --per-device-train-batch-size "$train_batch_size" \
  --per-device-eval-batch-size "$eval_batch_size" \
  --gradient-accumulation-steps "$grad_acc" \
  "${extra_args[@]}"
