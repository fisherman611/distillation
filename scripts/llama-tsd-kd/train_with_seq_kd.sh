beta=0.9
lmbda=1.0
threshold=0.1
model_name=meta-llama/Llama-3.2-1B-Instruct
indirect_kd_alpha=0.1
teacher_model_name=meta-llama/Meta-Llama-3-8B-Instruct
output_dir="${OUTPUT_DIR:-results/tsd-kd/llama/with_seq_kd}"

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
  --seq-kd
