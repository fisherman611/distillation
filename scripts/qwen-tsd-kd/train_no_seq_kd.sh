beta=0.9
lmbda=1.0
threshold=0.1
model_name=Qwen/Qwen3-0.6B
indirect_kd_alpha=0.1
teacher_model_name=Qwen/Qwen3-4B-Instruct-2507

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

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
  --teacher-model-name $teacher_model_name
