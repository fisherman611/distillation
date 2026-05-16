beta=0.9
lambda=1.0
threshold=0.1
model_name=Qwen/Qwen3-0.6B
indirect_kd_alpha=0.1
teacher_model_name=Qwen/Qwen3-4B-Instruct-2507

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

accelerate launch --config_file TSD-KD/accelerate_ddp_config.yaml TSD-KD/train.py \
  --beta $beta \
  --lmbda $lambda \
  --threshold $threshold \
  --model-name $model_name \
  --indirect-kd-alpha $indirect_kd_alpha \
  --teacher-model-name $teacher_model_name
