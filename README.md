# distillation

## Lệnh dùng nhanh

```bash
# Login HF
hf auth login

uv sync
source .venv/bin/activate
bash running.sh \
  --filter updated_span_question_schema_2_update_span_weight \
  --gpus 0 \
  --gpus-per-job 1 \
  --infer-after-train \
  --infer-db full \
  --infer-batch-size 32
```