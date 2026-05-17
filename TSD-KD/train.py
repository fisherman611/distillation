from datasets import load_dataset
from trl import GKDConfig
from DistillTrainer import DistillTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import torch, os
import torch.distributed as dist
from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSD-KD on Cypherbench.")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--model-name", dest="model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--indirect-kd-alpha", dest="indirect_kd_alpha", type=float, default=0.1)
    parser.add_argument("--seq-kd", dest="seq_kd", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="tsd-kd-Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-train-samples", dest="max_train_samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", dest="max_eval_samples", type=int, default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", dest="per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", dest="per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int, default=10)
    parser.add_argument("--no-load-best-model-at-end", dest="load_best_model_at_end", action="store_false")
    parser.set_defaults(load_best_model_at_end=True)
    parser.add_argument(
        "--teacher-model-name",
        dest="teacher_model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "--teacher-peft-path",
        dest="teacher_peft_path",
        type=str,
        default=None,
        help="Optional LoRA path for teacher model. Supports local path, HF repo id, or hf://<owner>/<repo>/<subfolder>.",
    )
    return parser.parse_args()


args_cli = parse_args()
beta = args_cli.beta
lmbda = args_cli.lmbda
threshold = args_cli.threshold
model_name = args_cli.model_name
indirect_kd_alpha = args_cli.indirect_kd_alpha

import torch._dynamo
torch._dynamo.config.disable = True

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def resolve_peft_source(source: str) -> str:
    raw = str(source).strip()
    if os.path.isdir(raw):
        return raw

    normalized = raw.rstrip("/")
    if normalized.startswith("hf://"):
        content = normalized[len("hf://"):].strip("/")
        parts = [p for p in content.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hf:// path '{source}'. Expected hf://<owner>/<repo>/<optional/subfolder>"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
        if not subfolder:
            return repo_id

        token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
            token=token,
        )
        return os.path.join(snapshot_dir, subfolder)

    return raw

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

teacher_model_name = args_cli.teacher_model_name
attn = "sdpa"
# The model to optimise
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.pad_token_id, trust_remote_code=True)#.to(f"cuda:{local_rank}")

# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, attn_implementation=attn, torch_dtype=torch.bfloat16, trust_remote_code=True)#.to(f"cuda:{local_rank}")
if args_cli.teacher_peft_path:
    from peft import PeftModel

    resolved_teacher_peft_path = resolve_peft_source(args_cli.teacher_peft_path)
    if is_main_process():
        print(f"Loading teacher LoRA from: {resolved_teacher_peft_path}")
    teacher_model = PeftModel.from_pretrained(teacher_model, resolved_teacher_peft_path)
    teacher_model = teacher_model.merge_and_unload()

def align_model_special_tokens(model, tokenizer):
    for config in (model.config, getattr(model, "generation_config", None)):
        if config is None:
            continue
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id

align_model_special_tokens(model, tokenizer)
align_model_special_tokens(teacher_model, tokenizer)
model.resize_token_embeddings(teacher_model.lm_head.weight.shape[0])

print(model.lm_head.weight.shape)
print(teacher_model.lm_head.weight.shape)

assert model.lm_head.weight.shape[0] == teacher_model.lm_head.weight.shape[0]

dataset_root = os.environ.get(
    "DATASET_ROOT",
    "hf://datasets/fisherman611/text_to_cypher_distillation/Cypherbench",
)
ds = load_dataset(
    "json",
    data_files={
        "train": f"{dataset_root}/train.jsonl",
        "validation": f"{dataset_root}/dev.jsonl",
        "test": f"{dataset_root}/test.jsonl",
    },
)

raw_train_dataset = ds["train"]
raw_eval_dataset = ds["validation"]
if args_cli.max_train_samples is not None:
    raw_train_dataset = raw_train_dataset.select(range(min(args_cli.max_train_samples, len(raw_train_dataset))))
if args_cli.max_eval_samples is not None:
    raw_eval_dataset = raw_eval_dataset.select(range(min(args_cli.max_eval_samples, len(raw_eval_dataset))))

def add_messages(example):
    instruction = example["system_prompt"].strip() + "\n\n" + example["user_prompt"].strip()
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": example["response"]}
        ]
    }
    
train_dataset = raw_train_dataset.map(add_messages, remove_columns=raw_train_dataset.column_names)
eval_dataset = raw_eval_dataset.map(add_messages, remove_columns=raw_eval_dataset.column_names)

print(train_dataset[0])
print(eval_dataset[0])

fsdp_config={'limit_all_gathers': True, 'forward_prefetch': True, 'backward_prefetch': 'backward_pre'}
training_args = GKDConfig(
                        output_dir=args_cli.output_dir,
                        logging_steps=args_cli.logging_steps,
                        num_train_epochs=args_cli.num_train_epochs,
                        max_steps=args_cli.max_steps,
                        warmup_ratio=0.1,
                        per_device_eval_batch_size=args_cli.per_device_eval_batch_size,
                        per_device_train_batch_size=args_cli.per_device_train_batch_size,
                        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
                        gradient_checkpointing=False,
                        learning_rate=args_cli.learning_rate,
                        eval_strategy='epoch',
                        save_strategy="epoch",
                         metric_for_best_model="eval_loss",
                        load_best_model_at_end=args_cli.load_best_model_at_end,
                        lr_scheduler_type="cosine",
                        bf16=True, 
                        max_length=1024,
                        save_total_limit=3,
                        report_to=[],
                        lmbda=lmbda,
                        beta=beta,
                        temperature=1.0,
                        seq_kd=args_cli.seq_kd,
                        )


trainer = DistillTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    token_entropy_percentile_threshold=threshold,
    indirect_kd_alpha=indirect_kd_alpha,
)
trainer.train()
