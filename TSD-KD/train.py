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


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSD-KD on Cypherbench.")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--model-name", dest="model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--indirect-kd-alpha", dest="indirect_kd_alpha", type=float, default=0.1)
    parser.add_argument("--seq-kd", dest="seq_kd", action="store_true")
    parser.add_argument(
        "--teacher-model-name",
        dest="teacher_model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
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

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

teacher_model_name = args_cli.teacher_model_name
attn = "sdpa"
# The model to optimise
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn, torch_dtype=torch.bfloat16, pad_token_id=tokenizer.pad_token_id, trust_remote_code=True)#.to(f"cuda:{local_rank}")

# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, attn_implementation=attn, torch_dtype=torch.bfloat16, trust_remote_code=True)#.to(f"cuda:{local_rank}")
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

def add_messages(example):
    instruction = example["system_prompt"].strip() + "\n\n" + example["user_prompt"].strip()
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": example["response"]}
        ]
    }
    
train_dataset = ds["train"].map(add_messages, remove_columns=ds["train"].column_names)
eval_dataset = ds["validation"].map(add_messages, remove_columns=ds["validation"].column_names)
print(train_dataset[0])
print(eval_dataset[0])

fsdp_config={'limit_all_gathers': True, 'forward_prefetch': True, 'backward_prefetch': 'backward_pre'}
training_args = GKDConfig(
                        output_dir=f"tsd-kd-Qwen2.5-1.5B-Instruct",
                        logging_steps=10, 
                        num_train_epochs=3,
                        warmup_ratio=0.1,
                        per_device_eval_batch_size=4,
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing=False,
                        learning_rate=5e-6,
                        eval_strategy='epoch',
                        save_strategy="epoch",
                         metric_for_best_model="eval_loss",
                        load_best_model_at_end=True,
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
