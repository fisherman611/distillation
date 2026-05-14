import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from src.llm_services import parse_json_from_string, parse_llm_response

hf_token = os.getenv("HF_READ_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Qwen model and test parsing LLM output"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run the model on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max generation length"
    )
    return parser.parse_args()


def init_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        token=hf_token
    )
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, messages, max_length=512, enable_thinking=False):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.7,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def main():
    args = parse_args()
    selected_device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading model: {args.model}")
    print(f"Using device: {selected_device}")

    tokenizer, model = init_model(args.model)

    # Example: force the model to answer with <think> + JSON
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Always answer in this exact format:\n"
                "<think>your reasoning here</think>\n"
                '{"final_result": "..."}'
            ),
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    raw_response = generate_response(tokenizer, model, messages, args.max_length)

    print("\n=== RAW MODEL RESPONSE ===")
    print(raw_response)

    parsed = parse_llm_response(raw_response)

    print("\n=== PARSED THINK ===")
    print(parsed["think"])

    print("\n=== PARSED FINAL ANSWER ===")
    print(parsed["final_answer"])

    parsed_json = parse_json_from_string(parsed["final_answer"])

    print("\n=== PARSED JSON ===")
    print(parsed_json)


if __name__ == "__main__":
    main()