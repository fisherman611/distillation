import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any

from huggingface_hub import login
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

hf_token = os.getenv("HF_READ_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)


def init_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    dtype: str = "bfloat16",
    enable_prefix_caching: bool = True,
):
    """
    Initialize a vLLM engine and a tokenizer (tokenizer is only used for
    apply_chat_template – vLLM handles actual tokenisation internally).

    Returns
    -------
    tokenizer : transformers.AutoTokenizer
    llm       : vllm.LLM
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=hf_token,
    )

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        trust_remote_code=True,
        # Pass HF token if needed
        **({"tokenizer_revision": None}),  # keep signature clean
    )

    return tokenizer, llm


# ---------------------------------------------------------------------------
# Single-sample interface (mirrors qwen3.generate_response)
# ---------------------------------------------------------------------------

def generate_response(
    tokenizer,
    llm: "LLM",
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.7,
    enable_thinking: bool = False,
) -> str:
    """
    Generate a single response.  Drop-in replacement for
    ``src.baseline.qwen3.generate_response``.
    """
    responses = generate_batch(
        tokenizer,
        llm,
        [messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    return responses[0]


# ---------------------------------------------------------------------------
# Batch interface (the real value-add of vLLM)
# ---------------------------------------------------------------------------

def generate_batch(
    tokenizer,
    llm: "LLM",
    batch_messages: List[List[Dict[str, str]]],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.7,
    enable_thinking: bool = False,
) -> List[str]:
    """
    Generate responses for a list of message lists in a single vLLM call.

    Parameters
    ----------
    tokenizer        : HF tokenizer (used only for apply_chat_template)
    llm              : vllm.LLM instance
    batch_messages   : list of chat-message lists
    max_new_tokens   : maximum tokens to generate per sample
    temperature      : sampling temperature
    top_p            : nucleus sampling probability
    enable_thinking  : whether to enable Qwen3-style <think> token

    Returns
    -------
    list of decoded strings, one per input sample
    """
    # Build prompts using the tokenizer's chat template
    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for msgs in batch_messages
    ]

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    outputs = llm.generate(prompts, sampling_params)

    return [out.outputs[0].text for out in outputs]
