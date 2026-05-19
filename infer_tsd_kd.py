import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig

from infer import (
    get_question_and_schema,
    init_model,
    is_full_db,
    load_schema_and_subset_test_data,
    write_output_snapshot,
)
from src.benchmark_data_loader import DEFAULT_HF_DATASET_REPO
from src.llm_services import parse_json_from_string, parse_llm_response
from src.schema import Nl2CypherSample
from src.utils import load_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSD-KD inference with the same chat message format used during TSD-KD training."
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=["Cypherbench", "Mind_the_query", "Neo4j_Text2Cypher"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--data_source",
        default="hf",
        choices=["local", "hf", "auto"],
        help="Where to load benchmark data from",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default=DEFAULT_HF_DATASET_REPO,
        help="Hugging Face dataset repo id for benchmark files",
    )
    parser.add_argument(
        "--hf_dataset_revision",
        type=str,
        default=None,
        help="Optional revision (branch/tag/commit) for --hf_dataset_repo",
    )
    parser.add_argument(
        "--db",
        default=None,
        help='Database name of each benchmark. If omitted or set to "full", use all data.',
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model name or path")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Local path, HF repo id, hf:// path, or HF URL of TSD-KD full checkpoint",
    )
    parser.add_argument(
        "--ckpt_revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision for --ckpt_path",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to run the model on",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help="Write partial JSON output after this many processed samples.",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k for sampling")
    parser.add_argument(
        "--generation-mode",
        default="train_eval",
        choices=["train_eval", "legacy"],
        help="Generation mode. 'train_eval' matches DistillTrainer generation behavior.",
    )
    parser.add_argument(
        "--enable-thinking-template",
        action="store_true",
        help="Allow tokenizer chat templates that support thinking to enable it. Default is no-think.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument(
        "--prompts-dir",
        default="prompts/generator",
        help="Directory containing system_prompt.txt and user_prompt.txt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path",
    )
    return parser.parse_args()


def build_tsd_kd_messages(question: str, schema: str, prompts_dir: str) -> list[dict[str, str]]:
    system_prompt = load_prompt(f"{prompts_dir}/system_prompt.txt").strip()
    user_prompt_template = load_prompt(f"{prompts_dir}/user_prompt.txt")
    user_prompt = user_prompt_template.format(question=question, schema=schema).strip()
    instruction = f"{system_prompt}\n\n{user_prompt}"
    return [{"role": "user", "content": instruction}]


def apply_tsd_kd_chat_template(tokenizer, messages, enable_thinking_template: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=enable_thinking_template,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_response_batch_tsd_kd(
    tokenizer,
    model,
    batch_messages,
    max_length=1024,
    temperature=0.5,
    top_p=0.95,
    top_k=0,
    enable_thinking_template=False,
    generation_mode: str = "train_eval",
):
    texts = [
        apply_tsd_kd_chat_template(tokenizer, messages, enable_thinking_template)
        for messages in batch_messages
    ]

    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        if generation_mode == "train_eval":
            generation_config = GenerationConfig(
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            if (
                hasattr(model, "generation_config")
                and getattr(model.generation_config, "eos_token_id", None) is not None
            ):
                generation_config.eos_token_id = model.generation_config.eos_token_id

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            sequences = outputs.sequences
        else:
            sequences = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

    generated_ids = sequences[:, inputs["input_ids"].shape[-1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def run_batch_inference_tsd_kd(
    test_data: list[Nl2CypherSample],
    benchmark: str,
    db: Optional[str],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    prompts_dir: str,
    enable_thinking_template: bool = False,
    generation_mode: str = "train_eval",
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
    output_path: Optional[Path] = None,
    flush_every: Optional[int] = None,
):
    results = []
    errors = []
    run_name = db if not is_full_db(db) else "full"
    next_flush_at = flush_every if flush_every else None

    progress_bar = tqdm(total=len(test_data), desc=f"Running TSD-KD {benchmark}/{run_name}")

    for i in range(0, len(test_data), batch_size):
        batch_samples = test_data[i : i + batch_size]
        batch_messages = []
        batch_questions = []

        for sample in batch_samples:
            question, schema_str = get_question_and_schema(
                sample, benchmark, shared_schema_str, schema_map
            )
            messages = build_tsd_kd_messages(question, schema_str, prompts_dir)
            batch_messages.append(messages)
            batch_questions.append(question)

        batch_error = None
        try:
            raw_responses = generate_response_batch_tsd_kd(
                tokenizer,
                model,
                batch_messages,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking_template=enable_thinking_template,
                generation_mode=generation_mode,
            )
        except Exception as batch_e:
            batch_error = str(batch_e)
            raw_responses = ["" for _ in batch_samples]

        for sample, question, raw_response in zip(batch_samples, batch_questions, raw_responses):
            qid = sample.qid if sample.qid is not None else sample.instance_id
            if qid is None:
                qid = "unknown"

            try:
                if batch_error is not None:
                    raise RuntimeError(batch_error)
                parsed = parse_llm_response(raw_response)
                parsed_json = parse_json_from_string(parsed["final_answer"])
                if not parsed_json or "cypher" not in parsed_json:
                    raise ValueError("Failed to parse JSON or missing 'cypher' key")
                cypher = parsed_json["cypher"]
                error = None
                success = True
            except Exception as e:
                cypher = None
                error = str(e)
                success = False
                errors.append({"qid": qid, "error": error})

            sample.pred_cypher = cypher
            results.append(
                {
                    "qid": qid,
                    "graph": sample.graph,
                    "question": question,
                    "raw_response": raw_response,
                    "cypher": cypher,
                    "success": success,
                    "error": error,
                    "sample": sample.model_dump(mode="json"),
                }
            )

        progress_bar.update(len(batch_samples))
        processed_count = min(i + len(batch_samples), len(test_data))
        if output_path and flush_every and processed_count >= next_flush_at:
            write_output_snapshot(results, output_path)
            while next_flush_at <= processed_count:
                next_flush_at += flush_every

    progress_bar.close()
    return results, errors


def main():
    args = parse_args()
    if args.flush_every is not None and args.flush_every <= 0:
        raise ValueError("--flush-every must be a positive integer")

    subset_test_data, schema_str, schema_map = load_schema_and_subset_test_data(
        args.benchmark,
        args.db,
        args.limit,
        args.data_source,
        args.hf_dataset_repo,
        args.hf_dataset_revision,
    )
    db_name = args.db if not is_full_db(args.db) else "full"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path("results") / "tsd-kd" / args.benchmark / f"{db_name}_cyphers_result.json"
    )

    tokenizer, model = init_model(args.model, args.ckpt_path, args.ckpt_revision, device=args.device)

    print("Using TSD-KD prompt format: one user message containing system_prompt + user_prompt")
    print(f"Tokenizer thinking template enabled: {args.enable_thinking_template}")
    print(f"Generation mode: {args.generation_mode}")
    print(f"Running benchmark={args.benchmark}, db={db_name}, samples={len(subset_test_data)}")
    print(f"Using batch_size={args.batch_size}, flush_every={args.flush_every}")

    results, errors = run_batch_inference_tsd_kd(
        test_data=subset_test_data,
        benchmark=args.benchmark,
        db=args.db,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        prompts_dir=args.prompts_dir,
        enable_thinking_template=args.enable_thinking_template,
        generation_mode=args.generation_mode,
        shared_schema_str=schema_str,
        schema_map=schema_map,
        output_path=output_path,
        flush_every=args.flush_every,
    )
    write_output_snapshot(results, output_path)

    print(f"Saved results to: {output_path}")
    print(f"Success: {len(results) - len(errors)}, Failed: {len(errors)}")


if __name__ == "__main__":
    main()
