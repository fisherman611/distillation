import os
import json
import argparse
import threading
import torch
from urllib.parse import urlparse
from pathlib import Path
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from src.benchmark_data_loader import (
    DEFAULT_HF_DATASET_REPO,
    load_benchmark_json,
)
from src.utils import build_messages
from src.llm_services import parse_json_from_string, parse_llm_response
from src.schema import Nl2CypherSample
from src.logger_config import setup_logger

RESULTS_DIR = "results"
LOG_DIR = "logging_data/qwen3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_infer_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Qwen model and benchmark on text2cypher problems with CKPT support"
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=["Cypherbench", "Mind_the_query", "Neo4j_Text2Cypher"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--data_source",
        default="auto",
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
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Local path, HF repo id, hf:// path, or HF URL of LoRA/full fine-tuned checkpoint",
    )
    parser.add_argument(
        "--ckpt_revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (branch/tag/commit) for --ckpt_path",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device to run the model on",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Max generation length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help="Write partial JSON output after this many processed samples. Defaults to no incremental flush.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p for sampling"
    )
    parser.add_argument(
        "--top-k", type=int, default=0, help="Top-k for sampling"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test samples"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path. If omitted, use the default path pattern in results/<benchmark>/",
    )
    return parser.parse_args()


def _parse_hf_source(source: str, revision: Optional[str] = None) -> Dict[str, Optional[str]]:
    raw = str(source).strip()
    normalized = raw.rstrip("/")

    if os.path.isdir(raw):
        return {
            "kind": "local",
            "path": raw,
            "repo_id": None,
            "revision": None,
            "subfolder": None,
        }

    def _pack_hf(repo_id: str, rev: Optional[str], subfolder: Optional[str]) -> Dict[str, Optional[str]]:
        return {
            "kind": "hf",
            "path": None,
            "repo_id": repo_id,
            "revision": rev,
            "subfolder": subfolder,
        }

    if normalized.startswith("hf://"):
        content = normalized[len("hf://"):].strip("/")
        parts = [p for p in content.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid HF path '{source}'. Expected: hf://<owner>/<repo>/<optional/subfolder>"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
        return _pack_hf(repo_id, revision, subfolder)

    parsed = urlparse(normalized)
    if parsed.scheme in {"http", "https"} and parsed.netloc in {"huggingface.co", "www.huggingface.co"}:
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            resolved_revision = revision
            subfolder = None
            if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
                resolved_revision = revision or parts[3]
                subfolder = "/".join(parts[4:]) if len(parts) > 4 else None
            elif len(parts) > 2:
                subfolder = "/".join(parts[2:])
            return _pack_hf(repo_id, resolved_revision, subfolder)

    shorthand = normalized.strip("/")
    parts = [p for p in shorthand.split("/") if p]
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        repo_id = f"{parts[0]}/{parts[1]}"
        resolved_revision = revision or parts[3]
        subfolder = "/".join(parts[4:]) if len(parts) > 4 else None
        return _pack_hf(repo_id, resolved_revision, subfolder)

    if len(parts) >= 3 and not normalized.startswith(".") and not os.path.exists(raw):
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:])
        return _pack_hf(repo_id, revision, subfolder)

    return _pack_hf(normalized, revision, None)


def _materialize_ckpt_source(source: str, revision: Optional[str] = None) -> Tuple[str, Optional[str]]:
    parsed = _parse_hf_source(source, revision)
    if parsed["kind"] == "local":
        return parsed["path"], None  # type: ignore[return-value]

    repo_id = parsed["repo_id"]  # type: ignore[assignment]
    repo_revision = parsed["revision"]
    subfolder = parsed["subfolder"]

    if not subfolder:
        return repo_id, repo_revision  # type: ignore[return-value]

    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    snapshot_dir = snapshot_download(
        repo_id=repo_id,  # type: ignore[arg-type]
        revision=repo_revision,
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
        token=token,
    )
    resolved = os.path.join(snapshot_dir, subfolder)
    return resolved, None


CKPT_MARKER_FILES = (
    "adapter_config.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


def _local_has_ckpt_marker(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and any((path / filename).exists() for filename in CKPT_MARKER_FILES)


def _find_latest_local_ckpt_dir(path: str | Path) -> Optional[str]:
    root = Path(path)
    if not root.is_dir():
        return None
    if _local_has_ckpt_marker(root):
        return str(root)

    candidates = []
    for child in root.rglob("*"):
        if not child.is_dir() or not _local_has_ckpt_marker(child):
            continue
        step = int(child.name) if child.name.isdigit() else -1
        candidates.append((step, child.stat().st_mtime, child))

    if not candidates:
        return None
    return str(max(candidates, key=lambda item: (item[0], item[1]))[2])


def _resolve_ckpt_dir(source: str, revision: Optional[str] = None) -> Tuple[str, Optional[str]]:
    resolved_source, resolved_revision = _materialize_ckpt_source(source, revision)
    if os.path.isdir(resolved_source):
        nested_ckpt = _find_latest_local_ckpt_dir(resolved_source)
        if nested_ckpt is not None and nested_ckpt != resolved_source:
            print(f"Resolved checkpoint directory: {nested_ckpt}")
            logger.info(f"Resolved checkpoint directory: {nested_ckpt}")
            return nested_ckpt, None
    return resolved_source, resolved_revision


def _local_or_hf_has_file(source: str, filename: str, revision: Optional[str] = None) -> bool:
    resolved_source, resolved_revision = _resolve_ckpt_dir(source, revision)

    if os.path.isdir(resolved_source):
        return os.path.exists(os.path.join(resolved_source, filename))

    try:
        repo_files = list_repo_files(repo_id=resolved_source, revision=resolved_revision)
        return filename in repo_files
    except Exception:
        return False


def _resolve_ckpt_file(source: str, filename: str, revision: Optional[str] = None) -> Optional[str]:
    resolved_source, resolved_revision = _resolve_ckpt_dir(source, revision)

    if os.path.isdir(resolved_source):
        candidate = os.path.join(resolved_source, filename)
        return candidate if os.path.exists(candidate) else None

    try:
        return hf_hub_download(repo_id=resolved_source, filename=filename, revision=resolved_revision)
    except Exception:
        return None


def init_model(model_name_or_path, ckpt_path=None, ckpt_revision=None, device=None):
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_peft = False
    resolved_ckpt_source = ckpt_path
    resolved_ckpt_revision = ckpt_revision
    if ckpt_path:
        if _local_or_hf_has_file(ckpt_path, "adapter_config.json", ckpt_revision):
            is_peft = True
            resolved_ckpt_source, resolved_ckpt_revision = _resolve_ckpt_dir(ckpt_path, ckpt_revision)
            print("This is LoRA finetune")
        elif _local_or_hf_has_file(ckpt_path, "adapter_model.safetensors", ckpt_revision) or _local_or_hf_has_file(ckpt_path, "adapter_model.bin", ckpt_revision):
            raise RuntimeError(
                "Found adapter weights but missing adapter_config.json at --ckpt_path. "
                "Please point --ckpt_path to the exact LoRA checkpoint directory (the one containing adapter_config.json), "
                "not its parent folder."
            )
        elif _local_or_hf_has_file(ckpt_path, "config.json", ckpt_revision):
            # If it's a full HF checkpoint we should just load directly from it
            resolved_ckpt_source, resolved_ckpt_revision = _resolve_ckpt_dir(ckpt_path, ckpt_revision)
            model_name_or_path = resolved_ckpt_source
            print("This is a full finetune")

    logger.info(f"Loading tokenizer from {model_name_or_path}")
    print(f"Loading tokenizer from {model_name_or_path}")
    # Force left-padding for decoder-only architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
        revision=resolved_ckpt_revision if model_name_or_path == resolved_ckpt_source else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_name_or_path} on {device}")
    print(f"Loading model from {model_name_or_path} on {device}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float16

    device_map = "auto" if device == "cuda" else {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        revision=resolved_ckpt_revision if model_name_or_path == resolved_ckpt_source else None,
    )

    if ckpt_path and is_peft:
        print(f"Loading PEFT checkpoint weights from {resolved_ckpt_source}")
        logger.info(f"Loading PEFT checkpoint weights from {resolved_ckpt_source}")
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, resolved_ckpt_source, revision=resolved_ckpt_revision)
            model = model.merge_and_unload()
            print("Successfully loaded and merged LoRA weights.")
        except Exception as peft_e:
            print(f"Failed to load as PEFT model: {peft_e}.")

    # For full checkpoint manually passed without config.json (rare, just fallback)
    elif ckpt_path and model_name_or_path != ckpt_path:
        print(f"Attempting manual weight loading from {ckpt_path}")
        import safetensors.torch

        safetensor_path = _resolve_ckpt_file(ckpt_path, "model.safetensors", ckpt_revision)
        bin_path = _resolve_ckpt_file(ckpt_path, "pytorch_model.bin", ckpt_revision)

        def _load_weights_tolerant(state_dict, source_label: str):
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys = set(incompatible.missing_keys)
            unexpected_keys = set(incompatible.unexpected_keys)

            # Tied embeddings can omit lm_head in some checkpoints.
            allowed_missing = {"lm_head.weight"}
            bad_missing = sorted(missing_keys - allowed_missing)
            bad_unexpected = sorted(unexpected_keys)
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    f"Failed loading checkpoint weights from {ckpt_path}.\n"
                    f"Missing keys: {sorted(missing_keys)}\n"
                    f"Unexpected keys: {sorted(unexpected_keys)}\n"
                    "This usually means --ckpt_path is not a compatible full checkpoint directory."
                )

            if hasattr(model, "tie_weights"):
                model.tie_weights()
            if missing_keys:
                print(f"Warning: missing keys tolerated for {source_label}: {sorted(missing_keys)}")

        if safetensor_path:
            state_dict = safetensors.torch.load_file(safetensor_path)
            try:
                _load_weights_tolerant(state_dict, "model.safetensors")
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed loading checkpoint weights from {ckpt_path}: {e}\n"
                    "For LoRA checkpoints, pass the folder containing adapter_config.json.\n"
                    "For full checkpoints, pass the folder containing config.json and full model weights."
                ) from e
            print("Loaded full model weights from safetensors.")
        elif bin_path:
            try:
                _load_weights_tolerant(torch.load(bin_path, map_location="cpu"), "pytorch_model.bin")
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed loading checkpoint weights from {ckpt_path}: {e}\n"
                    "For LoRA checkpoints, pass the folder containing adapter_config.json.\n"
                    "For full checkpoints, pass the folder containing config.json and full model weights."
                ) from e
            print("Loaded full model weights from pytorch_model.bin.")
        else:
            print(
                f"Warning: neither adapter_config, config.json, model.safetensors, nor pytorch_model.bin found in {ckpt_path}."
            )

    model.eval()
    return tokenizer, model


def generate_response_batch(tokenizer, model, batch_messages, max_length=1024, temperature=0.5, top_p=0.95, top_k=0):
    def _apply_template(messages):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    texts = [_apply_template(messages) for messages in batch_messages]
    
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
    generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def is_full_db(db: Optional[str]) -> bool:
    return db is None or str(db).strip().lower() in {"full", "all", ""}


def _schema_relative_path(benchmark: str, graph_name: str) -> Optional[str]:
    if benchmark == "Cypherbench":
        return f"graphs/schemas/{graph_name}_schema.json"
    if benchmark == "Mind_the_query":
        return f"graphs/schemas/{graph_name}.json"
    return None


def load_schema_for_graph(
    benchmark: str,
    graph_name: str,
    data_source: str = "auto",
    hf_dataset_repo: str = DEFAULT_HF_DATASET_REPO,
    hf_dataset_revision: Optional[str] = None,
) -> Optional[str]:
    schema_relative_path = _schema_relative_path(benchmark, graph_name)
    if schema_relative_path is None:
        return None

    try:
        schema, schema_source = load_benchmark_json(
            benchmark=benchmark,
            relative_path=schema_relative_path,
            data_source=data_source,
            hf_dataset_repo=hf_dataset_repo,
            hf_dataset_revision=hf_dataset_revision,
        )
    except Exception as e:
        logger.warning(f"Schema file not found for graph={graph_name}: {e}")
        return None

    logger.info(f"Loaded schema for graph={graph_name} from {schema_source}")
    return json.dumps(schema, indent=4, ensure_ascii=False)


def load_schema_and_subset_test_data(
    benchmark,
    db=None,
    limit=None,
    data_source: str = "auto",
    hf_dataset_repo: str = DEFAULT_HF_DATASET_REPO,
    hf_dataset_revision: Optional[str] = None,
):
    raw_test_data, test_source = load_benchmark_json(
        benchmark=benchmark,
        relative_path="test.json",
        data_source=data_source,
        hf_dataset_repo=hf_dataset_repo,
        hf_dataset_revision=hf_dataset_revision,
    )
    logger.info(f"Loaded test set from {test_source}")

    use_all = is_full_db(db)
    subset_test_data = []

    for item in raw_test_data:
        try:
            sample = Nl2CypherSample(**item)
            if use_all or sample.graph == db:
                subset_test_data.append(sample)
        except Exception as e:
            logger.warning(f"Skipping invalid item: {e}")

    if limit is not None:
        subset_test_data = subset_test_data[:limit]

    shared_schema_str = None
    schema_map = {}

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if use_all:
            unique_graphs = sorted(
                {sample.graph for sample in subset_test_data if sample.graph}
            )
            for graph_name in unique_graphs:
                schema_map[graph_name] = load_schema_for_graph(
                    benchmark=benchmark,
                    graph_name=graph_name,
                    data_source=data_source,
                    hf_dataset_repo=hf_dataset_repo,
                    hf_dataset_revision=hf_dataset_revision,
                )
        else:
            shared_schema_str = load_schema_for_graph(
                benchmark=benchmark,
                graph_name=db,
                data_source=data_source,
                hf_dataset_repo=hf_dataset_repo,
                hf_dataset_revision=hf_dataset_revision,
            )

    return subset_test_data, shared_schema_str, schema_map


def get_question_and_schema(
    sample: Nl2CypherSample,
    benchmark: str,
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
):
    question = sample.nl_question

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if shared_schema_str is not None:
            schema_str = shared_schema_str
        else:
            schema_str = (schema_map or {}).get(sample.graph)
    else:
        schema_str = sample.schema

    return question, schema_str


def run_batch_inference(
    test_data,
    benchmark,
    db,
    tokenizer,
    model,
    max_length,
    batch_size=1,
    temperature=0.5,
    top_p=0.95,
    top_k=0,
    shared_schema_str=None,
    schema_map=None,
    output_path=None,
    flush_every=None,
):
    results = []
    errors = []
    run_name = db if not is_full_db(db) else "full"
    next_flush_at = flush_every if flush_every else None

    progress_bar = tqdm(total=len(test_data), desc=f"Running {benchmark}/{run_name}")

    for i in range(0, len(test_data), batch_size):
        batch_samples = test_data[i : i + batch_size]
        
        batch_messages = []
        batch_questions = []
        for sample in batch_samples:
            question, schema_str = get_question_and_schema(
                sample, benchmark, shared_schema_str, schema_map
            )
            messages = build_messages(question, schema_str)
            batch_messages.append(messages)
            batch_questions.append(question)
            
        try:
            raw_responses = generate_response_batch(
                tokenizer, model, batch_messages, max_length=max_length,
                temperature=temperature, top_p=top_p, top_k=top_k
            )
            
            for sample, question, raw_response in zip(batch_samples, batch_questions, raw_responses):
                qid = sample.qid if sample.qid is not None else sample.instance_id
                if qid is None:
                    qid = "unknown"
                
                logger.info("-" * 80)
                logger.info(f"Processing item: {qid}")
                logger.info(f"Graph: {sample.graph}")
                logger.info(f"Question: {question}")
                
                try:
                    parsed = parse_llm_response(raw_response)
                    parsed_json = parse_json_from_string(parsed["final_answer"])
            
                    if not parsed_json or "cypher" not in parsed_json:
                        raise ValueError("Failed to parse JSON or missing 'cypher' key")
            
                    cypher = parsed_json["cypher"]
                    sample.pred_cypher = cypher
            
                    logger.info(f"Generated cypher for item {qid}: {cypher}")
            
                    results.append({
                        "qid": qid,
                        "graph": sample.graph,
                        "question": question,
                        "raw_response": raw_response,
                        "think": parsed.get("think", ""),
                        "final_answer": parsed.get("final_answer", ""),
                        "cypher": cypher,
                        "success": True,
                        "error": None,
                        "sample": sample.model_dump(mode="json"),
                    })
                except Exception as e:
                    logger.error(f"Error parsing item {qid}: {e}", exc_info=True)
                    sample.pred_cypher = None
                    errors.append({"qid": qid, "error": str(e)})
                    results.append({
                        "qid": qid,
                        "graph": sample.graph,
                        "question": question,
                        "raw_response": raw_response,
                        "think": "",
                        "final_answer": "",
                        "cypher": None,
                        "success": False,
                        "error": str(e),
                        "sample": sample.model_dump(mode="json"),
                    })
                    
        except Exception as batch_e:
            logger.error(f"Error generating batch starting at index {i}: {batch_e}", exc_info=True)
            for sample in batch_samples:
                qid = sample.qid if sample.qid is not None else sample.instance_id
                errors.append({"qid": qid, "error": str(batch_e)})

        progress_bar.update(len(batch_samples))
        processed_count = min(i + len(batch_samples), len(test_data))
        if output_path and flush_every and processed_count >= next_flush_at:
            write_output_snapshot(results, output_path)
            while next_flush_at <= processed_count:
                next_flush_at += flush_every
        
    logger.info(f"Finished. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    return results, errors


def compact_output(results):
    output = []
    for r in results:
        output.append({
            "question": r["question"],
            "graph": r["graph"],
            "gold_cypher": r["sample"].get("gold_cypher"),
            "pred_cypher": r["cypher"],
        })
    return output


def write_output_snapshot(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(compact_output(results), f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, output_path)


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

    tokenizer, model = init_model(args.model, args.ckpt_path, args.ckpt_revision, device=args.device)

    print(f"Running benchmark={args.benchmark}, db={db_name}, samples={len(subset_test_data)}")
    print(f"Using batch_size={args.batch_size}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distillm_adaptive_srkl_kd0.7_wrel1.6.json"
    )

    results, errors = run_batch_inference(
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
        shared_schema_str=schema_str,
        schema_map=schema_map,
        output_path=output_path,
        flush_every=args.flush_every,
    )

    os.makedirs(Path(RESULTS_DIR) / args.benchmark, exist_ok=True)
    
    # Differentiate output file based on whether a custom ckpt was used
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_rkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_sfkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_csd.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_distillm.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_sfkl.json"   #default kd_ratio 0.5
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl_updated_1.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl_updated_3_0.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distillm.json" 

        # Differentiate output file based on whether a custom ckpt was used
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_updated_span_fkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_updated_span_rkl_2.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_updated_span_sfkl.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_csd.json"
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_sfkl.json"   #default kd_ratio 0.5
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl_updated_1.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distill_fdd_srkl_updated_3_0.json" 
    # output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_{args.model.split('/')[-1]}_distillm.json" 
    write_output_snapshot(results, output_path)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
