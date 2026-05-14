import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import ast
import re
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from src.benchmark_data_loader import (
    DEFAULT_HF_DATASET_REPO,
    load_benchmark_json,
)
from src.schema import Nl2CypherSample
from src.logger_config import setup_logger

load_dotenv()

LOG_DIR = "logging_data/baseline_llm"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)


MODEL_CONFIGS = {
    # ──────────────────────────────── coding models ───────────────────────
    "qwen2.5-coder-7b": {
        "model_id": "qwen/qwen2.5-coder-7b-instruct",
        "api_type": "chat",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    },
    "mamba-codestral-7b": {
        "model_id": "mistralai/mamba-codestral-7b-v0.1",
        "api_type": "chat",
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
    },
    "mixtral-8x22b":{
        "model_id": "mistralai/mixtral-8x22b-instruct-v0.1",
        "api_type": "chat",
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
    },
    "qwen2.5-coder-32b": {
        "model_id": "qwen/qwen2.5-coder-32b-instruct",
        "api_type": "chat",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    },
    "llama3.3-70b": {
        "model_id": "meta/llama-3.3-70b-instruct",
        "api_type": "chat",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    },
    "devstral-2-123b": {
        "model_id": "mistralai/devstral-2-123b-instruct-2512",
        "api_type": "chat",
        "temperature": 0.15,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": {"seed": 42},
    },
    "minimax-m2.5": {
        "model_id":"minimaxai/minimax-m2.5",
        "api_type": "chat",
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
    # ──────────────────────────────── general / reasoning ─────────────────
    "deepseek-v3": {
        "model_id": "deepseek-ai/deepseek-v3.1",
        "api_type": "chat",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 8192,
        "extra_body": {"chat_template_kwargs": {"thinking": True}},
    },
    "glm4.7": {
        "model_id": "z-ai/glm4.7",
        "api_type": "chat",
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 16384,
        "extra_body":{"chat_template_kwargs":{"enable_thinking":True,"clear_thinking":False}},

    }

}

MODEL_ALIASES = {
    "qwen7b": "qwen2.5-coder-7b",
    "mamba": "mamba-codestral-7b",
    "mixtral": "mixtral-8x22b",
    "qwen32b": "qwen2.5-coder-32b",
    "llama": "llama3.3-70b",
    "devstral": "devstral-2-123b",
    "minimax": "minimax-m2.5",
    "deepseek": "deepseek-v3",
    "glm4.7": "glm4.7",
}

def build_openai_client():
    from openai import OpenAI
    return OpenAI(
        base_url=os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
        api_key=os.getenv("NVIDIA_API_KEY", "no-key"),
    )

def call_nvidia_chat(client, cfg: dict, messages: list[dict]) -> str:
    """Call a chat-completion model and return the assistant text (streaming)."""
    kwargs = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": cfg.get("temperature", 0.2),
        "top_p": cfg.get("top_p", 0.7),
        "max_tokens": cfg.get("max_tokens", 8192),
        "stream": True,
    }
    if "extra_body" in cfg:
        kwargs["extra_body"] = cfg["extra_body"]

    response = client.chat.completions.create(**kwargs)
    
    full_content = ""
    # We use stream=True to handle reasoning_content such as in deepseek-v3
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
            
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            # We skip printing reasoning to avoid clutter, but we could log it
            pass
            
        if chunk.choices and chunk.choices[0].delta.content is not None:
            full_content += chunk.choices[0].delta.content

    return full_content


def call_nvidia_completion(client, cfg: dict, prompt: str) -> str:
    """Call a text-completion model and return the generated text."""
    kwargs = {
        "model": cfg["model_id"],
        "prompt": prompt,
        "temperature": cfg.get("temperature", 0.1),
        "top_p": cfg.get("top_p", 0.7),
        "max_tokens": cfg.get("max_tokens", 200),
        "stream": False,
    }
    response = client.completions.create(**kwargs)
    return response.choices[0].text or ""


_CYPHER_KEYS = (
    "cypher",
    "final_cypher",
    "pred_cypher",
    "query",
    "cypher_query",
    "answer",
    "final_answer",
)

_CYPHER_START_RE = re.compile(
    r"\b(OPTIONAL\s+MATCH|MATCH|CALL|WITH|UNWIND|RETURN|CREATE|MERGE)\b",
    flags=re.IGNORECASE,
)


def _strip_reasoning_and_special_tokens(text: str) -> str:
    """Remove common model reasoning wrappers and chat template tokens."""
    text = re.sub(r"<think\s*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<reasoning\s*>.*?</reasoning\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"(<\|eot_id\|>|<\|endoftext\|>|<end_of_turn>|</s>|<s>)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _iter_balanced_json_objects(text: str):
    """Yield balanced {...} candidates while respecting quoted strings."""
    start = None
    depth = 0
    quote = None
    escaped = False

    for idx, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            continue

        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : idx + 1]
                start = None


def _parse_json_like(block: str):
    block = block.strip()

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(block)
    except (SyntaxError, ValueError):
        pass

    normalised = block.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(normalised)
    except json.JSONDecodeError:
        return None


def _find_cypher_value(value) -> str:
    if isinstance(value, dict):
        for key in _CYPHER_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _cleanup_cypher_text(candidate)

        for candidate in value.values():
            found = _find_cypher_value(candidate)
            if found:
                return found

    if isinstance(value, list):
        for item in value:
            found = _find_cypher_value(item)
            if found:
                return found

    return ""


def _cleanup_cypher_text(text: str) -> str:
    text = _strip_reasoning_and_special_tokens(str(text))
    text = re.sub(r"^```(?:json|cypher|sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip("`").strip()

    text = re.sub(
        r"^(?:final\s+)?(?:cypher|query|answer|output)\s*(?:query)?\s*[:=-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    lines = []
    for line in text.splitlines():
        if lines and re.match(
            r"^\s*(explanation|note|notes|reasoning|schema|question)\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            break
        lines.append(line)

    return "\n".join(lines).strip()


def _extract_from_fenced_blocks(text: str) -> str:
    fence_re = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n?(?P<body>.*?)```", re.DOTALL)
    blocks = list(fence_re.finditer(text))

    for block in blocks:
        lang = block.group("lang").lower()
        body = block.group("body").strip()
        if lang == "json":
            parsed = _parse_json_like(body)
            found = _find_cypher_value(parsed)
            if found:
                return found
        if lang == "cypher":
            return _cleanup_cypher_text(body)

    for block in blocks:
        body = block.group("body").strip()
        if _CYPHER_START_RE.search(body):
            return _cleanup_cypher_text(body)

    return ""


def _extract_query_from_text(text: str) -> str:
    text = _cleanup_cypher_text(text)
    match = _CYPHER_START_RE.search(text)
    if not match:
        return ""

    query = text[match.start() :]
    query = re.split(
        r"\n\s*(?:explanation|note|notes|reasoning|schema|question)\s*:",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _cleanup_cypher_text(query)


def extract_cypher(raw_text: str) -> str:
    if not raw_text or not isinstance(raw_text, str):
        return ""

    raw_text = _strip_reasoning_and_special_tokens(raw_text)

    fenced = _extract_from_fenced_blocks(raw_text)
    if fenced:
        return fenced

    for block in _iter_balanced_json_objects(raw_text):
        parsed = _parse_json_like(block)
        found = _find_cypher_value(parsed)
        if found:
            return found

    query = _extract_query_from_text(raw_text)
    if query:
        return query

    return _cleanup_cypher_text(raw_text)


# ======================================================
# GENERATOR (single-step: question + schema → Cypher)
# ======================================================
def generator(question, schema, client, model_cfg):
    """
    Baseline LLM generator: takes the question and full schema,
    produces a Cypher query in one shot.
    """
    logger.info("-" * 60)
    logger.info("GENERATOR: Generating Cypher query from question + schema")
    logger.info(f"  Question: {question}")

    max_retries = 5
    base_wait = 3

    for attempt in range(max_retries):
        try:
            schema_str = json.dumps(schema, indent=4, ensure_ascii=False)
            
            system_msg = "You are a Cypher expert. Generate a valid Neo4j Cypher query for the question based on the schema. ONLY return JSON: {\"cypher\": \"...\"}"
            user_msg = f"Schema:\n{schema_str}\n\nQuestion: {question}"

            if model_cfg["api_type"] == "chat":
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                raw = call_nvidia_chat(client, model_cfg, messages)
            else:
                prompt = f"{system_msg}\n\n{user_msg}"
                raw = call_nvidia_completion(client, model_cfg, prompt)

            cypher = extract_cypher(raw)

            if cypher:
                logger.info(f"GENERATOR: Generated Cypher:\n{cypher}")
                logger.info("-" * 60)
                return cypher

            logger.error("GENERATOR: Failed to extract Cypher from LLM response.")
            logger.info("-" * 60)
            return ""

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                wait_time = base_wait * (2 ** attempt)
                logger.warning(f"GENERATOR: NVIDIA NIM Rate limit hit (40 RPM limit). Waiting {wait_time}s and retrying (Attempt {attempt+1}/{max_retries})...")
                sleep(wait_time)
            else:
                logger.error(f"GENERATOR: Error — {e}")
                logger.info("-" * 60)
                return ""
    
    logger.error(f"GENERATOR: Failed after {max_retries} attempts due to rate limiting.")
    return ""


# ======================================================
# MAIN PIPELINE
# ======================================================
def pipeline(sample, schema, client, model_cfg):
    question = sample.nl_question

    logger.info("=" * 80)
    logger.info(f"Processing QID {sample.qid}: {question}")
    logger.info("=" * 80)

    try:
        # Single step: generate Cypher directly
        cypher = generator(question, schema, client, model_cfg)

        logger.info("=" * 80 + "\n")

        # Sleep to keep requests under NVIDIA NIM free-tier limit (40 RPM/1000 requests per day)
        # With max_workers=4, a 6s sleep ensures it averages <1 request per 1.5s
        sleep(10)

        return {
            "question": question,
            "gold": sample.gold_cypher,
            "pred": cypher,
            "graph": sample.graph,
        }

    except Exception as e:
        logger.error(f"Pipeline error for QID {sample.qid}: {e}", exc_info=True)
        raise


# ======================================================
# PARALLEL RUNNER
# ======================================================
def run_parallel_pipeline(
    test_data,
    schema,
    client,
    model_cfg,
    max_workers=4,
):
    results = []
    errors = []

    logger.info("=" * 80)
    logger.info(
        f"Starting parallel pipeline with {len(test_data)} samples, {max_workers} workers"
    )
    logger.info("=" * 80 + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {}

        for item in test_data:
            sample = Nl2CypherSample(**item)
            future = executor.submit(
                pipeline,
                sample,
                schema,
                client,
                model_cfg,
            )
            future_to_qid[future] = sample.qid

        for future in as_completed(future_to_qid):
            qid = future_to_qid[future]
            try:
                results.append(future.result())
            except Exception as e:
                errors.append({"qid": qid, "error": str(e)})
                logger.error(f"Error processing QID {qid}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY:")
    logger.info(f"  Total processed: {len(test_data)}")
    logger.info(f"  Successful: {len(results)}")
    logger.info(f"  Failed: {len(errors)}")
    if errors:
        logger.warning(f"  Failed QIDs: {[e['qid'] for e in errors]}")
    logger.info("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Generator for Cypherbench")
    parser.add_argument("--benchmark", default="Cypherbench")
    parser.add_argument(
        "--data_source",
        default="auto",
        choices=["local", "hf", "auto"],
        help="Where to load benchmark data from",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        default=DEFAULT_HF_DATASET_REPO,
        help="Hugging Face dataset repo id for benchmark files",
    )
    parser.add_argument(
        "--hf_dataset_revision",
        default=None,
        help="Optional revision (branch/tag/commit) for --hf_dataset_repo",
    )
    parser.add_argument("--model", "-m", required=True, help="Model key or alias (e.g. qwen7b, deepseek)")
    parser.add_argument("--graphs", "-g", nargs="+", default=["company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"], help="Subset of graphs to run")
    parser.add_argument("--max_workers", type=int, default=4, help="Max workers for parallel generating")
    parser.add_argument("--output_dir", "-o", default="results", help="Output directory")

    args = parser.parse_args()

    model_key = args.model
    if model_key in MODEL_ALIASES:
        model_key = MODEL_ALIASES[model_key]

    if model_key not in MODEL_CONFIGS:
        logger.error(f"Unknown model '{args.model}'. Available models: {list(MODEL_CONFIGS.keys())}")
        return

    model_cfg = MODEL_CONFIGS[model_key]
    logger.info(f"Using model: {model_key} -> {model_cfg['model_id']}")

    client = build_openai_client()

    try:
        all_test_data, test_source = load_benchmark_json(
            benchmark=args.benchmark,
            relative_path="test.json",
            data_source=args.data_source,
            hf_dataset_repo=args.hf_dataset_repo,
            hf_dataset_revision=args.hf_dataset_revision,
        )
    except Exception as e:
        logger.error(f"Failed to load test.json: {e}")
        return

    logger.info(f"Loaded test set from {test_source}")
    os.makedirs(args.output_dir, exist_ok=True)

    for graph in args.graphs:
        logger.info(f"=============================")
        logger.info(f"Targeting graph: {graph}")
        
        graph_data = [t for t in all_test_data if t.get('graph') == graph]
        if not graph_data:
            logger.warning(f"No test samples found for graph '{graph}'.")
            continue

        if args.benchmark == "Cypherbench":
            schema_relative_path = f"graphs/schemas/{graph}_schema.json"
        elif args.benchmark == "Mind_the_query":
            schema_relative_path = f"graphs/schemas/{graph}.json"
        else:
            logger.warning(f"Schema loading is not configured for benchmark '{args.benchmark}'.")
            continue

        try:
            schema, schema_source = load_benchmark_json(
                benchmark=args.benchmark,
                relative_path=schema_relative_path,
                data_source=args.data_source,
                hf_dataset_repo=args.hf_dataset_repo,
                hf_dataset_revision=args.hf_dataset_revision,
            )
            logger.info(f"Loaded schema for '{graph}' from {schema_source}")
        except Exception as e:
            logger.warning(f"Schema not found for graph '{graph}': {e}")
            continue

        results = run_parallel_pipeline(
            test_data=graph_data,
            schema=schema,
            client=client,
            model_cfg=model_cfg,
            max_workers=args.max_workers
        )

        out_path = os.path.join(args.output_dir, f"{graph}_{model_key}_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
