import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from huggingface_hub import hf_hub_download

DEFAULT_HF_DATASET_REPO = "fisherman611/text_to_cypher_distillation"
VALID_DATA_SOURCES = {"local", "hf", "auto"}


def _read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_json_from_hf_dataset(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
) -> Tuple[Any, str]:
    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN")
    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return _read_json(local_file), local_file


def load_benchmark_json(
    benchmark: str,
    relative_path: str,
    data_source: str = "auto",
    hf_dataset_repo: str = DEFAULT_HF_DATASET_REPO,
    hf_dataset_revision: Optional[str] = None,
) -> Tuple[Any, str]:
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(
            f"Invalid data_source='{data_source}'. Choose from {sorted(VALID_DATA_SOURCES)}."
        )

    normalized_rel_path = relative_path.replace("\\", "/").lstrip("/")
    local_path = Path("benchmarks") / benchmark / Path(normalized_rel_path)
    local_error: Optional[Exception] = None

    if data_source in {"local", "auto"} and local_path.exists():
        try:
            return _read_json(local_path), str(local_path)
        except Exception as e:
            local_error = e
            if data_source == "local":
                raise

    if data_source == "local":
        raise FileNotFoundError(f"Local benchmark file not found: {local_path}")

    hf_file = f"{benchmark}/{normalized_rel_path}"
    try:
        return _download_json_from_hf_dataset(
            repo_id=hf_dataset_repo,
            filename=hf_file,
            revision=hf_dataset_revision,
        )
    except Exception as hf_error:
        if local_error is not None:
            raise RuntimeError(
                f"Failed reading local file '{local_path}' and failed downloading "
                f"dataset file '{hf_file}' from '{hf_dataset_repo}'."
            ) from hf_error
        raise RuntimeError(
            f"Failed downloading dataset file '{hf_file}' from '{hf_dataset_repo}'."
        ) from hf_error
