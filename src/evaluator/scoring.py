from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from src.neo4j_connector import Neo4jConnector
    from src.schema import Nl2CypherSample


MetricResult = float | dict[str, str]
MetricFn = Callable[..., float]

VALID_METRICS = ("execution_accuracy", "psjs", "executable")
DEFAULT_METRICS = ("execution_accuracy", "psjs", "executable")
END_OF_TURN = "<end_of_turn>"


def identity_database(subset: str) -> str:
    return subset


def dotted_database(subset: str) -> str:
    return subset.replace("_", ".")


@dataclass(frozen=True)
class DatasetScoreConfig:
    description: str
    default_output_dir: str
    default_subset: str
    subset_choices: Sequence[str]
    connector_name: str
    host: str
    port: int = 7687
    username: str | None = None
    password: str | None = None
    use_subset_credentials: bool = False
    database_transform: Callable[[str], str] = dotted_database
    debug: bool = False


def clean_pred_cypher(pred_cypher: str | None) -> str:
    pred_cypher = pred_cypher or ""
    if pred_cypher.endswith(END_OF_TURN):
        return pred_cypher[: -len(END_OF_TURN)].strip()
    return pred_cypher


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fin:
        return json.load(fin)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def avg_and_round(nums: Iterable[float], n: int = 4) -> float:
    nums = list(nums)
    return round(sum(nums) / len(nums), n) if nums else math.nan


def metric_value(value: Any) -> float:
    return value if isinstance(value, (int, float)) else 0.0


def calculate_aggregates(
    result: Iterable[dict[str, Any] | Nl2CypherSample],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, dict[str, float]]:
    records = list(result)
    return {
        "overall": {
            metric: avg_and_round(
                metric_value(_get_metrics(record).get(metric))
                for record in records
                if metric in _get_metrics(record)
            )
            for metric in metrics
        }
    }


def _get_metrics(record: dict[str, Any] | Nl2CypherSample) -> dict[str, Any]:
    if isinstance(record, dict):
        return record.get("metrics", {})
    return getattr(record, "metrics", {})


@lru_cache(maxsize=1)
def get_metric_functions() -> dict[str, MetricFn]:
    from src.metrics import (  # noqa: PLC0415
        executable,
        execution_accuracy,
        provenance_subgraph_jaccard_similarity,
    )

    return {
        "execution_accuracy": execution_accuracy,
        "psjs": provenance_subgraph_jaccard_similarity,
        "executable": executable,
    }


def safe_compute(
    metric_name: str,
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector: Neo4jConnector,
) -> MetricResult:
    try:
        return get_metric_functions()[metric_name](
            pred_cypher=pred_cypher,
            target_cypher=target_cypher,
            neo4j_connector=neo4j_connector,
        )
    except Exception as exc:
        return {"error": f"{metric_name} failed: {exc}"}


def compute_sample_metrics(
    item: Nl2CypherSample,
    metrics: Sequence[str],
    neo4j_connector: Neo4jConnector,
) -> Nl2CypherSample:
    item = copy.deepcopy(item)
    pred_cypher = clean_pred_cypher(item.pred_cypher)
    for metric in metrics:
        item.metrics[metric] = safe_compute(
            metric,
            pred_cypher,
            item.gold_cypher,
            neo4j_connector,
        )
    return item


def score_records(
    records: Iterable[dict[str, Any]],
    neo4j_connector: Neo4jConnector,
    metrics: Sequence[str] = DEFAULT_METRICS,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    output_records = []
    for item in tqdm(list(records), desc=desc):
        pred_cypher = clean_pred_cypher(item.get("pred_cypher", ""))
        target_cypher = item.get("gold_cypher", "")
        output_records.append(
            {
                "graph": item.get("graph", ""),
                "gold_cypher": target_cypher,
                "pred_cypher": pred_cypher,
                "metrics": {
                    metric: safe_compute(
                        metric,
                        pred_cypher,
                        target_cypher,
                        neo4j_connector,
                    )
                    for metric in metrics
                },
            }
        )
    return output_records


def add_dataset_score_args(
    parser: argparse.ArgumentParser,
    config: DatasetScoreConfig,
) -> None:
    parser.add_argument("--input", required=True, help="Path to input final_results.json")
    parser.add_argument(
        "--output_dir",
        default=config.default_output_dir,
        help="Directory where the scored JSON file will be written",
    )
    parser.add_argument(
        "--subset",
        default=config.default_subset,
        choices=list(config.subset_choices),
        help="Subset / graph name to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit number of samples to evaluate",
    )
    parser.add_argument("--host", default=config.host, help="Neo4j host")
    parser.add_argument("--port", type=int, default=config.port, help="Neo4j port")
    parser.add_argument("--name", default=config.connector_name, help="Connector name")
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name. Defaults to the selected subset.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(VALID_METRICS),
        default=list(DEFAULT_METRICS),
        help="Metrics to calculate",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.debug,
        help="Enable Neo4j connector debug logging",
    )

    if config.use_subset_credentials:
        parser.add_argument(
            "--username",
            default=None,
            help="Neo4j username. Defaults to the selected subset.",
        )
        parser.add_argument(
            "--password",
            default=None,
            help="Neo4j password. Defaults to the selected subset.",
        )
    else:
        parser.add_argument("--username", default=config.username, help="Neo4j username")
        parser.add_argument("--password", default=config.password, help="Neo4j password")


def build_connector(
    args: argparse.Namespace,
    config: DatasetScoreConfig,
) -> Neo4jConnector:
    from src.neo4j_connector import Neo4jConnector  # noqa: PLC0415

    username = args.username
    password = args.password
    if config.use_subset_credentials:
        username = username or args.subset
        password = password or args.subset

    database = args.database or config.database_transform(args.subset)
    return Neo4jConnector(
        name=args.name,
        host=args.host,
        port=args.port,
        username=username,
        password=password,
        database=database,
        debug=args.debug,
    )


def filter_subset(
    records: Iterable[dict[str, Any]],
    subset: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    subset_records = [item for item in records if item.get("graph") == subset]
    return subset_records[:limit] if limit is not None else subset_records


def run_dataset_score_cli(config: DatasetScoreConfig) -> None:
    parser = argparse.ArgumentParser(description=config.description)
    add_dataset_score_args(parser, config)
    args = parser.parse_args()

    records = filter_subset(read_json(args.input), args.subset, args.limit)
    connector = build_connector(args, config)
    output_records = score_records(
        records,
        connector,
        metrics=args.metrics,
        desc=f"Evaluating {args.subset}",
    )

    output_path = Path(args.output_dir) / f"{args.subset}_cyphers_result.json"
    write_json(output_path, output_records)
    print(f"Saved {len(output_records)} results to {output_path}")
