import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.evaluator.scoring import (
    DEFAULT_METRICS,
    calculate_aggregates,
    compute_sample_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j_info", default="neo4j_info.json")
    parser.add_argument("--result_dir", default="output/gpt-4o")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    args = parser.parse_args()
    print(args)
    print()

    from src.neo4j_connector import Neo4jConnector
    from src.schema import Nl2CypherSample

    with open(os.path.join(args.result_dir, "result.json"), encoding="utf-8") as fin:
        result = [Nl2CypherSample(**item) for item in json.load(fin)]

    with open(args.neo4j_info, encoding="utf-8") as fin:
        neo4j_info = json.load(fin)

    graph2conn = {
        graph: Neo4jConnector(name=graph, **neo4j_info["full"][graph])
        for graph in neo4j_info["test_domains"]
    }

    result_with_metrics = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(
                compute_sample_metrics,
                item,
                args.metrics,
                graph2conn[item.graph],
            )
            for item in result
        ]
        for future in tqdm(as_completed(futures), total=len(result)):
            result_with_metrics.append(future.result())

    aggregated = calculate_aggregates(result_with_metrics, metrics=args.metrics)

    result_path = os.path.join(args.result_dir, "result_with_metrics.json")
    with open(result_path, "w", encoding="utf-8") as fout:
        json.dump(
            [item.model_dump(mode="json") for item in result_with_metrics],
            fout,
            indent=2,
        )
    print(f"Saved result with metrics to {result_path}")

    aggregate_path = os.path.join(args.result_dir, "aggregated_metrics.json")
    with open(aggregate_path, "w", encoding="utf-8") as fout:
        json.dump(aggregated, fout, indent=2)
    print(f"Saved aggregated metrics to {aggregate_path}")

    print()
    print("Aggregated metrics:")
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
