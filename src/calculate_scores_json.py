import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.scoring import (  # noqa: E402
    DEFAULT_METRICS,
    calculate_aggregates,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metric scores from a scored result JSON file"
    )
    parser.add_argument("--input", required=True, help="Path to scored result JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for aggregated metrics JSON",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric names to aggregate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregates = calculate_aggregates(read_json(args.input), metrics=args.metrics)

    if args.output:
        write_json(args.output, aggregates)
        print(f"Saved aggregated metrics to {args.output}")

    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
