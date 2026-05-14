import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.scoring import DatasetScoreConfig, run_dataset_score_cli  # noqa: E402


CONFIG = DatasetScoreConfig(
    description="Calculate Mind the Query metrics and save results to JSON",
    default_output_dir="results/Mind_the_query/calculated_scores_Qwen3_0.6B/",
    default_subset="bloom50",
    subset_choices=["bloom50", "healthcare", "wwc"],
    connector_name="mind-the-query-db",
    host="neo4j://127.0.0.1",
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD"),
)


if __name__ == "__main__":
    run_dataset_score_cli(CONFIG)
