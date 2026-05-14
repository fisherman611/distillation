import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.scoring import DatasetScoreConfig, run_dataset_score_cli  # noqa: E402


CONFIG = DatasetScoreConfig(
    description="Calculate CypherBench metrics and save results to JSON",
    default_output_dir="results/Cypherbench/calculated_scores_Qwen3_0.6B/",
    default_subset="nba",
    subset_choices=[
        "nba",
        "flight_accident",
        "fictional_character",
        "company",
        "geography",
        "movie",
        "politics",
    ],
    connector_name="cypherbench-db",
    host="neo4j://127.0.0.1",
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD"),
)


if __name__ == "__main__":
    run_dataset_score_cli(CONFIG)
