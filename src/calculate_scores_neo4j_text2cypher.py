import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.scoring import (  # noqa: E402
    DatasetScoreConfig,
    identity_database,
    run_dataset_score_cli,
)


CONFIG = DatasetScoreConfig(
    description="Calculate Neo4j Text2Cypher metrics and save results to JSON",
    default_output_dir="results/Neo4j_Text2Cypher/calculated_scores_Qwen3_0.6B/",
    default_subset="bluesky",
    subset_choices=[
        "bluesky",
        "buzzoverflow",
        "companies",
        "neoflix",
        "fincen",
        "gameofthrones",
        "grandstack",
        "movies",
        "network",
        "northwind",
        "offshoreleaks",
        "recommendations",
        "stackoverflow2",
        "twitch",
        "twitter",
    ],
    connector_name="neo4j_text2cypher_db",
    host="bolt+s://demo.neo4jlabs.com",
    use_subset_credentials=True,
    database_transform=identity_database,
    debug=True,
)


if __name__ == "__main__":
    run_dataset_score_cli(CONFIG)
