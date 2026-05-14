import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.neo4j_connector import Neo4jConnector

def read_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filepath} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"File {filepath} is not a valid JSON.")
        return None
    

def create_connection(benchmark: str, db_name: str, password: str):
    if benchmark == "Cypherbench":
        connector =  Neo4jConnector(
            name="cypherbench-db",
            host="neo4j://127.0.0.1",
            port=7687,
            username="neo4j",
            password=password,
            database=db_name,
            debug=True,
        )
    elif benchmark == "Mind_the_query":
        connector = Neo4jConnector(
            name="mind-the-query-db",
            host="neo4j://127.0.0.1",
            port=7687,
            username="neo4j",
            password=password,
            database=db_name,
            debug=True,
        )
    elif benchmark == "Neo4j_Text2Cypher":
        connector = Neo4jConnector(
            name="neo4j-text2cypher-db",
            host="bolt+s://demo.neo4jlabs.com",
            port=7687,
            username=db_name,
            password=password,
            database=db_name,
            debug=True,
        )

    num_entities = connector.get_num_entities()
    num_relations = connector.get_num_relations()

    print("\n[SUCCESS] Connected successfully!")
    print(f"Database: {db_name}")
    print(f"Total nodes: {num_entities}")
    print(f"Total relationships: {num_relations}")

    return connector

def load_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def build_messages(question: str, schema: str, prompts_dir: str = "prompts/generator") -> list:
    system_prompt = load_prompt(f"{prompts_dir}/system_prompt.txt")
    user_prompt_template = load_prompt(f"{prompts_dir}/user_prompt.txt")

    # inject variables vào user prompt
    user_prompt = user_prompt_template.format(
        question=question,
        schema=schema
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return messages