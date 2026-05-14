import re
import json
from src.logger_config import setup_logger

logger = setup_logger(__name__)

def parse_json_from_string(text):
    try:
        # 1. Find the first JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in the string.")
        json_block = match.group(0)

        # 2. Normalize Python-style booleans
        json_block = (
            json_block.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )

        # 3. Handle triple quotes: """...""" → "..."
        def convert_triple_quotes(match):
            content = match.group(1)
            content = content.replace("\n", "\\n").replace('"', '\\"')
            return f'"{content}"'

        json_block = re.sub(r'"""(.*?)"""', convert_triple_quotes, json_block, flags=re.DOTALL)

        # 4. Escape newlines inside normal strings
        def escape_newlines_in_string(m):
            return m.group(0).replace("\n", "\\n")

        json_block = re.sub(r'"(.*?)"', escape_newlines_in_string, json_block, flags=re.DOTALL)

        # 5. Parse JSON
        return json.loads(json_block)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Original text: {text}")
        return None
    except Exception as e:
        print(f"Unknown error while parsing JSON: {e}")
        print(f"Original text: {text}")
        return None


def parse_llm_response(response: str):
    """
    Extract the <think>...</think> section and the final answer.
    """
    if not response or not isinstance(response, str):
        return {"think": "", "final_answer": ""}

    match = re.search(
        r"<think\s*>\s*(.*?)\s*</think\s*>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        think = match.group(1).strip()
        final_answer = re.sub(
            r"<think\s*>.*?</think\s*>",
            "",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    else:
        think = ""
        final_answer = response.strip()

    return {"think": think, "final_answer": final_answer}