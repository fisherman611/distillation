#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK="${BENCHMARK:-Cypherbench}"
MAX_WORKERS="${MAX_WORKERS:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-src/results/baseline_llm/${BENCHMARK}}"

MODELS=(${MODELS:-qwen7b mamba mixtral qwen32b llama devstral minimax deepseek glm4.7})
GRAPHS=(${GRAPHS:-company fictional_character flight_accident geography movie nba politics})

mkdir -p "${OUTPUT_BASE}"

echo "Benchmark: ${BENCHMARK}"
echo "Models: ${MODELS[*]}"
echo "Graphs: ${GRAPHS[*]}"
echo "Output base: ${OUTPUT_BASE}"
echo

for model in "${MODELS[@]}"; do
  model_output_dir="${OUTPUT_BASE}/${model}"
  mkdir -p "${model_output_dir}"

  echo "Running model: ${model}"
  "${PYTHON_BIN}" src/baseline_llm.py \
    --benchmark "${BENCHMARK}" \
    --model "${model}" \
    --graphs "${GRAPHS[@]}" \
    --max_workers "${MAX_WORKERS}" \
    --output_dir "${model_output_dir}"

  echo "Finished model: ${model}"
  echo
done

echo "All baseline LLM runs completed."
