#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$(pwd)/models/local}"
MERGED_CSV="${MERGED_CSV:-$(pwd)/modeling/combined_listings.csv}"
EMBEDDINGS_CSV="${EMBEDDINGS_CSV:-$(pwd)/modeling/embeddings.csv}"
INPUT_DIR="${INPUT_DIR:-}"

export AVM_OUTPUT_CSV="${MERGED_CSV}"
if [[ -n "${INPUT_DIR}" ]]; then
  export AVM_LOCAL_INPUT_DIR="${INPUT_DIR}"
fi

echo "1/3 Merging listings..."
python3 modeling/download_and_merge.py

echo "2/3 Generating image embeddings..."
export AVM_INPUT_CSV="${MERGED_CSV}"
export AVM_OUTPUT_CSV="${EMBEDDINGS_CSV}"
python3 modeling/generate_embeddings.py

echo "3/3 Training final model..."
export AVM_INPUT_CSV="${EMBEDDINGS_CSV}"
export AVM_MODEL_DIR="${MODEL_DIR}"
unset AVM_MODEL_GCS_URI || true
python3 modeling/final_training.py

echo "Done."
echo "Merged CSV: ${MERGED_CSV}"
echo "Embeddings CSV: ${EMBEDDINGS_CSV}"
echo "Model dir: ${MODEL_DIR}"
