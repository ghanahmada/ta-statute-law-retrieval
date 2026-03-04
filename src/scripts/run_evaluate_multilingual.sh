#!/bin/bash
set -euo pipefail

# Evaluate multilingual dense retrieval models on kuhperdata
# Supports: BGE-M3 (default), multilingual-e5-base, mdeberta-v3-base
# Run from TA project root: bash src/scripts/run_evaluate_multilingual.sh [model]
#
# Usage:
#   bash src/scripts/run_evaluate_multilingual.sh                          # BGE-M3
#   bash src/scripts/run_evaluate_multilingual.sh multilingual-e5-base    # E5 multilingual
#   bash src/scripts/run_evaluate_multilingual.sh bge-m3                  # BGE-M3 explicit

MODEL="${1:-bge-m3}"

case "$MODEL" in
  bge-m3)
    HF_MODEL="BAAI/bge-m3"
    OUTPUT_TAG="bge_m3"
    ;;
  multilingual-e5-base)
    HF_MODEL="intfloat/multilingual-e5-base"
    OUTPUT_TAG="me5_base"
    ;;
  multilingual-e5-large)
    HF_MODEL="intfloat/multilingual-e5-large"
    OUTPUT_TAG="me5_large"
    ;;
  *)
    echo "Unknown model: $MODEL"
    echo "Options: bge-m3, multilingual-e5-base, multilingual-e5-large"
    exit 1
    ;;
esac

echo "Model: ${HF_MODEL}"
echo "Output tag: ${OUTPUT_TAG}"

python src/evaluate_dense_retrieval.py \
  --bge_model "${HF_MODEL}" \
  --embeddings_dir "outputs/embeddings/${OUTPUT_TAG}" \
  --save_embeddings \
  --batch_size 32 \
  --max_length 1024
