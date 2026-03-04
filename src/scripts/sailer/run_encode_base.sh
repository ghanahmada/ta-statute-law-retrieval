#!/bin/bash
set -euo pipefail

# Encode corpus + queries with BASE SAILER_en model (no finetuning)
# Run from TA project root: bash src/scripts/sailer/run_encode_base.sh

SAILER_ROOT="$(cd "$(dirname "$0")/../../../SAILER" && pwd)"
export PYTHONPATH="${SAILER_ROOT}/src:${PYTHONPATH:-}"

MODEL_DIR="CSHaitao/SAILER_en"
ENCODE_DIR="./data/sailer/encode"
OUTPUT_DIR="./outputs/sailer_base/embeddings"

mkdir -p "${OUTPUT_DIR}"

echo "SAILER root: ${SAILER_ROOT}"
echo "Model: ${MODEL_DIR}"

# Encode corpus
echo "Encoding corpus..."
python -m dense.driver.encode \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_DIR}" \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --encode_in_path "${ENCODE_DIR}/corpus.jsonl" \
  --encoded_save_path "${OUTPUT_DIR}/corpus_emb.pkl" \
  --p_max_len 256

# Encode queries
echo "Encoding queries..."
python -m dense.driver.encode \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_DIR}" \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --encode_in_path "${ENCODE_DIR}/queries.jsonl" \
  --encoded_save_path "${OUTPUT_DIR}/query_emb.pkl" \
  --q_max_len 512 \
  --encode_is_qry

echo "Encoding complete. Embeddings saved to ${OUTPUT_DIR}"
echo "Run: python src/scripts/sailer/evaluate_retrieval.py --embeddings_dir ${OUTPUT_DIR}"
