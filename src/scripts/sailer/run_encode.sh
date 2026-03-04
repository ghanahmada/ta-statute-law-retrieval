#!/bin/bash
set -euo pipefail

# Encode corpus + queries with finetuned SAILER model for retrieval evaluation
# Run from TA project root: bash src/scripts/sailer/run_encode.sh

SAILER_ROOT="$(cd "$(dirname "$0")/../../../../SAILER" && pwd)"
export PYTHONPATH="${SAILER_ROOT}/src:${PYTHONPATH:-}"

MODEL_DIR="./outputs/sailer_kuhperdata"
ENCODE_DIR="./data/sailer/encode"
OUTPUT_DIR="./outputs/sailer_kuhperdata/embeddings"

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
echo "Run: python src/scripts/sailer/evaluate_retrieval.py"
