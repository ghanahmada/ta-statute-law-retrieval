# Full Pipeline — Rerun Commands for Paper

Complete sequence to reproduce all results. Each phase depends on the previous one.

## Prerequisites

- GPU VPS with vLLM, PyTorch, DGL installed
- HuggingFace token set (`HF_TOKEN` env var)
- Docker (for annotation tool, local machine)

---

## Phase 0 — Pull Data

```bash
python src/scripts/pull_kuhperdata.py
```

---

## Phase 1 — Validate Expansion (requires vLLM)

Start vLLM server (separate terminal):

```bash
vllm serve QuantTrio/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b \
  --max-num-seqs 50 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

Run validation:

```bash
# Validate humanized expansion
python src/data/validate_expansion.py \
  --expansion_logs data/kuhperdata-exp/expansion_logs.jsonl \
  --dataset kuhperdata-humanized \
  --output_name kuhperdata-exp-v2 \
  --concurrency 8

# Validate summarized expansion
python src/data/validate_expansion.py \
  --expansion_logs data/kuhperdata-summ-exp/expansion_logs.jsonl \
  --dataset kuhperdata-summarized \
  --output_name kuhperdata-summ-exp-v2 \
  --concurrency 8
```

Overwrite originals with validated qrels:

```bash
cp data/kuhperdata-exp-v2/qrels_train.tsv data/kuhperdata-exp/qrels_train.tsv
cp data/kuhperdata-exp-v2/qrels_test.tsv data/kuhperdata-exp/qrels_test.tsv
cp data/kuhperdata-exp-v2/validation_log.jsonl data/kuhperdata-exp/validation_log.jsonl

cp data/kuhperdata-summ-exp-v2/qrels_train.tsv data/kuhperdata-summ-exp/qrels_train.tsv
cp data/kuhperdata-summ-exp-v2/qrels_test.tsv data/kuhperdata-summ-exp/qrels_test.tsv
cp data/kuhperdata-summ-exp-v2/validation_log.jsonl data/kuhperdata-summ-exp/validation_log.jsonl
```

---

## Phase 2 — Split Test → Val/Test

```bash
# KUHPerdata (4 variants)
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-humanized
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-summarized
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-exp
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-summ-exp

# Cross-lingual
python src/scripts/split_test_to_val.py --dataset_dir data/bsard
python src/scripts/split_test_to_val.py --dataset_dir data/stard

# IL-PCSR: native val split (dev → val, test → test)
python src/scripts/prepare_ilpcsr.py
```

---

## Phase 3 — Precompute BM25 + Embeddings for Para-GNN

```bash
python src/paragnn/precompute.py --dataset kuhperdata-exp --max_relevant 0
python src/paragnn/precompute.py --dataset kuhperdata-summ-exp --max_relevant 0
python src/paragnn/precompute.py --dataset bsard --max_relevant 0
python src/paragnn/precompute.py --dataset stard --max_relevant 0
python src/paragnn/precompute.py --dataset ilpcsr --max_relevant 0
```

---

## Phase 4 — Evaluate All Methods on KUHPerdata (Main Results Table)

```bash
# BM25
python src/evaluate_bm25.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_bm25.py --dataset kuhperdata-summ-exp --max_relevant 0

# BM25 + Reranker
python src/evaluate_rerank.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_rerank.py --dataset kuhperdata-summ-exp --max_relevant 0

# BM25 + GAR
python src/evaluate_gar.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_gar.py --dataset kuhperdata-summ-exp --max_relevant 0

# Dense (BGE-M3)
python src/evaluate_dense_retrieval.py --dataset kuhperdata-exp --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset kuhperdata-summ-exp --max_relevant 0 --save_embeddings

# JNLP Stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_jnlp.py --dataset kuhperdata-summ-exp --max_relevant 0

# Para-GNN
python src/evaluate_paragnn.py --dataset kuhperdata-exp --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode none --max_relevant 0

# StructGNN
python src/evaluate_paragnn.py --dataset kuhperdata-exp --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode structural --max_relevant 0
```

---

## Phase 5 — Cross-Lingual Evaluation (BSARD, STARD, IL-PCSR)

```bash
# BM25
python src/evaluate_bm25.py --dataset bsard --max_relevant 0
python src/evaluate_bm25.py --dataset stard --max_relevant 0
python src/evaluate_bm25.py --dataset ilpcsr --max_relevant 0

# JNLP
python src/evaluate_jnlp.py --dataset bsard --max_relevant 0
python src/evaluate_jnlp.py --dataset stard --max_relevant 0
python src/evaluate_jnlp.py --dataset ilpcsr --max_relevant 0

# Para-GNN
python src/evaluate_paragnn.py --dataset bsard --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset stard --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset ilpcsr --structure_mode none --max_relevant 0

# StructGNN
python src/evaluate_paragnn.py --dataset bsard --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset stard --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset ilpcsr --structure_mode structural --max_relevant 0
```

---

## Phase 6 — Annotation Study (parallel, local machine)

```bash
cd annotation-tool

# Generate 80 pairs (40 cases × 2 variants)
python generate_pairs.py

# Start annotation tool
docker compose up --build

# After annotation complete, compute kappa
python compute_agreement.py --from-api http://localhost:8000
```

---

## Notes

- `--max_relevant 0` = no filtering on number of relevant articles per query (paper setting)
- Expanded datasets (`kuhperdata-exp`, `kuhperdata-summ-exp`) use validated qrels after Phase 1
- vLLM server only needed for Phase 1 (validation). Can be stopped after.
- Phase 6 runs on local machine with Docker, independent of GPU VPS.
- All evaluate scripts output metrics to stdout. Capture with `| tee outputs/log_<method>_<dataset>.txt`.
