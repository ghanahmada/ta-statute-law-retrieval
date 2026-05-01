# Multilingual Statute Law Retrieval

Benchmarking lexical, dense, learned, graph-based, and agentic retrieval methods across 4 multilingual statute law datasets.

## Datasets

| Dataset | Language | Corpus | Queries | Source |
|---------|----------|--------|---------|--------|
| KUHPerdata (humanized/summarized) | Indonesian | 2,127 | 1,847 | Constructed (this work) |
| BSARD | French | 22,633 | 1,108 | Louis et al., 2022 |
| IL-PCSR | English | 936 | 6,271 | Parikh et al., 2023 |
| STARD | Chinese | 55,348 | 1,543 | Li et al., 2023 |

## Retrieval Methods

| Method | Type | Script |
|--------|------|--------|
| BM25 | Lexical | `src/evaluate_bm25.py` |
| BGE-M3 | Dense | `src/evaluate_dense_retrieval.py` |
| JNLP (CatBoost + BGE-M3) | Learned | `src/evaluate_jnlp.py` |
| Para-GNN | Graph | `src/evaluate_paragnn.py --structure_mode none` |
| StructGNN | Graph + Structural | `src/evaluate_paragnn.py --structure_mode structural` |
| GAR | Graph Adaptive Reranking | `src/evaluate_gar.py` |
| Rerank | BM25 + Reranker | `src/evaluate_rerank.py` |
| QUAM | SetAff Reranking | `src/evaluate_quam.py` |
| Context-1 | Agentic (LLM + Hybrid Search) | `src/context_1/evaluate_context1.py` |

## Key Results (KUHPerdata-humanized, test split)

| Method | MRR@10 | Recall@10 | Hit Rate |
|--------|--------|-----------|----------|
| BM25 | 0.0601 | 0.1032 | 14.6% |
| Dense (BGE-M3) | 0.0926 | 0.1451 | 20.4% |
| JNLP Stage 1 | 0.4356 | 0.6024 | 71.8% |
| Para-GNN | 0.4857 | 0.5375 | 70.2% |
| **StructGNN** | **0.5176** | **0.6213** | **77.3%** |

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Quick Start

```bash
# BM25
python src/evaluate_bm25.py --dataset kuhperdata-humanized --top_k 10

# Dense (BGE-M3)
python src/evaluate_dense_retrieval.py --dataset kuhperdata-humanized --save_embeddings

# JNLP Stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-humanized --stage 1

# Para-GNN (precompute embeddings first)
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode none

# StructGNN
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural

# Evaluate all datasets
python src/evaluate_bm25.py --dataset all --top_k 10
```

## Project Structure

```
src/
├── evaluate_bm25.py              # BM25 evaluation
├── evaluate_dense_retrieval.py   # BGE-M3 cosine similarity
├── evaluate_jnlp.py             # JNLP pipeline entry point
├── evaluate_paragnn.py           # Para-GNN / StructGNN evaluation
├── evaluate_gar.py               # Graph Adaptive Reranking
├── evaluate_rerank.py            # BM25 + Reranker ablation
├── evaluate_quam.py              # QUAM evaluation
├── dataset.py                    # KUHPerdata dataset builder
├── context_1/                    # Agentic retrieval (LLM + hybrid search)
├── paragnn/                      # Para-GNN / StructGNN models
├── jnlp/                         # JNLP 3-stage pipeline
├── gar/                          # Graph Adaptive Reranking
├── quam/                         # QUAM models
├── data/                         # Ground truth expansion pipeline
├── util/                         # DataLoader, BM25, metrics
├── analysis/                     # Hub bias and error analysis
├── inference/                    # Model export and demo inference
└── scripts/                      # Dataset preparation scripts
```

## Documentation

- [`documentation/HLD.md`](documentation/HLD.md) — High-level design and architecture
- [`documentation/EVAL-Baseline-Results.md`](documentation/EVAL-Baseline-Results.md) — All evaluation results
- [`documentation/PIPELINE-QrelsExpansion.md`](documentation/PIPELINE-QrelsExpansion.md) — Ground truth expansion pipeline
- [`documentation/benchmark-implementation-guide.md`](documentation/benchmark-implementation-guide.md) — Guide for adding new methods
