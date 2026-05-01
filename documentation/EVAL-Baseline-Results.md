# Evaluation Results (2026-05-01)

All results on test split, `max_relevant=5`.

## KUHPerdata Results

| Method | Dataset | MRR@10 | Recall@10 | Precision@10 | Hit Rate |
|--------|---------|--------|-----------|--------------|----------|
| BM25 (stemmer+stopwords) | humanized | 0.0601 | 0.1032 | 0.0167 | 14.6% |
| BM25 (stemmer+stopwords) | summarized | 0.0947 | 0.1484 | 0.0249 | 23.3% |
| Dense (BGE-M3) | humanized | 0.0926 | 0.1451 | 0.0264 | 20.4% |
| Dense (BGE-M3) | summarized | 0.1085 | 0.1440 | 0.0282 | 22.5% |
| JNLP Stage 1 | humanized | 0.4356 | 0.6024 | 0.1436 | 71.8% |
| JNLP Stage 1 | summarized | 0.3930 | 0.5473 | 0.1354 | 70.5% |
| Para-GNN (base) | humanized | 0.4857 | 0.5375 | — | 70.2% |
| Para-GNN (base) | summarized | 0.4793 | 0.4305 | — | 67.0% |
| StructGNN | humanized | 0.5176 | 0.6213 | — | 77.3% |
| StructGNN | summarized | 0.5058 | 0.5818 | — | 80.7% |

## Cross-Dataset Results (BM25)

Source: `logs/2/bm25.txt`

| Dataset | Lang | MRR@10 | Recall@10 | Precision@10 | Hit Rate | N Queries |
|---------|------|--------|-----------|--------------|----------|-----------|
| KUHPerdata | id | 0.1467 | 0.0858 | 0.0316 | 24.06% | 212 |
| BSARD | fr | 0.2488 | 0.2664 | 0.0716 | 42.34% | 222 |
| IL-PCSR | en | 0.1558 | 0.1017 | 0.0332 | 25.04% | 1,254 |
| STARD | zh | 0.3382 | 0.4272 | 0.0643 | 53.25% | 308 |

## Cross-Dataset Results (JNLP Stage 1)

Source: `logs/3/jnlp_stage1.txt`

| Dataset | Lang | MRR@10 | Recall@10 | Hit Rate | vs BM25 MRR |
|---------|------|--------|-----------|----------|-------------|
| KUHPerdata | id | 0.3997 | 0.3939 | 62.0% | +173% |
| BSARD | fr | 0.3284 | 0.3047 | 44.6% | +32% |
| IL-PCSR | en | 0.0493 | 0.0323 | 12.9% | -68% |
| STARD | zh | 0.2705 | 0.3895 | 49.4% | -20% |

## Cross-Dataset Results (Para-GNN)

Source: `documentation/EVAL-ParaGNN.md`

| Dataset | Lang | MRR@10 | Recall@10 | Hit Rate | Alpha | vs JNLP S1 |
|---------|------|--------|-----------|----------|-------|-------------|
| KUHPerdata-humanized | id | 0.456 | 0.542 | 67.8% | 0.9 | +24% |
| KUHPerdata-summarized | id | 0.458 | 0.522 | 71.1% | 0.8 | +36% |
| BSARD | fr | 0.493 | 0.487 | 68.5% | 0.7 | +50% |
| STARD | zh | 0.527 | 0.617 | 72.4% | 0.8 | +94% |

## Cross-Dataset Results (GAR)

Source: `logs/3/gar.txt`

| Dataset | Lang | MRR@10 | Recall@10 | Hit Rate |
|---------|------|--------|-----------|----------|
| KUHPerdata | id | 0.1909 | 0.2450 | 37.3% |
| BSARD | fr | 0.1938 | 0.2058 | 36.5% |
| STARD | zh | 0.3728 | 0.4562 | 55.2% |
| IL-PCSR | en | 0.1558 | 0.1017 | 25.0% |

## Cross-Dataset Results (StructGNN)

StructGNN cross-dataset results (BSARD, IL-PCSR, STARD) pending.

## Key Observations

- **StructGNN is best on both MRR and Recall** for humanized (0.5176 / 0.6213)
- JNLP Stage 1 has higher recall than Para-GNN base, but StructGNN beats both
- Para-GNN debiased scores drop drastically (0.4857 → 0.1881 humanized), confirming hub bias
- StructGNN debiased also drops but less (0.5176 → 0.2272), structure features help non-hub articles
- Summarized queries generally perform better on BM25/Dense (more formal language matches statute text)
- Best alpha: Para-GNN=0.8 (80% GNN + 20% BM25), StructGNN=0.9 (90% GNN + 10% BM25)
- Para-GNN generalizes well: +24% to +94% over JNLP S1 across all tested datasets
- JNLP Stage 1 regresses on IL-PCSR (-68%) and STARD (-20%) due to long queries and large corpus scale

## Commands Used

```bash
# BM25
python src/evaluate_bm25.py --dataset kuhperdata-humanized --use_stemmer --remove_stopwords
python src/evaluate_bm25.py --dataset kuhperdata-summarized --use_stemmer --remove_stopwords

# Dense (BGE-M3)
python src/evaluate_dense_retrieval.py --dataset kuhperdata-humanized --save_embeddings
python src/evaluate_dense_retrieval.py --dataset kuhperdata-summarized --save_embeddings

# JNLP Stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-humanized --stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-summarized --stage 1

# Para-GNN (precompute first)
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode none

# StructGNN
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural

# Cross-dataset (all methods)
python src/evaluate_bm25.py --dataset all
python src/evaluate_jnlp.py --dataset all --stage 1
python src/evaluate_paragnn.py --dataset bsard --structure_mode none
python src/evaluate_paragnn.py --dataset stard --structure_mode none
```

## Environment

- GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120)
- Torch: 2.7.0+cu128 (paragnn env), .venv-jnlp for others
- BGE-M3: BAAI/bge-m3 (1024d)
- Dataset: kuhperdata-humanized (383 test queries), kuhperdata-summarized (373 test queries)
