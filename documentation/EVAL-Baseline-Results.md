# Baseline Evaluation Results (2026-05-01)

All results on kuhperdata test split, `max_relevant=5`.

## Results Table

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

## Key Observations

- **StructGNN is best on both MRR and Recall** for humanized (0.5176 / 0.6213)
- JNLP Stage 1 has higher recall than Para-GNN base, but StructGNN beats both
- Para-GNN debiased scores drop drastically (0.4857 → 0.1881 humanized), confirming hub bias
- StructGNN debiased also drops but less (0.5176 → 0.2272), structure features help non-hub articles
- Summarized queries generally perform better on BM25/Dense (more formal language matches statute text)
- Best alpha: Para-GNN=0.8 (80% GNN + 20% BM25), StructGNN=0.9 (90% GNN + 10% BM25)

## Alpha Grid Search (Para-GNN base, humanized)

| Alpha | MRR@10 | R@10 | Hit |
|-------|--------|------|-----|
| 0.8 | 0.4857 | 0.5375 | 70.2% |

## Alpha Grid Search (StructGNN, humanized)

| Alpha | MRR@10 | R@10 | Hit |
|-------|--------|------|-----|
| 0.9 | 0.5176 | 0.6213 | 77.3% |

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
```

## Environment

- GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120)
- Torch: 2.7.0+cu128 (paragnn env), .venv-jnlp for others
- BGE-M3: BAAI/bge-m3 (1024d)
- Dataset: kuhperdata-humanized (383 test queries), kuhperdata-summarized (373 test queries)
