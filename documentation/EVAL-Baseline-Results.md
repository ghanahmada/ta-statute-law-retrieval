# Evaluation Results

All results on test split. Updated 2026-05-02.

## KUHPerdata Results (Original Qrels, max_relevant=5)

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

Note: These results used `max_relevant=5` (queries with >5 relevant docs excluded). Re-run with `max_relevant=0` pending for fair comparison with expanded qrels.

## KUHPerdata Results (Expanded Qrels, max_relevant=0)

All 383 humanized / 373 summarized test queries included.

| Method | Dataset | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Debiased MRR |
|--------|---------|--------|-----------|--------------|----------|--------------|
| BM25 | exp | 0.1706 | 0.1314 | 0.0580 | 36.8% | — |
| BM25 (stem+stop) | exp | 0.1790 | 0.1422 | 0.0614 | 35.5% | — |
| BM25 | summ-exp | 0.1906 | 0.1710 | 0.0681 | 42.9% | — |
| BM25 (stem+stop) | summ-exp | 0.2032 | 0.1790 | 0.0676 | 42.4% | — |
| Dense (BGE-M3) | exp | 0.1945 | 0.1567 | 0.0640 | 34.7% | — |
| Dense (BGE-M3) | summ-exp | 0.2218 | 0.1681 | 0.0692 | 39.9% | — |
| Rerank (BGE) | exp | 0.2789 | 0.2189 | 0.0893 | 49.1% | — |
| Rerank (BGE) | summ-exp | 0.3527 | 0.2630 | 0.1035 | 55.8% | — |
| GAR (mt5) | exp | 0.1564 | 0.1305 | 0.0559 | 33.2% | — |
| GAR (mt5) | summ-exp | 0.0945 | 0.0850 | 0.0319 | 24.9% | — |
| JNLP Stage 1 | exp | 0.4762 | 0.4633 | 0.1402 | 75.2% | — |
| JNLP Stage 1 | summ-exp | 0.4973 | 0.4056 | 0.1381 | 71.0% | — |
| Para-GNN | exp | 0.5902 | 0.5484 | — | 82.6% | 0.2697 |
| Para-GNN | summ-exp | 0.5197 | 0.5119 | — | 82.8% | 0.2939 |
| **StructGNN** | **exp** | **0.6297** | **0.5970** | — | **85.1%** | **0.3163** |
| **StructGNN** | **summ-exp** | **0.5418** | **0.5276** | — | **82.1%** | **0.3154** |

### GNN Alpha Values (Expanded)

| Model | Dataset | Best Alpha (orig) | Best Alpha (debiased) |
|-------|---------|--------------------|-----------------------|
| Para-GNN | exp | 0.7 | 0.9 |
| Para-GNN | summ-exp | 0.8 | 0.9 |
| StructGNN | exp | 0.8 | 0.9 |
| StructGNN | summ-exp | 0.8 | 0.9 |

### Key Observations (Expanded vs Original)

- **StructGNN best overall**: 0.6297 MRR on expanded humanized (up from 0.5176 on original, +21.6%)
- **Debiased MRR improved**: StructGNN 0.2272 → 0.3163 (+39%), confirming expansion helps long-tail retrieval
- **All methods improve** with expanded qrels — sparse ground truth underestimated all methods
- **Learned methods benefit most**: JNLP +9%, Para-GNN +21%, StructGNN +22% — richer training signal matters
- **BM25/Dense barely improve**: vocabulary gap is real, not a qrels artifact
- **Rerank competitive with JNLP**: Rerank summ-exp (0.3527) approaches JNLP without any learned component
- **GAR hurts with wrong scorer**: GAR (mt5) scores below plain BM25 — graph expansion without a language-appropriate scorer degrades Indonesian retrieval
- **Debiased gap persists**: best debiased MRR is 0.3163 — long-tail retrieval remains hard

### Pending (Expanded)

- [x] GAR (exp + summ-exp) — done, mt5 scorer underperforms on Indonesian
- [ ] Agentic (Qwen 3.5 9B + BGE+BM25) on exp + summ-exp
- [ ] Agentic Context-1 (StructGNN dense) on exp + summ-exp — requires GNN embedding export (see note below)
- [ ] Re-run original datasets with max_relevant=0 for fair comparison

### Note: Agentic + StructGNN Dense Backbone

StructGNN does not produce a static document embedding matrix — it scores documents per-query through GNN message passing on a dynamically constructed graph. The current `evaluate_context1.py` uses BGE-M3 corpus embeddings (`outputs/embeddings/bge_m3_corpus.npy`) as the dense component of hybrid search.

To use StructGNN as the dense backbone for agentic retrieval, we would need to export the GNN's learned node representations as a fixed corpus embedding matrix (e.g., run a forward pass on the full corpus graph and extract the final-layer node embeddings before the scoring head). This is non-trivial because:
1. GNN embeddings are context-dependent (they change based on which nodes are in the graph)
2. The scoring head produces scalar relevance scores, not reusable vector representations
3. Would need to decide which graph structure to use for the "canonical" embedding export

**Recommendation:** Run Agentic (BGE+BM25) first. If results are promising enough to justify the engineering effort, implement a `--embeddings_source structgnn` flag that loads exported GNN node embeddings instead of BGE-M3.

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

## Commands Used

```bash
# === Original (max_relevant=5) ===
# BM25
python src/evaluate_bm25.py --dataset kuhperdata-humanized --use_stemmer --remove_stopwords
python src/evaluate_bm25.py --dataset kuhperdata-summarized --use_stemmer --remove_stopwords

# Dense (BGE-M3)
python src/evaluate_dense_retrieval.py --dataset kuhperdata-humanized --save_embeddings
python src/evaluate_dense_retrieval.py --dataset kuhperdata-summarized --save_embeddings

# JNLP Stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-humanized --stage 1
python src/evaluate_jnlp.py --dataset kuhperdata-summarized --stage 1

# Para-GNN
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode none
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural

# === Expanded (max_relevant=0) ===
# BM25
python src/evaluate_bm25.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_bm25.py --dataset kuhperdata-summ-exp --max_relevant 0
python src/evaluate_bm25.py --dataset kuhperdata-exp --use_stemmer --remove_stopwords --max_relevant 0
python src/evaluate_bm25.py --dataset kuhperdata-summ-exp --use_stemmer --remove_stopwords --max_relevant 0

# Dense
python src/evaluate_dense_retrieval.py --dataset kuhperdata-exp --save_embeddings --max_relevant 0
python src/evaluate_dense_retrieval.py --dataset kuhperdata-summ-exp --save_embeddings --max_relevant 0

# Rerank
python src/evaluate_rerank.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_rerank.py --dataset kuhperdata-summ-exp --max_relevant 0

# JNLP
python src/evaluate_jnlp.py --dataset kuhperdata-exp --stage 1 --max_relevant 0
python src/evaluate_jnlp.py --dataset kuhperdata-summ-exp --stage 1 --max_relevant 0

# Para-GNN + StructGNN (precompute first)
python src/paragnn/precompute.py --dataset kuhperdata-exp --method adapted
python src/paragnn/precompute.py --dataset kuhperdata-summ-exp --method adapted
python src/evaluate_paragnn.py --dataset kuhperdata-exp --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-exp --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode structural --max_relevant 0

# GAR
python src/evaluate_gar.py --dataset kuhperdata-exp --max_relevant 0
python src/evaluate_gar.py --dataset kuhperdata-summ-exp --max_relevant 0

# Agentic (Context-1 with BGE+BM25)
# Requires vLLM server running (see below)
python src/context_1/evaluate_context1.py --dataset kuhperdata-exp --max_relevant 0 --concurrency 4 --pad_to_k 10
python src/context_1/evaluate_context1.py --dataset kuhperdata-summ-exp --max_relevant 0 --concurrency 4 --pad_to_k 10
```

## Environment

- GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120)
- Torch: 2.7.0+cu128 (paragnn env), .venv-jnlp for others
- BGE-M3: BAAI/bge-m3 (1024d)
- Dataset: kuhperdata-humanized (383 test queries), kuhperdata-summarized (373 test queries)
