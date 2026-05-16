# Benchmark Rerun — May 2026

All methods on expanded qrels (`kuhperdata-exp`, `kuhperdata-summ-exp`) and cross-lingual datasets (`bsard`, `ilpcsr`, `stard`).  
`--max_relevant 0` throughout. Commands follow `FULL-RUN-PIPELINE.md`.

Updated: 2026-05-15.

---

## Key Changes from Previous Eval (`EVAL-Expanded-Results.md`)

- Para-GNN / StructGNN: val carved from **train** (not test) — fixes alpha selection leakage
- Agentic: upgraded to Context-1 model with explicit CoT prompt (v3: reasoning required before every tool call)
- Agentic: `search_corpus` no longer excludes previously read docs
- Agentic: FinalAnswer-as-tool-call handled gracefully; empty response triggers retry
- All datasets re-pulled from canonical HF source

---

## Dataset Split Info

| Dataset | Lang | Corpus | Test Queries | GNN Val | Qrels |
|---------|------|--------|-------------|---------|-------|
| kuhperdata-exp | id | 2127 | 211 | from train | validated expanded |
| kuhperdata-summ-exp | id | 2127 | 213 | from train | validated expanded |
| bsard | fr | 22,633 | 222 | from train | original |
| ilpcsr | en | 936 | 627 | from train | original |
| stard | zh | 55,348 | 308 | from train | original |

Split logic: Para-GNN + StructGNN use val (carved from train) for alpha grid search. All other methods use full test.

---

## KUHPerdata — Humanized (`kuhperdata-exp`, Indonesian)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | 0.0329 | 0.0434 | 0.0081 | 8.06% | 211 queries |
| Dense (BGE-M3) | 0.0635 | 0.0682 | 0.0152 | 12.30% | 211 queries |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | 0.5159 | 0.6569 | 0.1853 | 85.80% | |
| Para-GNN | 0.5111 | 0.5934 | 0.1583 | 83.4% | alpha=0.9 from val |
| StructGNN | 0.5040 | 0.5745 | 0.1512 | 83.9% | alpha=0.8 from val |
| Agentic Flat (BGE+BM25) | 0.1136 | 0.1163 | 0.0256 | 21.3% | flat ablation, no hierarchy/gate/guard |
| Agentic (Context-1 + StructGNN) | 0.5134 | 0.5866 | 0.1550 | 82.5% | CoT v3, flat, 5 turns avg |

## KUHPerdata — Summarized (`kuhperdata-summ-exp`, Indonesian)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | 0.0799 | 0.1137 | 0.0202 | 19.25% | 213 queries |
| Dense (BGE-M3) | 0.0974 | 0.1212 | 0.0263 | 20.20% | 213 queries |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | 0.4239 | 0.5385 | 0.1479 | 73.24% | |
| Para-GNN | 0.4394 | 0.4331 | 0.1122 | 71.4% | alpha=0.8 from val |
| StructGNN | 0.4795 | 0.5255 | 0.1423 | 75.1% | alpha from val |
| Agentic (Context-1 + BGE+BM25) | — | — | — | — | CoT v3, max_turns=10 |
| Agentic (Context-1 + StructGNN) | — | — | — | — | CoT v3, max_turns=10 |

---

## BSARD (French)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | 0.2488 | 0.2664 | 0.0716 | 42.34% | 222 queries |
| Dense (BGE-M3) | 0.2921 | 0.3142 | 0.0865 | 51.40% | 222 queries |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | 0.3031 | 0.2797 | 0.1153 | 41.00% | |
| Para-GNN | 0.4412 | 0.4677 | 0.1604 | 65.3% | alpha from val |
| StructGNN | 0.4917 | 0.5197 | 0.1721 | 71.2% | alpha from val |

## IL-PCSR (English)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | 0.1570 | 0.1002 | 0.0335 | 25.52% | 627 queries |
| Dense (BGE-M3) | 0.1259 | 0.0892 | 0.0306 | 23.60% | `--max_length 8192 --batch_size 8` |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | 0.0365 | 0.0230 | 0.0132 | 10.5% | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |

## STARD (Chinese)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | 0.3398 | 0.4272 | 0.0643 | 53.25% | 308 queries |
| Dense (BGE-M3) | 0.4645 | 0.5484 | 0.0854 | 64.94% | 308 queries |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | 0.2531 | 0.3827 | 0.0562 | 47.73% | |
| Para-GNN | 0.5197 | 0.6285 | 0.0958 | 73.1% | alpha from val |
| StructGNN | 0.5264 | 0.6148 | 0.0935 | 73.4% | alpha from val |
| Agentic Flat (BGE+BM25) | 0.6744 | 0.7191 | 0.1097 | 81.8% | flat ablation, 308 queries, 5.0 turns avg |

---

## GNN Alpha Grid Search

| Model | Dataset | Best Alpha | Val MRR | Test MRR |
|-------|---------|-----------|---------|----------|
| Para-GNN | kuhperdata-exp | 0.9 | 0.7109 | 0.5111 |
| Para-GNN | kuhperdata-summ-exp | 0.8 | 0.3770 | 0.4394 |
| Para-GNN | bsard | — | — | 0.4412 |
| Para-GNN | ilpcsr | — | — | — |
| Para-GNN | stard | — | — | 0.5197 |
| StructGNN | kuhperdata-exp | 0.8 | 0.7306 | 0.5040 |
| StructGNN | kuhperdata-summ-exp | — | — | 0.4795 |
| StructGNN | bsard | — | — | 0.4917 |
| StructGNN | ilpcsr | — | — | — |
| StructGNN | stard | — | — | 0.5264 |

---

## Agentic Agent Stats

| Run | Dataset | Queries | Avg Turns | Avg Seen | Avg Read | Avg Selected | Avg Time/q |
|-----|---------|---------|-----------|----------|----------|--------------|------------|
| Context-1 + BGE+BM25 | kuhperdata-exp | — | — | — | — | — | — |
| Context-1 + BGE+BM25 | kuhperdata-summ-exp | — | — | — | — | — | — |
| Context-1 + StructGNN | kuhperdata-exp | — | — | — | — | — | — |
| Context-1 + StructGNN | kuhperdata-summ-exp | — | — | — | — | — | — |
| Flat (BGE+BM25) | stard | 308 | 5.0 | 50.3 | 3.1 | 10.2 | 154.9s |

---

## Pending Checklist

**KUHPerdata (exp + summ-exp):**
- [x] BM25
- [x] Dense (BGE-M3)
- [ ] GAR (bge)
- [ ] Rerank (BGE)
- [x] JNLP Stage 1
- [x] Para-GNN
- [x] StructGNN
- [ ] Agentic Context-1 + BGE+BM25
- [ ] Agentic Context-1 + StructGNN

**Cross-lingual (bsard, ilpcsr, stard):**
- [x] BM25
- [x] Dense (BGE-M3)
- [ ] GAR (bge)
- [ ] Rerank (BGE)
- [x] JNLP Stage 1
- [ ] Para-GNN
- [ ] StructGNN

---

## Environment

- GPU: RTX A6000 (Ada Lovelace, SM 8.9) — DGL requires non-Blackwell
- CUDA driver: ≥ 13.0 | Toolkit: 12.4
- Torch: 2.7.0+cu128 (paragnn conda env)
- BGE-M3: BAAI/bge-m3 (1024d)
- vLLM model: QuantTrio/Qwen3.6-27B-AWQ served as `qwen3.6-27b`
- Agentic prompt: v3 (CoT required, exclude-after-read removed)
