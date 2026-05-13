# Benchmark Rerun — May 2026

All methods on expanded qrels (`kuhperdata-exp`, `kuhperdata-summ-exp`) and cross-lingual datasets (`bsard`, `ilpcsr`, `stard`).  
`--max_relevant 0` throughout. Commands follow `FULL-RUN-PIPELINE.md`.

Updated: 2026-05-14.

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
| kuhperdata-exp | id | 2127 | ~242 | from train | validated expanded |
| kuhperdata-summ-exp | id | 2127 | ~242 | from train | validated expanded |
| bsard | fr | 22,633 | ~60 | from train | original |
| ilpcsr | en | 936 | ~627 | from train | original |
| stard | zh | 55,348 | ~78 | from train | original |

Split logic: Para-GNN + StructGNN use val (carved from train) for alpha grid search. All other methods use full test.

---

## KUHPerdata — Humanized (`kuhperdata-exp`, Indonesian)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | — | — | — | — | |
| Dense (BGE-M3) | — | — | — | — | |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | — | — | — | — | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |
| Agentic (Context-1 + BGE+BM25) | — | — | — | — | CoT v3, max_turns=10 |
| Agentic (Context-1 + StructGNN) | — | — | — | — | CoT v3, max_turns=10 |

## KUHPerdata — Summarized (`kuhperdata-summ-exp`, Indonesian)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | — | — | — | — | |
| Dense (BGE-M3) | — | — | — | — | |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | — | — | — | — | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |
| Agentic (Context-1 + BGE+BM25) | — | — | — | — | CoT v3, max_turns=10 |
| Agentic (Context-1 + StructGNN) | — | — | — | — | CoT v3, max_turns=10 |

---

## BSARD (French)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | — | — | — | — | |
| Dense (BGE-M3) | — | — | — | — | |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | — | — | — | — | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |

## IL-PCSR (English)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | — | — | — | — | |
| Dense (BGE-M3) | — | — | — | — | `--max_length 8192 --batch_size 8` |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | — | — | — | — | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |

## STARD (Chinese)

| Method | MRR@10 | Recall@10 | Precision@10 | Hit Rate | Notes |
|--------|--------|-----------|--------------|----------|-------|
| BM25 | — | — | — | — | |
| Dense (BGE-M3) | — | — | — | — | |
| GAR (bge) | — | — | — | — | |
| Rerank (BGE) | — | — | — | — | |
| JNLP Stage 1 | — | — | — | — | |
| Para-GNN | — | — | — | — | alpha from val |
| StructGNN | — | — | — | — | alpha from val |

---

## GNN Alpha Grid Search

| Model | Dataset | Best Alpha | Val MRR | Test MRR |
|-------|---------|-----------|---------|----------|
| Para-GNN | kuhperdata-exp | — | — | — |
| Para-GNN | kuhperdata-summ-exp | — | — | — |
| Para-GNN | bsard | — | — | — |
| Para-GNN | ilpcsr | — | — | — |
| Para-GNN | stard | — | — | — |
| StructGNN | kuhperdata-exp | — | — | — |
| StructGNN | kuhperdata-summ-exp | — | — | — |
| StructGNN | bsard | — | — | — |
| StructGNN | ilpcsr | — | — | — |
| StructGNN | stard | — | — | — |

---

## Agentic Agent Stats

| Run | Dataset | Queries | Avg Turns | Avg Seen | Avg Read | Avg Selected | Avg Time/q |
|-----|---------|---------|-----------|----------|----------|--------------|------------|
| Context-1 + BGE+BM25 | kuhperdata-exp | — | — | — | — | — | — |
| Context-1 + BGE+BM25 | kuhperdata-summ-exp | — | — | — | — | — | — |
| Context-1 + StructGNN | kuhperdata-exp | — | — | — | — | — | — |
| Context-1 + StructGNN | kuhperdata-summ-exp | — | — | — | — | — | — |

---

## Pending Checklist

**KUHPerdata (exp + summ-exp):**
- [ ] BM25
- [ ] Dense (BGE-M3)
- [ ] GAR (bge)
- [ ] Rerank (BGE)
- [ ] JNLP Stage 1
- [ ] Para-GNN
- [ ] StructGNN
- [ ] Agentic Context-1 + BGE+BM25
- [ ] Agentic Context-1 + StructGNN

**Cross-lingual (bsard, ilpcsr, stard):**
- [ ] BM25
- [ ] Dense (BGE-M3)
- [ ] GAR (bge)
- [ ] Rerank (BGE)
- [ ] JNLP Stage 1
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
