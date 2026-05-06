# Evaluation Results — Validated Expanded Qrels

All results on **test split** after Phase 1 (validation) + Phase 2 (split_test_to_val).  
Expanded datasets: `kuhperdata-exp` (humanized) and `kuhperdata-summ-exp` (summarized).  
`--max_relevant 0` throughout (no query filtering).

Updated: 2026-05-06.

## Test Split Info

| Dataset | Lang | Corpus | Test Queries | Qrels |
|---------|------|--------|-------------|-------|
| kuhperdata-exp | id | 2127 docs | 211 queries | validated expanded |
| kuhperdata-summ-exp | id | 2127 docs | 213 queries | validated expanded |
| bsard | fr | — | — | original |
| stard | zh | — | — | original |
| ilpcsr | en | — | — | original |

## KUHPerdata Results (Indonesian, Expanded Qrels)

| Method | Dataset | MRR@10 | Recall@10 | Precision@10 | Hit Rate |
|--------|---------|--------|-----------|--------------|----------|
| BM25 | exp | 0.0329 | 0.0434 | 0.0081 | 8.06% |
| BM25 | summ-exp | 0.0799 | 0.1137 | 0.0202 | 19.25% |
| GAR (bge) | exp | 0.0718 | 0.0656 | 0.0142 | 13.27% |
| GAR (bge) | summ-exp | 0.1709 | 0.1571 | 0.0319 | 25.35% |
| Dense (BGE-M3) | exp | 0.0634 | 0.0682 | 0.0152 | 12.3% |
| Dense (BGE-M3) | summ-exp | 0.0974 | 0.1212 | 0.0263 | 20.2% |
| Rerank (BGE) | exp | 0.0841 | 0.0863 | 0.0175 | 16.59% |
| Rerank (BGE) | summ-exp | 0.1850 | 0.1711 | 0.0362 | 29.11% |
| JNLP Stage 1 | exp | 0.5226 | 0.6500 | 0.1839 | 86.3% |
| JNLP Stage 1 | summ-exp | 0.4450 | 0.5195 | 0.1446 | 71.4% |
| Agentic v1 (Qwen3.6-27B, max_turns=5, BGE+BM25) | exp | 0.2927 | 0.2408 | 0.0611 | 41.23% |

**Agentic v1 agent stats** (exp, 211 queries): avg turns 4.9, avg selected 10.3, avg seen 48.4, avg read 4.5, avg time/query 230s.  
Prompt: basic statutory term abstraction instruction. **Baseline before hierarchy prompt.**

## Cross-Lingual Results (Original Qrels)

| Method | Dataset | Lang | MRR@10 | Recall@10 | Precision@10 | Hit Rate |
|--------|---------|------|--------|-----------|--------------|----------|
| BM25 | bsard | fr | — | — | — | — |
| BM25 | stard | zh | — | — | — | — |
| BM25 | ilpcsr | en | — | — | — | — |
| GAR | bsard | fr | — | — | — | — |
| GAR | stard | zh | — | — | — | — |
| GAR | ilpcsr | en | — | — | — | — |
| JNLP Stage 1 | bsard | fr | — | — | — | — |
| JNLP Stage 1 | stard | zh | — | — | — | — |
| JNLP Stage 1 | ilpcsr | en | — | — | — | — |
| Para-GNN | bsard | fr | — | — | — | — |
| Para-GNN | stard | zh | — | — | — | — |
| Para-GNN | ilpcsr | en | — | — | — | — |
| StructGNN | bsard | fr | — | — | — | — |
| StructGNN | stard | zh | — | — | — | — |
| StructGNN | ilpcsr | en | — | — | — | — |
| Agentic v1 (Qwen3.6-27B, flat, max_turns=5) | bsard | fr | 0.5393 | 0.4705 | 0.1242 | 65.83% |
| Agentic v1 (Qwen3.6-27B, flat, max_turns=7) | bsard | fr | 0.5354 | 0.4622 | 0.1308 | 65.00% |
| Agentic v1 (Qwen3.6-27B, flat, max_turns=5) | stard | zh | 0.6900 | 0.7329 | 0.1135 | 82.05% |
| Agentic v1 (Qwen3.6-27B, flat, max_turns=7) | stard | zh | 0.6839 | 0.7585 | 0.1167 | 83.33% |
| Agentic v2 (Qwen3.6-27B, hierarchy+gate) | bsard | fr | 0.4537 | 0.4083 | 0.1192 | 59.17% |
| Agentic v2 (Qwen3.6-27B, hierarchy+gate) | stard | zh | 0.6448 | 0.6672 | 0.1038 | 77.56% |

## Pending

**KUHPerdata (exp + summ-exp):**
- [ ] BM25 (stem+stop)
- [x] GAR (bge)
- [x] Dense (BGE-M3)
- [x] Rerank
- [x] JNLP Stage 1
- [ ] Para-GNN
- [ ] StructGNN
- [ ] Agentic (Context-1)

**Agentic v1 agent stats** (bsard t5, 120 queries): avg turns 4.9, avg seen 37.6, avg read 2.9, avg time/query 423s.  
**Agentic v1 agent stats** (bsard t7, 120 queries): avg turns 6.7, avg seen 44.1, avg read 3.8, avg time/query 678s. 1 timeout (q231).  
**Agentic v1 agent stats** (stard t5, 156 queries): avg turns 4.9, avg seen 39.3, avg read 3.5, avg time/query 344s.  
**Agentic v1 agent stats** (stard t7, 156 queries): avg turns 6.4, avg seen 45.4, avg read 5.2, avg time/query 538s. 1 timeout (q1247).  
**Agentic v2 agent stats** (bsard, 120 queries): avg turns 4.9, avg seen 37.7, avg read 3.2, avg time/query 269s, avg frames declared 2.8, avg frames covered 1.0, gate triggers 111, similarity rejections 5.  
**Agentic v2 agent stats** (stard, 156 queries): avg turns 5.0, avg seen 36.6, avg read 4.3, avg time/query 178s, avg frames declared 3.0, avg frames covered 1.7, gate triggers 125, similarity rejections 7.

> **Note:** Flat prompt (v1) outperforms hierarchy+gate (v2) on both cross-lingual datasets: +0.086 MRR on bsard, +0.045 on stard. Hierarchy overhead consumes turns that would otherwise be spent searching — with max_turns=5 the coverage table and frame declarations eat into the search budget. The L1-L4 scaffold likely helps more when turns are not a bottleneck (avg turns 4.9–5.0 = always turn-limited).  
> **max_turns=7 finding:** Extra turns improve Recall@10 (stard +0.026, bsard -0.008) but do not improve MRR@10. Avg turns reaches 6.4–6.7 (not always hitting the limit), suggesting the agent finds its natural stopping point. Additional searches in turns 6–7 surface marginally relevant docs that pad Recall but don't displace the top ranked results, leaving MRR flat. Cost increases ~60%. **Conclusion: max_turns=5 is sufficient for MRR optimization; max_turns=7 only if Recall@10 is the target metric.**

**Cross-lingual (bsard, stard, ilpcsr):**
- [ ] BM25
- [ ] GAR
- [ ] JNLP Stage 1
- [ ] Para-GNN
- [ ] StructGNN
- [x] Agentic v1 (bsard, stard)
- [x] Agentic v2 (bsard, stard)
