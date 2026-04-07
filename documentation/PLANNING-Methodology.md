# Planning: Methodology Innovation for EMNLP 2026

## Status: Research Phase — Analyzing Prior Work to Identify Innovation Gap

---

## 1. What We Have (Baseline)

Para-GNN (adapted from IL-PCSR) with alpha grid search achieves:

| Dataset | Lang | MRR@10 | vs JNLP S1 |
|---|---|---|---|
| KUHPerdata-humanized | ID | 0.456 | +24% |
| KUHPerdata-summarized | ID | 0.458 | +36% |
| BSARD | FR | 0.493 | +50% |
| STARD | ZH | 0.527 | +94% |

This is strong but **not novel** — it's IL-PCSR's method with our engineering improvements (alpha grid search, BGE-M3 embeddings, early stopping).

---

## 2. Critical Analysis of Related Methods

### LegalMALR (Li et al., 2026) — Best on STARD (0.737 MRR)

**What**: 6 specialized LLM agents rewrite queries from different legal perspectives + GRPO training + Qwen-Max reranker.

**Key design choice**: Multi-agent reformulation. Each agent handles a different type of fact-to-norm mismatch:
- Single-Element Rewrite: vocabulary bridging (user says "fired" → statute says "termination")
- Supplementary-Element Rewrite: adds MISSING conditions (user asks about divorce but doesn't mention child custody, which the statute requires)
- Multi-Element Decomposition: splits compound queries
- Supportive-Law Rewrite: targets auxiliary/interpretive provisions
- Semantic-Abnormality Repair: handles doctrinal confusion

**Why they need 6 agents**: They DON'T prove each agent is independently necessary. No ablation removing individual agents. No correlation analysis between agent outputs (like QDER's ρ=-0.022). This is the weakness — it's engineering, not science.

**What we can learn**: The Supplementary-Element agent is the most interesting. It addresses our Tatbestand coverage problem (27% of query-statute pairs miss some statute conditions). The idea of ADDING missing legal elements to the query is powerful.

**What we shouldn't copy**: The multi-agent architecture is expensive (2+ LLM calls per query) and not proven to be better than a single well-prompted LLM. GRPO training requires 8xA100.

### SyLeR (Su et al., CIKM 2025) — Legal QA, not retrieval

**What**: Builds a legal knowledge tree (statutes → linked precedent cases), retrieves via hierarchical traversal, then generates syllogistic reasoning (major premise → minor premise → conclusion).

**Key design choice**: The knowledge tree structure — statutes as first layer, precedent cases as second layer, linked by semantic similarity. Retrieval starts from statutes, then drills down to relevant cases.

**Why they need the tree**: Legal reasoning follows syllogistic structure. The major premise (statute + cases) must be retrieved first, then the minor premise (case facts) is matched against it. This ordering reflects how lawyers actually think.

**What we can learn**: 
- The hierarchical retrieval idea: retrieve at the statute GROUP level first (which legal domain?), then drill down to specific articles. This addresses our "domain" perspective.
- The RL reward function design: separate rewards for each component (major premise alignment with statute, minor premise alignment with query, conclusion accuracy). This could inform our training signal.

**What we shouldn't copy**: SyLeR is a QA system, not a retrieval system. It retrieves K_L=1 statute (single retrieval), which doesn't work for our multi-article retrieval setting (avg 2-3 relevant articles per query).

### G-DSR (Louis et al., EACL 2023) — Graph on statutes, tested on BSARD

**What**: Models legislation structure as a graph (articles connected by same-chapter, cross-reference, hierarchical relationships), then uses GNN to propagate information between related articles before dense retrieval.

**Key design choice**: The graph is on the STATUTE SIDE, not query side. Articles that are structurally related (same section, cross-referenced) share information through the GNN. This enriches article representations with contextual information from neighboring articles.

**Why they need it**: A single article's text may be ambiguous or incomplete without context from related articles. For example, "Article 1365 (tort)" makes more sense when combined with "Article 1366 (negligence)" and "Article 1367 (vicarious liability)".

**What we can learn**:
- Statute-side graph structure. Our Para-GNN only has sentence→document edges. G-DSR's article→article edges could complement this.
- Cross-reference edges specifically. Our earlier analysis showed that cross-references exist in 17% (KUHPerdata) to 54% (IL-PCSR) of statutes.

**What we shouldn't copy**: G-DSR's graph is query-agnostic — it improves ALL statute representations equally regardless of the query. QDER showed that query-specific adaptation is more powerful.

### Structured Legal RAG (Zheng et al., 2025) — Knowledge Graph + Citation Authority

**What**: Combines semantic similarity with factor-level legal knowledge graph and citation network authority (PageRank) for copyright fair use.

**Key insight**: Legal relevance ≠ semantic similarity. Citation authority is a genuinely independent signal.

**What we can learn**: The idea that AUTHORITY (how established a statute is, how often it's cited/applied) is independent from CONTENT similarity. This maps to our "abstraction" perspective — general principles (Pasal 1365, cited 494 times) vs specific rules (Pasal 1964, cited 2 times).

---

## 3. Gap Analysis: What No One Has Done

| Capability | LegalMALR | SyLeR | G-DSR | IL-PCSR | Ours? |
|---|---|---|---|---|---|
| Multilingual (4+ languages) | No (ZH only) | No (ZH+FR) | No (FR only) | No (EN only) | **Yes** |
| Proved signal independence | No | No | No | No | **Partially (3 perspectives)** |
| Query-specific adaptation | Yes (reformulation) | No | No | Partial (GNN) | **?** |
| Statute structure exploitation | No | Tree structure | Graph structure | No | **?** |
| Learned combination (not hand-tuned) | Yes (GRPO) | Yes (RL) | No | Yes (alpha FFN) | **Alpha grid search** |
| No LLM at inference | No | No | Yes | Yes | **Yes (adapted)** |

**The gap**: No paper combines:
1. Query-side legal characteristic exploitation (like LegalMALR's reformulation)
2. Statute-side structural relationships (like G-DSR's graph)
3. Proven independence of signals (like QDER's methodology)
4. Across 4+ multilingual datasets

---

## 4. Methodology Directions (Step-by-Step Development)

### Direction A: Enhanced Para-GNN with Statute Structure

**Idea**: Add G-DSR-style statute-to-statute edges to our Para-GNN graph.

Current Para-GNN graph:
```
query_sentence → query_doc
statute_sentence → statute_doc
(no cross-document edges)
```

Enhanced graph:
```
query_sentence → query_doc
statute_sentence → statute_doc
statute_doc → related_statute_doc  (NEW: cross-reference edges)
statute_doc → same_section_statute_doc  (NEW: structural edges)
```

**What this tests**: Does statute context (neighboring articles) help retrieval?

**Empirical check — DONE** (KUHPerdata-humanized, 300 queries):

Co-relevant statutes (queries with 2+ relevant articles):
- 85.2% same book (same legal domain)
- 66.3% within 50 articles of each other
- Median distance: 18 articles — very close structural neighbors

Hard negatives (BM25 top-1 wrong):
- Only 56.6% same book
- Median distance: 297 articles — structurally far

Cross-references:
- Only 32 found in relevant statutes, 15.6% point to also-relevant → too sparse for edges

**Conclusion**: Proximity edges are strongly justified. Co-relevant statutes cluster tightly (median 18 apart), while hard negatives are far (median 297). Adding edges between statutes within ~50 articles would connect co-relevant articles without connecting hard negatives.

**Implementation plan**:
1. For each statute, add edges to statutes within N articles (N=50 based on 66.3% co-relevant coverage)
2. Edge feature = distance encoding (closer = stronger) or book membership embedding
3. Same EUGAT architecture — only graph structure changes
4. Test: does Para-GNN with proximity edges beat Para-GNN without?

### Direction B: Query Enrichment via Legal Characteristic Injection

**Idea**: Instead of LegalMALR's expensive multi-agent reformulation, use a single LLM call to extract legal characteristics and inject them into the query representation.

From our error analysis:
- 27% of query-statute pairs have partial Tatbestand coverage (query doesn't mention all statute conditions)
- The oracle experiment showed +5 legal terms → 99.5% BM25 hit rate

**Single-pass enrichment**: For each query, LLM extracts:
1. Legal domain (contract, tort, property, family)
2. Key legal terms the user SHOULD have used but didn't
3. Implicit conditions the statute requires

This enriched query gets encoded by BGE-M3 and used as the query node in Para-GNN.

**What this tests**: Does explicit legal term injection improve GNN performance?

**Empirical check needed**: Compare Para-GNN with original queries vs enriched queries. If enriched queries have higher vocabulary overlap with relevant statutes, the GNN should perform better.

### Direction C: Multi-Signal Fusion with Proved Independence

**Idea**: Following QDER's methodology, combine Para-GNN scores with a second provably independent signal using learned weights.

From our multi-representation analysis:
- Dense ↔ BM25 correlation: ρ=0.461 (moderately independent)
- Title ↔ Dense correlation: ρ=0.073 (independent but not discriminative)
- Domain ↔ Function ↔ Abstraction: all independent (ρ<0.05)

**Architecture**:
```
Signal 1: Para-GNN score (graph-enhanced semantic)
Signal 2: [NEW independent signal — TBD]
Signal 3: BM25 score (lexical)
Fusion: learned per-query alpha (or bilinear like QDER)
```

**What this tests**: Does adding a provably independent signal improve over Para-GNN + BM25?

**Empirical check needed**: First identify what Signal 2 should be, then measure its independence from Para-GNN scores (ρ < 0.1).

---

## 5. Recommended Path Forward

**Step 1**: Finish Para-GNN evaluation across all datasets (adapted + full) → establishes baseline
**Step 2**: Run Direction A experiment (add statute-to-statute edges) → cheap to test, reuses existing code
**Step 3**: Run Direction B experiment (LLM query enrichment) → tests whether better queries help the GNN
**Step 4**: Based on Step 2+3 results, design Direction C (multi-signal fusion with independence proof)

Each step produces a concrete experiment result and informs the next step. No speculative architecture — every component is empirically justified.

---

## 6. Open Questions

1. **Should we use the RR-labeled queries (full method) for the enrichment experiment?** The full method showed marginal gains (+0.01 on humanized, +0.03 on summarized). Maybe RR labels aren't the right decomposition — legal term injection might be more impactful.

2. **Can we use G-DSR's statute graph structure with our data?** We need to check if KUHPerdata articles have cross-references that can be parsed. Our analysis showed 17% of KUHPerdata statutes have cross-references.

3. **What is the right "second signal" for Direction C?** Candidates:
   - Cross-encoder reranker score (already in codebase)
   - LLM enrichment similarity (from Direction B)
   - Statute citation frequency (dataset-specific, not generalizable)

Sources:
- [LegalMALR](https://arxiv.org/html/2601.17692)
- [SyLeR](https://arxiv.org/html/2504.04042v1)
- [G-DSR](https://github.com/maastrichtlawtech/gdsr)
- [Structured Legal RAG](https://arxiv.org/html/2505.02164v1)
- [UQLegalAI@COLIEE2025](https://arxiv.org/html/2505.20743v1)
