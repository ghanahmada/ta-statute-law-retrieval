# Planning: Query Humanization for KUHPerdata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Date** | 2026-03-05 |
| **Related HLD Section** | 2.3 Future Work: Query Humanization |

---

## 1. Problem Statement

### 1.1 The Query Realism Gap

KUHPerdata queries are LLM-generated summaries of full court decisions, averaging ~1,900 characters. Real legal information seekers — lawyers, judges, law students, or the general public — type much shorter queries. This mismatch means:

- Retrieval methods benchmarked on long queries may not reflect real-world performance
- Models may overfit to the dense, information-rich query format that doesn't exist in practice
- Cross-dataset comparison is skewed: STARD queries average 27 words, while KUHPerdata queries are 50-100x longer

### 1.2 Evidence from STARD (Su et al., 2024)

The STARD paper (arXiv:2406.15313) demonstrates this gap empirically. Their dataset uses real non-professional queries from China's 12348 Legal Service Website (avg 27 words). Key findings:

- **Dense-CAIL** (trained on court-derived queries) achieved only **R@100=0.328** on real queries — worse than BM25 (0.649). Training on court-style queries does not transfer to real queries.
- **SAILER** (legal-domain PLM pre-trained on long formal text) scored R@100=0.567 — worse than generic RoBERTa (0.663). Long-text pre-training hurts on short informal queries.
- **Dense-STARD** (trained on actual non-professional queries) reached R@100=0.907, a 2.8x improvement over Dense-CAIL.

The distributional gap between professional legal language and real user queries is not merely an inconvenience — it fundamentally changes which methods work.

### 1.3 Why Naive Shortening Breaks Relevance

KUHPerdata's relevance judgments (qrels) are derived from explicit statute citations in court decisions. The full query text contains the complete legal reasoning chain that connects the factual situation to each cited article. Consider a concrete scenario:

A court decision cites both **Pasal 1365** (perbuatan melawan hukum / unlawful acts) and **Pasal 1367** (tanggung jawab atas perbuatan orang lain / vicarious liability). The full LLM-summarized query contains:
- Facts establishing an unlawful act occurred (supports Pasal 1365)
- Facts establishing the perpetrator acted under someone else's authority (supports Pasal 1367)

If an LLM naively shortens this query, it may retain facts for Pasal 1365 but drop facts for Pasal 1367. The result:
- Pasal 1367 remains in qrels as "relevant"
- The shortened query no longer contains any signal pointing to it
- **The evaluation becomes unsound**: false positive relevance labels poison all metrics

This is not a hypothetical concern. The SIGIR-AP 2023 paper (Zhou et al., arXiv:2312.03494) on legal query content selection found that human-annotated salient content retains only **19% of the original query** — meaning 81% of query text is non-salient for retrieval. But which 19% matters depends entirely on which articles are relevant.

---

## 2. Related Work

### 2.1 STARD — Real Non-Professional Queries (Su et al., 2024)

**Source**: arXiv:2406.15313

STARD collects real queries from China's government legal consultation platform. Their 3-step annotation pipeline (Recall → Decompose → Filter) mirrors legal syllogistic reasoning:

1. **Recall**: Annotators narrow from broad legal categories to specific statute domains
2. **Decompose**: Informal "life facts" are mapped to formal "legal facts" (e.g., "bitten by pet" → "domestic animals" + "causing damage")
3. **Filter**: Intersect decomposed legal facts with recalled statutes → minimal comprehensive set

Key insight: STARD doesn't shorten existing queries — they start from naturally short queries and annotate relevance. Their annotation framework is generalizable to other civil law systems (German, French, Japanese), which includes the Dutch-derived Indonesian system.

**Relevance to KUHPerdata**: STARD's annotation pipeline could serve as a validation framework — after shortening queries, we can verify that the shortened query still implies the same legal facts that connect to each relevant article.

### 2.2 Query Content Selection with LLMs (Zhou et al., SIGIR-AP 2023)

**Source**: arXiv:2312.03494

This paper directly addresses our problem: long legal case queries → identify salient content → shorter query for retrieval. Tested on LeCaRD (107 queries, Chinese criminal cases).

**Three reformulation methods** (all zero-shot GPT-3.5-turbo):
1. **Keyword extraction**: Avg 67 chars, InfoR=1.42, 34% overlap with gold annotations
2. **Key sentence extraction**: Avg 219 chars, InfoR=1.07, 52% overlap
3. **Summarization**: Avg 121 chars, InfoR=1.41, 44% overlap

**InfoR metric**: `InfoR(Q, U, A) = Overlap(U, A) * (|Q| / |A|)` — measures salient content density in the reformulated query. Values >1 mean higher concentration than the original.

**Results** (BERT-PLI model):
| Query Type | MAP | P@5 | NDCG@10 |
|---|---|---|---|
| Original (495 chars) | 52.55% | 43.36% | 77.75% |
| Summary (121 chars) | **58.24%** | **48.03%** | **81.33%** |
| Gold annotation (72 chars) | 59.80% | 50.02% | 81.39% |

**Critical finding**: Summarized queries (24% of original length) **outperform** full-length originals. Removing noise improves retrieval. The summary results approach gold-standard human annotations (81.33% vs 81.39% NDCG@10).

**Salient content definition**: Based on China's Four Elements Theory (Subject, Object, Conduct, Mental State). Not all elements have equal retrieval value — the subject identity rarely matters; conduct and object are most salient.

### 2.3 KELLER — Knowledge-Guided Case Reformulation (Deng et al., 2024)

**Source**: arXiv:2406.19760

KELLER decomposes long legal cases (avg 7,446 tokens) into structured "sub-facts" using a two-stage LLM pipeline:

1. **Extract**: LLM (Qwen-72B) identifies all crimes and law articles from the case
2. **Map**: A legal expert database validates crime-to-article mappings
3. **Summarize**: LLM generates concise sub-fact snippets (max 100 words each) guided by the validated crime-article pairs

**Key innovation**: Law articles serve as "high-level abstractions" that guide the summarization, ensuring legally relevant facts are preserved. Without this guidance, naive LLM summaries lose the structured decomposition needed for matching.

**Results** (LeCaRDv2):
| Method | MAP |
|---|---|
| SAILER | 60.62 |
| Naive summarization (no guidance) | 61.91 |
| **KELLER (knowledge-guided)** | **68.29** |

Removing knowledge-guided reformulation causes the **largest performance drop** (-9.3% MAP), confirming that unguided summarization loses critical matching signals.

**Ablation on information loss**: Naive concatenation of sub-facts into a single vector drops MAP from 68.29 to 63.35. Multi-vector representation (one per sub-fact) with MaxSim aggregation preserves fine-grained matching that single-vector compression destroys.

### 2.4 LegalMALR — Multi-Agent Statute Retrieval (2026)

**Source**: arXiv:2601.17692

LegalMALR uses a multi-agent system to generate diverse query reformulations for STARD:

- 6 specialized agents (clarify terminology, expose implicit conditions, decompose multi-issue queries, discover auxiliary provisions, repair semantic abnormalities)
- Iterative exploration: Planner agent selects which reformulation type to try next based on candidate pool coverage
- GRPO (reinforcement learning) stabilizes the stochastic LLM-driven process

**Results on STARD**: R@10=0.8195 (vs Dense-STARD 0.6061, BM25 0.3943). Demonstrates that multi-perspective query reformulation dramatically improves statute retrieval on short informal queries.

**Relevance to KUHPerdata**: The iterative, coverage-driven reformulation approach is relevant if we need to generate multiple query variants to ensure all relevant articles remain reachable.

---

## 3. Analysis: KUHPerdata-Specific Challenges

### 3.1 Query Structure

Each KUHPerdata query is an LLM summary of a court decision that cites KUH Perdata articles. The query contains:
- **Factual narrative**: What happened (the dispute, parties, events)
- **Legal reasoning**: How the court analyzed the facts under law
- **Procedural context**: Court proceedings, appeals, prior decisions

Only the factual narrative directly connects to relevant statute articles. Legal reasoning and procedural context are noise for retrieval purposes — but they constitute the majority of query text.

### 3.2 Multi-Article Relevance

KUHPerdata averages **3.20 relevant articles per query** (vs STARD's 1.76). This means:
- Each query contains facts supporting multiple different legal provisions
- Shortening must preserve facts for ALL relevant articles, not just the dominant one
- The risk of breaking query-to-article alignment scales with the number of relevant articles

### 3.3 Civil Law Structure

The KUH Perdata follows the Dutch-derived civil law structure (Buku → Bab → Bagian → Pasal). Articles are often invoked in clusters:
- Pasal 1320-1337 (syarat sah perjanjian / contract validity)
- Pasal 1365-1380 (perbuatan melawan hukum / tort)
- Pasal 1457-1540 (jual beli / sale and purchase)

A shortened query must preserve enough factual grounding to distinguish which cluster (and which specific articles within the cluster) is relevant.

---

## 4. Proposed Approach: Fact-Aware Query Shortening

### 4.1 Design Principles

Drawing from the literature:

1. **Knowledge-guided, not blind** (from KELLER): Use the known relevant articles as guidance during shortening, so the LLM knows which facts to preserve
2. **Summarization over extraction** (from SIGIR-AP): LLM summarization produces the best balance of information density and retrieval performance
3. **Validate alignment post-shortening** (from STARD): Verify that each relevant article still has supporting facts in the shortened query
4. **Preserve evaluation soundness**: The shortened queries must be valid for the existing qrels — no orphaned relevance judgments

### 4.2 Pipeline Overview

```
Original Query (1,900 chars)
        │
        ▼
┌─────────────────────────┐
│  Step 1: Fact Extraction │  LLM extracts discrete legal facts
│  (article-aware)         │  from the full query, guided by
│                          │  the relevant article texts
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Step 2: Fact-to-Article │  Map each extracted fact to the
│  Alignment Verification  │  article(s) it supports
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Step 3: Query Synthesis │  LLM composes a natural short
│  (target: 100-200 chars) │  query from the extracted facts,
│                          │  using non-professional language
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Step 4: Alignment Check │  Verify every article in qrels
│                          │  has ≥1 supporting fact in the
│                          │  shortened query
└───────────┬─────────────┘
            │
       ┌────┴────┐
       │ Pass?   │
       ├─Yes─────┤──► Final Short Query
       └─No──────┘──► Retry Step 3 with coverage constraint
```

### 4.3 Step 1: Article-Aware Fact Extraction

**Input**: Original query text + list of relevant article texts (from qrels + corpus)

**LLM Prompt** (conceptual):
```
Anda adalah ahli hukum perdata Indonesia. Diberikan sebuah ringkasan perkara
dan pasal-pasal KUH Perdata yang relevan, identifikasi fakta-fakta hukum
diskrit dalam ringkasan perkara yang mendukung relevansi setiap pasal.

Ringkasan Perkara: {query_text}

Pasal-pasal yang relevan:
{for each article: "Pasal {id}: {text}"}

Untuk setiap pasal, tuliskan fakta hukum spesifik dari ringkasan perkara yang
menjadikan pasal tersebut relevan. Format output:

Pasal {id}: [fakta hukum yang mendukung]
```

This is directly inspired by KELLER's approach of using law articles as "high-level abstractions" to guide extraction. The key difference: KELLER extracts crimes first then maps to articles; we already know the articles (from qrels) and extract facts that justify them.

**Why article-aware**: The SIGIR-AP paper showed that unguided extraction retains only 34-52% of gold-standard salient content. By telling the LLM which articles are relevant, we direct its attention to the facts that matter for each article.

### 4.4 Step 2: Alignment Verification

For each relevant article in qrels, verify that at least one extracted fact maps to it. If an article has no supporting fact:
- The original query may not contain explicit support for that article (possible annotation noise)
- Or the LLM missed it → re-extract with stronger guidance for that specific article

### 4.5 Step 3: Short Query Synthesis

**Input**: Extracted facts (from Step 1)

**LLM Prompt** (conceptual):
```
Anda adalah warga biasa yang membutuhkan konsultasi hukum. Berdasarkan
fakta-fakta hukum berikut, tuliskan pertanyaan konsultasi hukum yang singkat
dan natural (100-200 karakter) menggunakan bahasa sehari-hari, TANPA
menyebutkan pasal atau istilah hukum teknis.

Fakta-fakta: {extracted_facts}

Pertanyaan konsultasi:
```

The prompt explicitly requests:
- **Non-professional language** (inspired by STARD's real-world query characteristics)
- **No legal terminology** (to simulate realistic queries)
- **Target length 100-200 characters** (~10-20x shorter than originals)
- **All facts must be represented** (to maintain alignment)

### 4.6 Step 4: Coverage Verification

Automated check: embed the shortened query and each relevant article, compute a lightweight relevance signal (e.g., BGE-M3 cosine or keyword overlap), flag any article that falls below a threshold. This catches cases where Step 3 inadvertently dropped a fact.

### 4.7 Output Format

The humanized dataset extends the existing BEIR structure:

```
data/kuhperdata/
├── corpus.jsonl              # unchanged
├── queries.jsonl             # original long queries
├── queries_humanized.jsonl   # new short queries (same _id mapping)
├── qrels_train.tsv           # unchanged (same relevance judgments)
├── qrels_test.tsv            # unchanged
└── dataset_stats.json        # updated with humanized query stats
```

Same query IDs, same qrels — only query text changes. This enables direct comparison: run the same retrieval method on both query sets to measure the impact of query length.

---

## 5. Experimental Design

### 5.1 Research Questions

1. **RQ1**: How much does query shortening degrade retrieval performance across methods (BM25, BGE-M3, JNLP Stage 1)?
2. **RQ2**: Which retrieval methods are most robust to short queries? (Do method rankings change between long and short queries?)
3. **RQ3**: Does the fact-aware shortening approach preserve more retrieval quality than naive summarization?

### 5.2 Query Variants

All experiments compare retrieval performance across query variants using the **same qrels** (same relevant articles per query). This isolates the effect of query formulation.

| Variant | Description | Article-Aware? | Avg Length (est.) |
|---|---|---|---|
| **Original (baseline)** | LLM-summarized court decisions — existing `queries.jsonl` | N/A (ground truth) | ~1,900 chars |
| **Naive summary** | LLM prompted to "summarize in 1-2 sentences" without article guidance | No | ~150-300 chars |
| **Fact-aware summary** | Full pipeline (Steps 1-4 above) with article-guided extraction | Yes | ~100-200 chars |

The original queries serve as the **performance ceiling** — the maximum retrieval quality achievable when the query contains all information from the court decision. The gap between original and each shortening strategy measures information loss.

### 5.3 Evaluation Matrix

Each retrieval method is evaluated on each query variant:

| | Original (long) | Naive Summary | Fact-Aware Summary |
|---|---|---|---|
| **BM25** | existing result | new | new |
| **BGE-M3 Dense** | existing result | new | new |
| **JNLP Stage 1** | existing result | new | new |
| **SAILER** | in progress | new | new |

This produces a 4×3 result matrix. Key comparisons:
- **Rows** (fixed method, across query types): measures method robustness to query shortening
- **Columns** (fixed query type, across methods): measures which method is best for each query length
- **Diagonal insight**: does the best method for long queries remain the best for short queries?

### 5.4 Alignment Quality Metric

Define **Article Coverage Rate (ACR)**: for each shortened query, what fraction of its qrels articles are still retrievable in the top-K by the best method?

```
ACR@K = (# relevant articles retrieved in top-K) / (# total relevant articles in qrels)
```

A high ACR means the shortening preserved enough signal for all articles. A low ACR pinpoints which articles lost their evidential link.

Additionally, compare ACR between naive and fact-aware summaries to validate that article-guided shortening preserves more relevance links.

---

## 6. Implementation Notes

### 6.1 LLM Choice

- Use GPT-4o-mini or similar for cost-effective batch processing (~1,368 queries)
- Indonesian language capability required
- Zero-shot prompting (following SIGIR-AP approach — no fine-tuning needed)

### 6.2 Scope

- Apply to ALL 1,368 queries (both train and test splits)
- Train split humanized queries can be used for training methods adapted to short queries
- Test split humanized queries are the evaluation target

### 6.3 Cost Estimate

At ~1,900 chars input + ~500 chars output per query, ~1,368 queries:
- Step 1 (extraction): ~1,368 calls
- Step 3 (synthesis): ~1,368 calls
- Step 4 retries: ~10-20% of queries → ~200 calls
- Total: ~3,000 LLM calls — modest cost with GPT-4o-mini

---

## 7. References

1. Su, W., Hu, Y., Xie, A., et al. (2024). "STARD: A Chinese Statute Retrieval Dataset with Real Queries Issued by Non-professionals." arXiv:2406.15313.
2. Zhou, Y., Huang, X., Wu, Y. (2023). "Boosting Legal Case Retrieval by Query Content Selection with Large Language Models." SIGIR-AP 2023. arXiv:2312.03494.
3. Deng, C., Mao, K., Dou, Z. (2024). "KELLER: Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation." arXiv:2406.19760.
4. LegalMALR (2026). "Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval." arXiv:2601.17692.
