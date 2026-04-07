# Research: Legal Characteristic-Guided Statute Retrieval

| Field | Value |
|-------|-------|
| **Date** | 2026-04-04 |
| **Scope** | Design a retrieval method grounded in legal characteristics, validated on 4 multilingual datasets |
| **Supervisor Direction** | "Cari legal characteristic untuk nyusun metode retrieval SOTA" |
| **Inspiration** | QDER (SIGIR 2025) dual-channel architecture, IL-PCSR (EMNLP 2025) legal-specific methods |

---

## 1. Research Reasoning Chain

### Step 1: What causes retrieval failure? (Empirical, not theoretical)

**Approach:** Instead of theorizing about legal characteristics, we measured what actually predicts BM25 failure across 4 datasets.

**Method:** Split test queries into BM25 hit@10 vs miss@10, compare characteristics.

**Finding:** Vocabulary overlap is the ONLY consistent failure predictor. Query length, sentence count, question format have near-zero predictive power.

| Dataset | Hit Rate | Overlap (Hit) | Overlap (Miss) | Gap |
|---------|----------|---------------|----------------|-----|
| KUHPerdata-humanized | 10.7% | 3.91 | 2.18 | +1.73 |
| KUHPerdata-summarized | 17.7% | 7.39 | 4.76 | +2.63 |
| BSARD | 42.3% | 2.43 | 1.11 | +1.32 |
| STARD | 53.2% | 13.71 | 7.40 | +6.31 |

**Why this matters:** Tells us the method must address vocabulary gap, not query structure.

### Step 2: What KIND of vocabulary gap? (Deeper than "no overlap")

**Approach:** For KUHPerdata-humanized, categorize all words into query-only, doc-only, and shared.

**Finding:** Only 160 shared words vs 1,375 query-only and 1,917 doc-only. Three gap types:

1. **Framing gap** — query words like "mengapa" (why), "apakah" (whether) that statutes never contain. These are question-construction artifacts.
2. **Concreteness gap** — users say "uang" (money), statutes say "harta" (property). Instance vs class.
3. **Register gap** — users say "membayar" (pay), statutes say "mewajibkan" (obligate). Everyday vs prescriptive.

**Cross-language validation:** Same 3 types confirmed in French (BSARD) and Chinese (STARD).

| Dataset | Query-only vocab | Doc-only vocab | Shared | Shared % |
|---------|-----------------|----------------|--------|----------|
| KUHPerdata | 1,375 | 1,917 | 160 | 4.6% |
| BSARD | 597 | 10,266 | 236 | 2.1% |
| STARD | 933 | 2,838 | 513 | 12.0% |

**BUT:** These 3 types are surface-level. Any specialized domain has vocabulary mismatch. What's LEGAL-specific?

### Step 3: Beyond surface vocabulary — Tatbestand coverage (Legal-specific)

**Approach:** Decompose statutes into clauses, measure how many clauses have ANY lexical overlap with the query.

**Finding:** 27% of query-statute pairs have PARTIAL Tatbestand coverage — the query mentions some statute conditions but not others.

- Query about contract binding → mentions consent and agreement → misses "itikad baik" (good faith), "kecakapan" (legal capacity)
- Query about contract validity (Pasal 1320, 4 conditions) → mentions 2 of 4

**Why this matters:** This is NOT a vocabulary problem. The user doesn't even THINK about good faith or capacity — they just ask "is my contract valid?" A vocabulary-bridging method won't help here because the concept is absent from the query entirely.

**Implication:** The method needs two layers:
1. Vocabulary bridging (address surface gap)
2. Tatbestand reasoning (address conceptual gap — infer missing conditions)

### Step 4: Oracle ceiling experiment — Is vocabulary bridging worth building?

**Approach:** Cheat by copying words from relevant statutes into queries, measure BM25 improvement.

**Finding:**

| Oracle | Hit@10 | MRR@10 | Gain |
|--------|--------|--------|------|
| Baseline (original query) | 10.7% | 0.045 | - |
| +1 key legal term per statute | 53.7% | 0.283 | +43.1 |
| +5 top IDF terms | 99.5% | 0.943 | +88.9 |
| +all content words | 100% | 1.000 | +89.3 |

**Why this matters:** The ceiling is enormous. Just 1 correct term → 5x improvement. 5 correct terms → near-perfect. This proves vocabulary bridging is worth building. The question becomes: can we predict those terms without seeing the answer?

**Context:** Current best (JNLP S1) gets 64.4% hit rate, which sits between the 1-term and 5-term oracle. So JNLP's embeddings implicitly capture ~2-3 terms worth of bridging.

### Step 5: How does QDER design multi-channel methods? (Design inspiration)

**QDER's approach (SIGIR 2025):**
- Problem: static embeddings can't adapt to query context
- Solution: two independent channels (text + entity), each capturing orthogonal signals
- Text channel: BERT embeddings, captures linguistic patterns
- Entity channel: Wikipedia2Vec, captures knowledge-level semantics
- Key: proven independent (Spearman ρ=-0.022)
- Combined via learned bilinear matrix, not fixed weights
- Each component proven necessary via ablation (-78% MAP without interactions)

**Design principle extracted:** Each channel must target a specific, measurable problem. Channels must be independent. Combination must be learned, not hardcoded.

### Step 6: How does IL-PCSR handle legal-specific retrieval? (Domain inspiration)

**IL-PCSR's key findings:**
- Different method types win for different tasks (semantic for LSR, lexical for PCR)
- Event extraction (S-A-O triplets) captures legal facts structurally
- Paragraph-level matching (Para-GNN) exploits document structure
- Ensemble of lexical + semantic always wins over individual methods
- LLM re-ranking adds another 10+ points (captures legal reasoning)
- Dynamic α per query (learned blending) > fixed α

**Design principle extracted:** Legal retrieval needs multiple signal types because the fact-to-norm gap has multiple dimensions. No single representation captures everything.

---

## 2. Proposed Architecture: Dual-Channel Legal Retrieval

### Core Idea
Like QDER uses text + entity channels, we use **text + legal characteristic channels**. Each channel captures an independent signal. Combined via learned scoring.

### Architecture
```
Query q, Statute s
    │
    ├── Channel 1: TEXT (semantic matching)
    │   q_emb = BGE-M3(q)              # [1 x 1024]
    │   s_emb = BGE-M3(s)              # [1 x 1024]
    │   features: q_emb ⊙ s_emb        # element-wise product (JNLP S1)
    │
    ├── Channel 2: LEGAL CHARACTERISTICS
    │   Extracted offline via LLM:
    │   q_legal = LegalExtractor(q)     # events, issues, formal terms
    │   s_legal = TatbestandParser(s)   # conditions, consequences
    │   Features: see Section 3 below
    │
    ├── Channel 3: LEXICAL
    │   bm25_score = BM25(q, s)         # scalar
    │
    └── Scoring
        features = [ch1_product; ch2_legal_features; ch3_bm25]
        score = CatBoost(features)      # or bilinear h^T M h
```

### Why 3 Channels?
1. **Text (BGE-M3 product):** Captures semantic similarity. Already proven (JNLP S1 = 0.34 MRR). But misses vocabulary-level precision.
2. **Legal characteristics:** Captures legal-structural matching. Addresses the 27% partial Tatbestand coverage. NOT captured by dense embeddings.
3. **Lexical (BM25):** Captures exact term overlap. Complements dense when vocabulary overlaps. Proven complementary by IL-PCSR ensembles.

### Training
- Same as JNLP S1: train CatBoost on train split per dataset
- Positive pairs from qrels, negative pairs from BM25 hard negatives
- Feature vector = concatenation of all 3 channels
- Evaluate on test split

---

## 3. Empirical Evidence: Three Independent Legal Perspectives

### Experiment Setup
For 300 test queries on KUHPerdata-humanized, we compared the positive (relevant) statute against BM25's top-1 wrong statute (hard negative) across three perspectives.

### Results: Discriminative Power

| Perspective | Hard neg distinguishable? | Signal strength |
|---|---|---|
| Legal domain | 42.3% of hard negs are from WRONG domain | Moderate |
| Statute function | 63.0% of hard negs have WRONG function | Strong |
| Abstraction | Positives avg 157 citations vs hard neg 3.4 | Very strong (46x) |

### Results: Independence (Spearman ρ)

| Pair | ρ | Independent? |
|---|---|---|
| Domain ↔ Function | -0.000 | Yes (perfectly) |
| Domain ↔ Abstraction | 0.157 | Mostly |
| Function ↔ Abstraction | 0.033 | Yes |

Reference: QDER proved their text ↔ entity channels independent at ρ=-0.022. Our perspectives meet the same standard.

### What Each Perspective Captures
1. **Domain** — is it the right area of law? (persons, property, obligations, evidence)
2. **Function** — does it serve the right legal purpose? (definition, requirement, consequence, procedure)
3. **Abstraction** — is it a general principle or specific rule?

### Limitation of Current Measurement
The current measurement uses KUHPerdata-specific heuristics:
- Domain: article number ranges (won't generalize)
- Function: Indonesian keyword matching (language-specific)
- Abstraction: citation frequency (dataset artifact)

**The perspectives are valid. The measurement methods need to be generalized.** See Section 4 for generalizable feature design.

---

## 4. Legal Feature Design for Channel 2

### TODO: Design exact features

Each feature must be:
1. Computable from query + statute text (no external knowledge at inference)
2. Legal-specific (not just generic NLP)
3. Potentially independent from Channel 1 (embeddings) and Channel 3 (BM25)

Candidate features to investigate:

**A. Vocabulary bridging features:**
- formalized_overlap: overlap after LLM formalizes query terms → legal terms
- term_coverage: fraction of statute content words that appear in enriched query
- register_distance: ratio of formal vs informal terms in query

**B. Structural legal features:**
- tatbestand_coverage: fraction of statute condition clauses matched by query
- event_alignment: do extracted (actor, action, object) triplets from query match statute conditions?
- issue_count: number of independent legal issues in query (from LLM decomposition)
- norm_type_match: does the query's legal domain (contract, property, family) match the statute's domain?

**C. Query-statute interaction features:**
- conditional_match: does the query describe a situation matching the statute's "if" conditions?
- consequence_mention: does the query mention outcomes described in statute's "then" part?
- exception_relevance: does the query describe an exception scenario the statute accounts for?

### Next: Measure which features are independently predictive
For each candidate feature, compute on train set and measure:
1. Correlation with relevance label (is it predictive?)
2. Correlation with Channel 1 features (is it independent?)
3. Cross-dataset consistency (does it work on all 4 datasets?)

Only features that pass all 3 tests should be included.

---

## 4. Implementation Plan

### Phase 1: Feature extraction (GPU VM)
- Build LegalExtractor prompt (extract events, formal terms, issues from queries)
- Build TatbestandParser prompt (decompose statutes into conditions)
- Run on all 4 datasets via vLLM + Qwen 3.5 9B
- Output: JSON per query and per statute with extracted features

### Phase 2: Feature engineering (local)
- Compute pairwise features for all query-statute pairs in train/test
- Measure correlation with relevance, correlation with BGE-M3 features
- Select independent, predictive features

### Phase 3: Training and evaluation (GPU VM)
- Train CatBoost with 3-channel features
- Compare vs JNLP S1 (Channel 1 only)
- Ablation: remove each channel, measure drop
- Run on all 4 datasets

### Phase 4: Independence analysis
- Measure Spearman correlation between channels (like QDER did)
- If channels are independent (|ρ| < 0.1), the architecture is justified
- If channels are correlated, merge or remove redundant ones

---

## 5. Open Questions

1. **Bilinear vs CatBoost for scoring?** QDER uses bilinear (h^T M h), JNLP uses CatBoost. CatBoost handles heterogeneous features (mix of dense embeddings and scalar legal features) better than bilinear which assumes uniform feature space. Start with CatBoost.

2. **How to handle datasets without paragraph structure?** Our statutes are single articles (no paragraphs). Para-GNN requires paragraph segmentation. Solution: treat statute clauses (split by periods/semicolons) as pseudo-paragraphs. Our analysis shows avg 2.2 clauses per statute.

3. **LLM extraction quality across languages?** Qwen 3.5 9B handles ID/ZH/EN well but FR might be weaker. Need to validate extraction quality per language on 50 samples before full run.

4. **Feature leakage risk?** Legal features extracted by LLM might inadvertently encode the answer (e.g., if LLM generates the exact statute terms). Mitigate by ensuring LLM only sees the query, never the statute, during extraction.
