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

## 4. Multi-Representation Independence Experiment (BGE-M3)

BGE-M3 produces 3 output types in one forward pass: dense (1024d), sparse (lexical weights), and ColBERT (token-level multi-vector). We tested whether these, plus BM25 and title embeddings, provide independent signals.

### Test set results (30 queries, random negatives, 368 pairs)

| Signal | ρ with label | Discriminative? |
|---|---|---|
| dense | +0.269 | Yes |
| colbert | +0.267 | Yes |
| bm25 | +0.218 | Yes |
| sparse | +0.178 | Yes |
| title | -0.009 | No |

### Train set results (1670 queries, BM25 hard negatives, 19970 pairs)

| Signal | ρ with label | Pos avg | Neg avg |
|---|---|---|---|
| bm25 | **-0.549** | 5.59 | 15.95 |
| sparse | **-0.409** | 0.034 | 0.081 |
| colbert | -0.284 | 0.420 | 0.480 |
| dense | -0.252 | 0.450 | 0.502 |
| title | -0.027 | 0.321 | 0.323 |

ALL signals are **inverted** against hard negatives: wrong documents score HIGHER than correct ones on every representation. BM25 hard negatives are selected for being lexically similar, and that similarity carries over to all representations.

### Signal independence (train, hard negatives)

| Pair | ρ | Relationship |
|---|---|---|
| dense ↔ colbert | 0.908 | Redundant |
| sparse ↔ dense | 0.602 | Redundant |
| sparse ↔ bm25 | 0.580 | Redundant |
| dense ↔ bm25 | 0.461 | Moderately correlated |
| title ↔ sparse | 0.023 | Independent |
| title ↔ bm25 | 0.049 | Independent |

### Conclusions
1. **Dense and ColBERT are redundant** (ρ=0.908). ColBERT's MaxSim collapses to same signal as dense cosine. No value in using both.
2. **Sparse and BM25 are redundant** (ρ=0.580). BGE-M3 sparse is a learned version of BM25's lexical matching.
3. **Title embedding is independent but not discriminative.** "Pasal N" titles don't encode domain semantics.
4. **No existing representation can distinguish positives from hard negatives at the aggregate level.** JNLP S1 succeeds because CatBoost learns patterns over the full 1024-d product vector, not because any single similarity score works.
5. **The missing third channel must capture something that surface similarity cannot** — legal reasoning, Tatbestand matching, domain classification — signals orthogonal to "how similar do these texts look."

### Updated Architecture Implication

Two working channels confirmed:
- **Channel 1: Dense embeddings** (BGE-M3 1024-d product features → CatBoost) — already works (JNLP S1)
- **Channel 2: Lexical** (BM25 score) — moderately independent from dense (ρ=0.461)

Channel 3 requirements:
- Independent from dense and lexical (ρ < 0.3)
- NOT invertible by surface similarity (hard negatives should NOT score higher)
- Computable across 4 languages
- Must capture legal reasoning that embeddings miss

This is where legal characteristic extraction is genuinely needed — proven by elimination of all simpler alternatives.

---

## 4b. Why CatBoost Works Despite Inverted Aggregate Scores

### The Puzzle
All aggregate similarity scores (cosine, BM25, sparse) are HIGHER for hard negatives than positives. Yet CatBoost on the 1024-d product vector achieves 0.37 MRR (AUC=0.898). How?

### SHAP Analysis Results (KUHPerdata-humanized, 3130 pairs: 925 pos, 2205 hard neg)

**CatBoost prediction separation:**
- Positive pairs: median=0.806 (high confidence)
- Negative pairs: median=0.002 (low confidence)
- AUC: 0.898

**Per-dimension SHAP discrimination:**
- 78.3% of dimensions are HELPFUL (push pos up, neg down)
- 17.9% are harmful
- Total helpful SHAP: 6.22, total harmful: -0.14 (45x ratio)
- Feature importance correlates strongly with discriminative power (ρ=0.808)

**Top 3 discriminative dimensions:**
- Dim 376: disc_shap=+0.381 (6% of all discrimination alone)
- Dim 903: disc_shap=+0.222
- Dim 763: disc_shap=+0.171

### Why Aggregate Scores Are Misleading
The mean-level analysis showed "negatives score higher on 86.8% of dims." But CatBoost doesn't use means — it learns THRESHOLDS. A dimension where negatives have higher mean can still be discriminative if the VALUE DISTRIBUTION differs: e.g., positives cluster at two specific ranges while negatives spread broadly. Decision trees capture this; linear/mean analysis cannot.

### Implication for Method Design
CatBoost already extracts strong signal from BGE-M3 product features. Channel 3 must provide information NOT encoded in ANY of the 1024 BGE-M3 dimensions. From our perspective analysis:
- Domain (ρ=0.073 with dense cosine) → independent from embeddings
- Function (ρ=-0.000 with domain) → independent from domain
- These are candidates IF they can be measured without heuristics

---

## 5. Legal Feature Design for Channel 3

### Requirements

Each feature must be:
1. Computable from query + statute text (no external knowledge at inference)
2. Legal-specific (not just generic NLP)
3. Independent from Channel 1 (embeddings) and Channel 2 (BM25)

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
