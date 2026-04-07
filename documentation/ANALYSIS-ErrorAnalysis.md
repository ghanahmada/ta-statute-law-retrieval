# Error Analysis: What Causes Retrieval Failure?

**Date:** 2026-04-04  
**Goal:** Identify specific characteristics of legal text that cause retrieval failure, to inform method design.

## Key Finding: Vocabulary Overlap is the ONLY Consistent Failure Predictor

Across all 4 multilingual datasets, BM25 failure correlates strongly with vocabulary overlap between query and relevant statutes. No other feature (query length, sentence count, question format) has predictive power.

### BM25 Error Analysis (Hit@10 vs Miss@10)

| Dataset | Lang | Hit Rate | Overlap (Hit) | Overlap (Miss) | Gap | ZeroOvl (Hit) | ZeroOvl (Miss) |
|---------|------|----------|---------------|----------------|-----|---------------|----------------|
| KUHPerdata-humanized | ID | 10.7% | 3.91 | 2.18 | +1.73 | 3.7% | 8.9% |
| KUHPerdata-summarized | ID | 17.7% | 7.39 | 4.76 | +2.63 | 0.6% | 0.3% |
| BSARD | FR | 42.3% | 2.43 | 1.11 | +1.32 | 14.8% | 45.3% |
| STARD | ZH | 53.2% | 13.71 | 7.40 | +6.31 | 0.0% | 0.7% |

### Features That DON'T Predict Failure

| Feature | Avg Gap (Hit - Miss) | Consistent? |
|---------|---------------------|-------------|
| query_len | -7 to +5 chars | No (sometimes longer queries fail more) |
| num_sentences | -0.07 to +0.02 | No (essentially zero) |
| num_relevant | +0.13 to +3.46 | Weak, inconsistent |
| is_question | ~0 | No |

## Implication for Method Design

The method must bridge the **vocabulary gap** between how users describe legal situations and how statutes are written. This is NOT about:
- Query structure (segmentation, decomposition)
- Query length (short vs long)
- Query format (question vs narrative)

It IS about:
- **Lexical mismatch**: users say "uang kembali" (money back), statutes say "restitusi" (restitution)
- **Register gap**: users use procedural language, statutes use normative language
- **Abstraction gap**: users describe concrete events, statutes define abstract conditions

## Vocabulary Gap Categorization (Cross-Language)

### 3 Surface Gap Types (Universal Across ID, FR, ZH)

**Gap Type 1: Framing Words** (query-only, should be stripped)
- ID: dianggap (123x), mengapa (123x), bagaimana (49x)
- FR: puis-je (38x), comment (27x), dois-je (19x)
- ZH: 是否 (54x), 什么 (37x), 如何 (31x)
- These are question framing artifacts. Statutes never contain them.

**Gap Type 2: Concreteness Gap** (need abstraction mapping)
- ID: uang→harta, tanah→benda, istri→pihak (specific→abstract)
- FR: propriétaire→personne, enfant→personne, payer→demande
- ZH: 公司→当事人, 邻居→当事人 (company/neighbor→parties)

**Gap Type 3: Register Gap** (need formalization mapping)
- ID: membayar→mewajibkan, bisa→berlaku (everyday→prescriptive)
- FR: faire→conformément, signé→lorsque
- ZH: 应该→应当, 需要→依照 (should→shall, need→in accordance with)

### Vocabulary Space Disjointness
| Dataset | Query-only | Doc-only | Shared | Shared % |
|---------|-----------|----------|--------|----------|
| KUHPerdata-humanized | 1,375 | 1,917 | 160 | 4.6% |
| BSARD | 597 | 10,266 | 236 | 2.1% |
| STARD | 933 | 2,838 | 513 | 12.0% |

## Deeper Legal Characteristic: Tatbestand Coverage

Beyond vocabulary, there is a **conceptual gap**: queries often mention only SOME conditions from a statute, not all.

**KUHPerdata-humanized Tatbestand coverage:**
- 64.2% of pairs: full coverage (query touches all statute clauses)
- 27.0% of pairs: partial coverage (some clauses missed)
- 8.8% of pairs: zero coverage

**What gets missed are specific Tatbestand elements:**
- Pasal 1338: query mentions binding force but misses "itikad baik" (good faith)
- Pasal 1320: query mentions contract validity but misses "kecakapan" (legal capacity)
- Pattern: queries describe the SITUATION but not all LEGAL REQUIREMENTS

**Implication:** A retrieval method that only bridges vocabulary will still miss statutes whose conditions the user doesn't think to mention. The method needs to understand what the statute REQUIRES and check if the query's facts COULD satisfy those requirements, even when not explicitly stated.

## Next Analysis Needed

1. **What does Dense/JNLP get right that BM25 misses?** (need GPU VM results)
2. **Where do ALL methods fail?** (hardest queries)
3. **Legal event extraction quality** (LLM-based, needs GPU)
