# Analysis 01: Vocabulary Gap Characterization

## Objective

Measure and characterize the lexical gap between layperson queries and their ground-truth statute documents. Show that:
1. The gap is **systematic** — colloquial language consistently diverges from formal legal vocabulary
2. The gap is **not random** — predictable patterns exist (specific legal concepts have specific colloquial counterparts)
3. The gap varies by dataset/language, explaining why BM25 collapses differently across datasets

## Script

```bash
python src/analysis/vocab_gap_analysis.py
```

**Dependencies:** `nltk`, `jieba` (for Chinese tokenization)

**Options:**
- `--datasets`: subset of datasets to analyze (default: all)
- `--output_dir`: where to write artifacts (default: `outputs/analysis/vocab_gap/`)

## Methodology

1. **Tokenization**: language-appropriate tokenization
   - Indonesian/French/English: regex word tokenization + NLTK stopword removal
   - Chinese: jieba word segmentation, filtering single-char and non-CJK tokens

2. **Metrics per query-document pair** (using test qrels):
   - Jaccard similarity: |Q ∩ D| / |Q ∪ D|
   - Query coverage: |Q ∩ D| / |Q| (what fraction of query terms appear in the doc)
   - Zero-overlap indicator: pairs sharing zero content tokens

3. **Aggregate statistics**:
   - Mean/median Jaccard, % zero-overlap, % below thresholds
   - Top discriminative terms (query-exclusive vs doc-exclusive vs shared)

## Results (2026-05-16)

| Dataset | Lang | Pairs | Zero% | Jaccard | QCov | <5%J | <10%J |
|---------|------|-------|-------|---------|------|------|-------|
| kuhperdata-exp | id | 609 | **73.4%** | 0.012 | 0.032 | 92.1% | 99.2% |
| kuhperdata-summ-exp | id | 582 | 31.3% | 0.029 | 0.044 | 82.8% | 96.4% |
| bsard | fr | 1061 | 28.0% | 0.022 | 0.220 | 89.7% | 99.6% |
| stard | zh | 512 | 13.1% | 0.082 | 0.326 | 45.3% | 71.5% |
| ilpcsr | en | 4835 | 0.0% | 0.045 | 0.054 | 66.0% | 92.4% |
| coliee | en | 107 | 0.0% | 0.299 | 0.638 | 1.9% | 6.5% |

**Legend:**
- Zero% = % of query-doc pairs sharing zero content tokens after stopword removal
- Jaccard = mean Jaccard similarity (|Q∩D| / |Q∪D|)
- QCov = mean query coverage (|Q∩D| / |Q|)
- <5%J / <10%J = % of pairs with Jaccard below threshold

## Key Findings

### 1. The gap is systematic

Colloquial queries consistently use different vocabulary from formal statutes:

**Indonesian (kuhperdata-exp):**
- Query terms (colloquial): `bayar` (pay), `nggak` (slang "no"), `gimana` (slang "how"), `bikin` (make), `bingung` (confused)
- Statute terms (formal): `perikatan` (obligation), `debitur` (debtor), `persetujuan` (agreement), `mengikatkan` (to bind oneself)

**French (bsard):**
- Query terms (citizen): `payer` (to pay), `comment` (how), `faire` (to do), `bruxelles`, `wallonie`
- Statute terms (legal): `art`, `code`, `chapitre`, `judiciaire`, `section`

**Chinese (stard):**
- Query terms (colloquial): `是否` (whether), `什么` (what), `怎么办` (what to do), `如何` (how)
- Statute terms (formal): `中华人民共和国` (PRC), `民法典` (Civil Code), `应当` (shall), `下列` (the following)

### 2. The gap is NOT random — it follows predictable patterns

Specific legal concepts have specific colloquial counterparts:
- Indonesian: `rugi`/`ganti` (loss/replace) → `kerugian` (damages, formal legal term)
- Indonesian: `janji`/`perjanjian` (promise/agreement, colloquial) → `perikatan`/`persetujuan` (obligation/consent, Civil Code terms)
- French: `aide` (help, citizen term) ↔ `aide judiciaire` (legal aid, statute term)
- Chinese: `赔偿` (compensate, colloquial) ↔ `承担...责任` (bear responsibility, legal phrasing)

### 3. Gap severity correlates with BM25 performance

| Dataset | Zero Overlap | BM25 MRR@10 | Interpretation |
|---------|-------------|-------------|----------------|
| kuhperdata-exp | 73.4% | 0.03 | Severe gap → BM25 collapses |
| kuhperdata-summ-exp | 31.3% | 0.08 | Moderate gap → BM25 weak |
| bsard | 28.0% | 0.25 | Moderate gap → BM25 partial |
| stard | 13.1% | 0.34 | Low gap → BM25 reasonable |
| ilpcsr | 0.0% | 0.16 | No gap but long docs → dilution |
| coliee | 0.0% | — | No gap → lexical matching works |

## Output Artifacts

- `outputs/analysis/vocab_gap/vocab_gap_summary.json` — per-dataset aggregate stats
- `outputs/analysis/vocab_gap/vocab_gap_detail.jsonl` — full results with top terms per dataset
