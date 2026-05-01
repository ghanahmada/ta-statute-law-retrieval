# Ground Truth Expansion via LLM Subsumption Judgment

## Motivation

The original KUHPerdata dataset has sparse ground truth: ~2.2 relevant articles per query on average, derived solely from court decision citations. This sparsity creates two problems:

1. **Incomplete evaluation**: Methods that correctly retrieve relevant-but-unlabeled articles are penalized (false negatives inflate recall denominators)
2. **Training signal poverty**: GNN and learning-to-rank models see too few positive examples per query

Manual annotation of 2127 articles × 1847 queries is infeasible. We use LLM-based subsumption judgment as a scalable alternative.

## Method: Rubrik Analisis Unsur Hukum Perdata

### Origin: Adaptation from Criminal Law

The method is adapted from **Rubrik Pidana** — a structured element analysis rubric from *Buku Saku: Memahami untuk Membasmi*, originally designed for criminal law element verification. We adapt it to the civil law domain (KUHPerdata) following this mapping:

| Komponen | Rubrik Pidana | Rubrik Perdata (Adaptasi) |
|----------|--------------|--------------------------|
| Unsur | Unsur Tindak Pidana | Unsur Hukum Perdata |
| Fakta | Fakta Perbuatan | Fakta Kasus yang Memenuhi Unsur |
| Dasar | Alat Bukti | Dasar Pertimbangan Hukum |
| Kesimpulan | Unsur terpenuhi → terbukti | Unsur terpenuhi/dibahas → relevan |

### Core Principle

> Sebuah pasal dinyatakan **RELEVAN** jika seluruh unsur pokok pasal **dibahas** dalam fakta kasus (baik terbukti maupun diperdebatkan). Jika ada unsur pokok yang sama sekali tidak ada dalam fakta, pasal dinyatakan **TIDAK RELEVAN**.

The critical distinction: **"dibahas" ≠ "terbukti"**. An article is relevant even if the court ultimately rejected the claim — what matters is that the legal elements became the subject of legal discourse, not the outcome. This mirrors how Indonesian judges reason: Pasal 1365 (PMH) is relevant to a tort case even when the plaintiff loses, because the elements (perbuatan, melawan hukum, kerugian, kesalahan, kausalitas) were all analyzed.

### Criteria

- **RELEVAN**: The article's core elements are addressed in the case facts, or the article serves as a basis for legal arguments/judicial reasoning
- **TIDAK RELEVAN**: Core elements are absent from the facts, the legal domain is different, or the match is only surface-level keyword overlap

### Worked Examples

**Example 1: Pasal 1365 (PMH) — RELEVAN despite losing**

A shipping damage case where the plaintiff claimed negligence but the court found no fault:

| No | Unsur | Fakta Kasus | Dasar Pertimbangan |
|----|-------|-------------|-------------------|
| 1 | Adanya perbuatan | Tergugat melakukan pengangkutan | Jasa dilaksanakan |
| 2 | Melanggar hukum | Penggugat mendalilkan kurang hati-hati | Dalil dalam gugatan |
| 3 | Adanya kerugian | Server rusak saat tiba | Kerugian materiil terbukti |
| 4 | Adanya kesalahan | **Tidak terbukti** kesalahan Tergugat | Beban pembuktian tidak terpenuhi |
| 5 | Hubungan kausal | **Tidak terbukti** kausalitas | Kerusakan bisa akibat pengemasan |

→ **RELEVAN** — Unsur (4) dan (5) tidak terbukti, tetapi seluruh unsur **dibahas** dalam kasus.

**Example 2: Pasal 1367 (Tanggung Jawab Bawahan) — TIDAK RELEVAN**

Same shipping case:

| No | Unsur | Fakta Kasus | Dasar Pertimbangan |
|----|-------|-------------|-------------------|
| 1 | Hubungan subordinasi | Hubungan kontraktual antara dua pihak setara | Tidak ada fakta subordinasi |

→ **TIDAK RELEVAN** — Unsur pokok (hubungan subordinasi) sama sekali tidak ada dalam fakta.

**Example 3: Pasal 570 (Hak Milik) — TIDAK RELEVAN (domain mismatch)**

A credit default case where land is involved as collateral:

| No | Unsur | Fakta Kasus | Dasar Pertimbangan |
|----|-------|-------------|-------------------|
| 1 | Barang yang dimiliki | Tanah jaminan memang objek kebendaan | Tersedia dalam fakta |
| 2 | Hak menikmati secara bebas | Kasus bukan tentang pelaksanaan hak milik | Inti kasus: hubungan perikatan |

→ **TIDAK RELEVAN** — Kasus tentang wanprestasi, bukan hak kebendaan. Surface-level presence of "tanah" does not make a property rights article relevant to a contract dispute.

## LLM Implementation

### Prompt Design

The rubric is encoded as an LLM system prompt that instructs the model as an Indonesian civil law expert:

```
Metode analisis unsur:
1. Identifikasi unsur-unsur pokok dari pasal
2. Periksa apakah SETIAP unsur pokok dibahas atau terpenuhi oleh fakta kasus
3. Jika seluruh unsur pokok terpenuhi/dibahas oleh fakta → RELEVAN
4. Jika ada unsur pokok yang sama sekali tidak ada dalam fakta → TIDAK_RELEVAN
```

Three anti-pattern rules prevent common LLM errors:

1. **"Dibahas" includes rejected arguments** — prevents the LLM from marking articles as irrelevant just because the court ruled against the plaintiff
2. **Keyword overlap ≠ relevance** — explicitly warns against surface matching (e.g., "mengembalikan" in both loan and lease contexts)
3. **Hub articles get no special treatment** — Pasal 1365, 1320, 1338 must pass the same element analysis as obscure articles

### Output Format

Each judgment produces chain-of-thought reasoning per article:

```
[n] unsur: (core elements) | fakta: (which match/don't) | RELEVAN/TIDAK_RELEVAN
```

This verbose format is intentional — forcing per-article reasoning prevents lazy classification. The tradeoff (more generation tokens) is acceptable because ground truth quality directly impacts all downstream evaluations.

## Pipeline

```
For each query q:
  1. Retrieve BM25 top-50 candidate articles (excluding existing ground truth)
  2. Send all 50 candidates + query to LLM in a single prompt
  3. LLM performs unsur analysis per article
  4. Parse RELEVAN/TIDAK_RELEVAN judgments
  5. Merge newly relevant articles into qrels with score=1
  6. Stream each result to expansion_log.jsonl immediately (resumable)
```

## Candidate Selection

### Current: BM25 Top-50

The initial implementation uses BM25 top-50 candidates per query. Known limitation: BM25 baseline recall is only ~10-15%, so the candidate pool misses semantically relevant articles that use different vocabulary.

### Future: Exhaustive Judgment

For the final benchmark, exhaustive judgment over all 2127 articles per query eliminates pooling bias entirely. With 1847 queries × 2127 articles ÷ 50 per batch = ~78K LLM calls — feasible with batch inference on a 27B model. This produces **complete qrels** with zero pooling bias, which is itself a dataset contribution.

## Implementation Details

- **Model**: Qwen 3.6 27B AWQ (quantized), served via vLLM
- **Concurrency**: 24 queries in parallel per dataset
- **Resumability**: Each judgment streamed to `expansion_log.jsonl` via `asyncio.as_completed` + `flush()`
- **Thinking disabled**: `enable_thinking: False` to avoid wasting tokens on internal reasoning
- **Temperature**: 0 for deterministic judgments
- **Max tokens**: 8092 per batch (50 articles × ~80-150 tokens per judgment)

## Preliminary Observations (BM25 Top-50 Run)

- ~1.7 new relevant articles per query on average
- Summarized queries yield slightly more expansions (better lexical overlap with statute text)
- The LLM correctly reasons through borderline cases, distinguishing legal relationships despite shared keywords
- Hub articles are not automatically marked relevant — genuine element matching required

## Validation Plan

1. **Human agreement study**: Sample 100 query-article pairs, have a legal expert judge independently, compute Cohen's kappa
2. **Error analysis**: Categorize disagreements into false positives (LLM too lenient) vs false negatives (LLM too strict)
3. **Cross-model validation**: Run with a different LLM family, measure inter-model agreement

## Commands

```bash
# Serve vLLM
vllm serve QuantTrio/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b \
  --max-num-seqs 50 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# Expand humanized
python src/data/expand_qrels.py \
  --dataset kuhperdata-humanized \
  --output_name kuhperdata-exp \
  --base_url http://127.0.0.1:8000/v1 \
  --model qwen3.6-27b \
  --concurrency 24

# Expand summarized
python src/data/expand_qrels.py \
  --dataset kuhperdata-summarized \
  --output_name kuhperdata-summ-exp \
  --base_url http://127.0.0.1:8000/v1 \
  --model qwen3.6-27b \
  --concurrency 24
```
