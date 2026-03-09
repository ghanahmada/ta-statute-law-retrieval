# Planning: QUAM (Query Affinity Modelling)

| Field | Value |
|-------|-------|
| **Status** | In Progress |
| **Date** | 2026-03-09 |
| **Paper** | Rathee, MacAvaney & Anand, "Quam: Adaptive Retrieval through Query Affinity Modelling" (WSDM 2025) |
| **GitHub** | https://github.com/Mandeep-Rathee/quam |
| **Pre-trained model** | `macavaney/laff` (HuggingFace, BERT-based, 0.1B params) |
| **Builds on** | GAR (MacAvaney et al., CIKM 2022) — already implemented in `src/gar/` |

---

## 1. Problem Statement

GAR overcomes re-ranking recall ceiling by expanding the candidate pool via corpus graph neighbors. However, GAR has two weaknesses:

1. **Ignores edge weights**: GAR assigns each neighbor the parent document's re-ranker score as frontier priority, regardless of how similar/related the neighbor actually is. All neighbors of a high-scoring doc get equal priority.
2. **No accumulated evidence**: GAR uses `max(parent_score)` — if multiple scored docs point to the same neighbor, only the highest parent score counts. Evidence from multiple parents is discarded.

QUAM fixes both with two independent contributions: **SetAff** (algorithm improvement) and **LAFF** (better graph edges).

---

## 2. Method 1: SetAff (Set Affinity) — Algorithm Improvement

### 2.1 Core Idea

Replace GAR's frontier update rule with one that:
- **Uses edge weights** from the corpus graph (BM25 similarity scores)
- **Accumulates** affinity from multiple parent documents
- **Normalizes** parent scores via softmax over the top-s scored documents

### 2.2 Frontier Update Rule

**GAR** (current `adaptive_reranker.py` line 95-97):
```python
frontier[neighbor] = max(frontier.get(neighbor, 0.0), parent_score)
```

**QUAM SetAff**:
```python
S = top_s_docs_by_reranker_score(all_scored_docs)
S_scores = softmax([score for _, score in S])

for doc_id in newly_scored_docs_that_are_in_S:
    for neighbor, aff_score in graph.neighbors(doc_id, weights=True):
        if neighbor not in scored:
            frontier[neighbor] += aff_score * S_scores[doc_id]
```

### 2.3 Key Differences from GAR

| Aspect | GAR | QUAM SetAff |
|--------|-----|-------------|
| Frontier score | `max(parent_score)` | `sum(aff_score * softmax(parent_score))` |
| Edge weights | Ignored | Used (multiplied into frontier score) |
| Accumulation | Replace (max) | Additive (+=) |
| Which parents contribute | All scored docs | Only top-s scored docs (set S) |
| Score normalization | None | Softmax over set S |

### 2.4 Hyperparameters

| Parameter | Budget=50 | Budget=100 | Budget=1000 |
|-----------|-----------|------------|-------------|
| Set size (s) | 10 | 30 | 300 |
| Batch size (b) | 16 | 16 | 16 |
| Graph k (runtime) | 16 | 16 | 16 |

The ratio is roughly `s = budget * 0.2 to 0.3`.

---

## 3. Method 2: LAFF (Learned Affinity) — Better Graph Edges

### 3.1 Core Idea

Instead of raw BM25 similarity weights on graph edges, train a BERT model to predict **co-relevance**: whether two documents tend to be relevant to the same queries.

### 3.2 Architecture

- **Base model**: `bert-base-uncased` (English) / `bert-base-multilingual-cased` (our multilingual case)
- **Input**: `[CLS] doc1_text [SEP] doc2_text [SEP]` (max 512 tokens)
- **Output**: sigmoid(logit) -> probability of co-relevance
- **Loss**: BCEWithLogitsLoss

### 3.3 Training Data Generation

From MS MARCO (original paper) or per-dataset (our case):
1. Sample N queries from train set
2. For each query:
   - Retrieve top-100 docs with first-stage retriever
   - Re-rank with cross-encoder scorer
   - **Positive pairs**: (top-5 retrieved, top-5 scored) -> label=1
   - **Negative pairs**: (bottom-5 retrieved, top-5 scored) -> label=0

### 3.4 Training Config

| Parameter | Value |
|-----------|-------|
| LR | 3e-7 |
| Batch size | 16 |
| Epochs | 5 |
| Max tokens | 512 |
| Gradient clip | 2.0 |
| Warmup | total_steps / 100 |
| Scheduler | Linear warmup + decay |
| Loss | BCEWithLogitsLoss |

### 3.5 Application to Corpus Graph

After training, re-weight all edges in the BM25 corpus graph:
- For each edge (doc_i, doc_j) with BM25 weight w:
  - Compute LAFF score: `laff_score = sigmoid(model([doc_i_text, doc_j_text]))`
  - New weight = `laff_score` (replaces BM25 weight)
- Save as a new corpus graph with LAFF-reweighted edges

---

## 4. Implementation Plan

### 4.1 Option A: SetAff Only (No Training) — IMPLEMENTING NOW

**Rationale**: SetAff gives most of the benefit (~5-7% nDCG improvement over GAR) with zero training. Uses existing BM25 corpus graph weights that are already computed and stored.

**New files**:
```
src/quam/
  __init__.py          # Module docstring
  adaptive_reranker.py # QUAM class with SetAff frontier update
src/evaluate_quam.py   # Evaluation script (same pattern as evaluate_gar.py)
```

**Reused from GAR**:
- `gar/corpus_graph.py` — BM25 corpus graph (shared, imported by both)
- Scorer classes (MonoT5Scorer, CrossEncoderScorer) — extracted or imported
- BM25 pool generation, pre-scoring logic

### 4.2 Option B: SetAff + LAFF (Future)

Adds learned affinity model for edge re-weighting. Requires:
- `quam/laff.py` — LAFF model training + graph re-weighting
- Training data generation per dataset
- `bert-base-multilingual-cased` for multilingual support
- Additional evaluate script flags for LAFF vs BM25 graph

---

## 5. Experimental Conditions

Following the QUAM paper's 5-condition design:

| # | Method | Graph | Status |
|---|--------|-------|--------|
| 1 | BM25 -> Scorer (no graph) | None | Baseline (evaluate_gar.py with budget=pool) |
| 2 | BM25 -> GAR + BM25 graph | BM25 kNN | Done (evaluate_gar.py) |
| 3 | BM25 -> QUAM + BM25 graph | BM25 kNN | **Option A (this PR)** |
| 4 | BM25 -> GAR + LAFF graph | LAFF-reweighted | Option B (future) |
| 5 | BM25 -> QUAM + LAFF graph | LAFF-reweighted | Option B (future) |

---

## 6. References

1. Rathee, M., MacAvaney, S., Anand, A. (2025). "Quam: Adaptive Retrieval through Query Affinity Modelling." WSDM 2025. arXiv:2410.20286.
2. MacAvaney, S., Tonellotto, N., Macdonald, C. (2022). "Adaptive Re-Ranking with a Corpus Graph." CIKM 2022. arXiv:2208.08942.
3. Official repo: https://github.com/Mandeep-Rathee/quam
4. Pre-trained LAFF model: https://huggingface.co/macavaney/laff
