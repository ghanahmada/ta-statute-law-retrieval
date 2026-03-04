# Benchmark Implementation Guide: Adding a New Retrieval Method

| Field | Value |
|-------|-------|
| **Date** | 2026-03-05 |

---

## 1. Overview

This guide describes how to add a new retrieval method evaluation script following the pattern established by `evaluate_bm25.py`, `evaluate_dense_retrieval.py`, `evaluate_jnlp.py`, and `scripts/sailer/evaluate_retrieval.py`.

---

## 2. The Pattern

Every evaluation script follows the same structure:

1. **DATASETS dict** at module level with dataset paths and method-specific config
2. **A retrieval function** that produces `rankings: {query_id: [doc_id, ...]}`
3. **Argparse** with `--dataset`, `--split`, `--top_k`
4. **Main loop** over datasets: load data, run retrieval, print metrics

### Existing scripts

| Script | DATASETS keys | Metrics approach |
|--------|---------------|-----------------|
| `evaluate_bm25.py` | `path`, `lang` | `evaluate_ranking()` |
| `evaluate_dense_retrieval.py` | flat args (no DATASETS dict) | individual `calculate_*` functions |
| `evaluate_jnlp.py` | `path`, `max_length`, `batch_size` | `evaluate_ranking()` via pipeline |
| `scripts/sailer/evaluate_retrieval.py` | hardcoded constants | `pytrec_eval` |

---

## 3. Step-by-Step: Create `src/evaluate_yourmethod.py`

### Step 1: Define DATASETS dict

Merge the configs you need from existing scripts. Common fields:

- `path` — dataset directory (all scripts use this)
- `lang` — language code, from `evaluate_bm25.py:9–14`
- `max_length`, `batch_size` — encoding params, from `evaluate_jnlp.py:5–10`

```python
DATASETS = {
    "kuhperdata": {"path": "data/kuhperdata", "lang": "id", "max_length": 1024, "batch_size": 64},
    "bsard":      {"path": "data/bsard",      "lang": "fr", "max_length": 1024, "batch_size": 64},
    "ilpcsr":     {"path": "data/ilpcsr",      "lang": "en", "max_length": 8192, "batch_size": 8},
    "stard":      {"path": "data/stard",       "lang": "zh", "max_length": 1024, "batch_size": 64},
}
```

Add any new fields your method needs to each entry.

### Step 2: Write the retrieval function

Follow this template:

```python
def run_yourmethod(loader: DataLoader, top_k: int, **your_params) -> dict:
    doc_ids, doc_texts = loader.get_corpus_texts()    # returns (list[str], list[str])
    query_ids, query_texts = loader.get_query_texts()  # returns (list[str], list[str])

    # --- Your retrieval logic ---
    # Produce rankings: {query_id: [doc_id, ...]} ordered by relevance, length top_k
    rankings = {}
    for qid in query_ids:
        rankings[qid] = [...]  # top_k doc_ids

    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}
    return evaluate_ranking(rankings, ground_truth, top_k)
```

**Key points:**
- Use `loader.get_corpus_texts()` and `loader.get_query_texts()` — see `util/dataloader.py:79–90`
- Build `ground_truth` from `loader.qrels` (dict of `{query_id: {doc_id: int}}`)
- Always return via `evaluate_ranking()` (`util/metrics.py:32–69`) for consistent output
- Return dict has keys: `f"mrr@{top_k}"`, `f"recall@{top_k}"`, `f"precision@{top_k}"`, `"n_queries"`, `"hit_rate"`

### Step 3: Add argparse

Follow `evaluate_bm25.py:46–53`:

```python
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
parser.add_argument("--top_k", type=int, default=10)
# Add method-specific args here
```

### Step 4: Main loop

Follow `evaluate_bm25.py:55–76`:

```python
args = parser.parse_args()
datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

for name, cfg in datasets.items():
    data_dir = cfg["path"]
    loader = DataLoader(
        f"{data_dir}/corpus.jsonl",
        f"{data_dir}/queries.jsonl",
        f"{data_dir}/qrels_{args.split}.tsv",
    ).load()

    metrics = run_yourmethod(loader, args.top_k, ...)

    k = args.top_k
    print(f"\n[{name}] ({args.split})")
    print(f"  MRR@{k}:       {metrics[f'mrr@{k}']:.4f}")
    print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
    print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f"  Hit rate:      {metrics['hit_rate']:.4f}")
    print(f"  Queries:       {metrics['n_queries']}")
```

---

## 4. Conventions

- **Imports**: `from util.dataloader import DataLoader`, `from util.metrics import evaluate_ranking` — project assumes `src/` is on `sys.path`
- **DataLoader**: Always chain `.load()` — `DataLoader(corpus, queries, qrels).load()`
- **File paths**: Built from `cfg["path"]` + split: `f"{data_dir}/qrels_{args.split}.tsv"`
- **Lazy imports**: Put heavy deps inside the function body (e.g., `FlagEmbedding`, `faiss`)

---

## 5. Note: Exposing Rankings from JNLP Pipeline

If your method delegates to `PipelineOrchestrator.evaluate_stage1_only`, note that it currently returns only metrics (`src/jnlp/pipeline.py:491`). To also get the raw rankings, edit line 491:

```python
# Before
return metrics

# After
return {**metrics, "rankings": rankings}
```

The `rankings` dict is built at lines 469–472. Existing callers are unaffected since all metric keys remain present.

---

## 6. Codebase Reference

| What | File | Lines |
|------|------|-------|
| BM25 DATASETS dict | `src/evaluate_bm25.py` | 9–14 |
| BM25 retrieval function | `src/evaluate_bm25.py` | 22–42 |
| BM25 argparse + main loop | `src/evaluate_bm25.py` | 46–76 |
| Dense encoding + retrieval | `src/evaluate_dense_retrieval.py` | 10–58 |
| JNLP DATASETS dict | `src/evaluate_jnlp.py` | 5–10 |
| JNLP pipeline delegation | `src/evaluate_jnlp.py` | 30–53 |
| SAILER FAISS retrieval | `scripts/sailer/evaluate_retrieval.py` | 52–80 |
| `DataLoader` class | `src/util/dataloader.py` | 25–97 |
| `get_corpus_texts` / `get_query_texts` | `src/util/dataloader.py` | 79–90 |
| `evaluate_ranking` function | `src/util/metrics.py` | 32–69 |
| `Config` dataclass | `src/jnlp/__init__.py` | 37–97 |
| `PipelineOrchestrator` | `src/jnlp/pipeline.py` | 15 |
| `evaluate_stage1_only` return | `src/jnlp/pipeline.py` | 491 |
