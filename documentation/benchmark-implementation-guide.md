# Benchmark Implementation Guide: Adding a New Retrieval Method

| Field | Value |
|-------|-------|
| **Date** | 2026-05-01 |

---

## 1. Overview

This guide describes how to add a new retrieval method evaluation script following the patterns established across the benchmark. The project evaluates **10 retrieval methods** across up to **7 datasets** in 4 languages (ID, FR, EN, ZH).

### Evaluation Scripts

| Script | Method | DATASETS config | Metrics |
|--------|--------|----------------|---------|
| `evaluate_bm25.py` | BM25 (lexical) | `path`, `lang` | `evaluate_ranking()` |
| `evaluate_dense_retrieval.py` | BGE-M3 (dense) | flat path strings | individual `calculate_*` per K |
| `evaluate_jnlp.py` | CatBoost + BGE-M3 (learned) | `path`, `max_length`, `batch_size` | via `PipelineOrchestrator` |
| `evaluate_paragnn.py` | Para-GNN / StructGNN (graph) | imported from `paragnn` module | via `trainer.train()` |
| `evaluate_gar.py` | Graph-based Adaptive Re-ranking | `path`, `lang` + scorer config | `evaluate_ranking()` + `avg_expanded_docs` |
| `evaluate_rerank.py` | BM25 + Reranker (ablation) | `path`, `lang` | `evaluate_ranking()` |
| `evaluate_quam.py` | QUAM (SetAff variant of GAR) | `path`, `lang` | `evaluate_ranking()` + `avg_expanded_docs` |
| `context_1/evaluate_context1.py` | Agentic retrieval | `path`, `lang` | `evaluate_ranking()` (async) |
| `scripts/sailer/evaluate_retrieval.py` | SAILER (legacy) | hardcoded kuhperdata only | `pytrec_eval` |

### Datasets

| Key | Path | Lang | Notes |
|-----|------|------|-------|
| `kuhperdata-humanized` | `data/kuhperdata-humanized` | id | Casual first-person queries |
| `kuhperdata-summarized` | `data/kuhperdata-summarized` | id | LLM-summarized case facts |
| `kuhperdata-exp` | `data/kuhperdata-exp` | id | Expanded ground truth (humanized) |
| `kuhperdata-summ-exp` | `data/kuhperdata-summ-exp` | id | Expanded ground truth (summarized) |
| `bsard` | `data/bsard` | fr | Belgian statute retrieval |
| `ilpcsr` | `data/ilpcsr` | en | Indian legal provision retrieval |
| `stard` | `data/stard` | zh | Chinese statute retrieval |

Not all scripts support all datasets. Para-GNN supports all 7; QUAM only has 4 (no humanized/summarized splits); Context-1 has 4 (no ilpcsr).

---

## 2. The Pattern

Every evaluation script follows this structure:

1. **DATASETS dict** at module level with dataset paths and method-specific config
2. **A retrieval function** that produces `rankings: {query_id: [doc_id, ...]}`
3. **Argparse** with `--dataset`, `--split`, `--top_k`, `--max_relevant`
4. **Main loop** over datasets: load data, run retrieval, print metrics

### Standard metrics (4 metrics, consistent across all scripts)

- `mrr@{top_k}` — Mean Reciprocal Rank
- `recall@{top_k}` — Recall at K
- `precision@{top_k}` — Precision at K
- `hit_rate` — fraction of queries with at least 1 hit

---

## 3. Step-by-Step: Create `src/evaluate_yourmethod.py`

### Step 1: Define DATASETS dict

Use the 5-dataset standard (or 7 if supporting expanded variants):

```python
DATASETS = {
    "kuhperdata-humanized":  {"path": "data/kuhperdata-humanized",  "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard":                 {"path": "data/bsard",                 "lang": "fr"},
    "ilpcsr":                {"path": "data/ilpcsr",                "lang": "en"},
    "stard":                 {"path": "data/stard",                 "lang": "zh"},
}
```

Add method-specific fields as needed:
- `max_length`, `batch_size` — encoding params (see `evaluate_jnlp.py:5–11`)
- Scorer/reranker config — see `evaluate_gar.py:47–52` for the SCORERS pattern

### Step 2: Write the retrieval function

```python
def run_yourmethod(loader: DataLoader, top_k: int, **your_params) -> dict:
    doc_ids, doc_texts = loader.get_corpus_texts()    # returns (list[str], list[str])
    query_ids, query_texts = loader.get_query_texts()  # returns (list[str], list[str])

    # --- Your retrieval logic ---
    rankings = {}
    for qid in query_ids:
        rankings[qid] = [...]  # top_k doc_ids ordered by relevance

    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}
    return evaluate_ranking(rankings, ground_truth, top_k)
```

**Key points:**
- Use `loader.get_corpus_texts()` and `loader.get_query_texts()` — `util/dataloader.py:86–96`
- Build `ground_truth` from `loader.qrels` (dict of `{query_id: {doc_id: int}}`)
- Return via `evaluate_ranking()` (`util/metrics.py:32–69`) for consistent output
- Return dict keys: `f"mrr@{top_k}"`, `f"recall@{top_k}"`, `f"precision@{top_k}"`, `"n_queries"`, `"hit_rate"`, plus `per_query_*` arrays

### Step 3: Add argparse

```python
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="kuhperdata-humanized", choices=[*DATASETS, "all"])
parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--max_relevant", type=int, default=5)
# Add method-specific args here
```

### Step 4: Main loop

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

    if args.max_relevant:
        loader.filter_max_relevant(args.max_relevant)

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
- **max_relevant filtering**: Use `loader.filter_max_relevant(n)` to exclude queries with >n relevant docs (default 5)
- **Dataset defaults**: For per-dataset hyperparameter overrides, use a two-tier dict pattern — see `evaluate_gar.py:54–74`

---

## 5. Advanced Patterns

### Config dataclass (JNLP, Para-GNN)

For complex methods, define a config dataclass instead of flat argparse:

- **JNLP**: `src/jnlp/__init__.py:36–109` — `Config` with stage1/stage2 params, encoding, LoRA, training
- **Para-GNN**: `src/paragnn/__init__.py:49–91` — `ParaGNNConfig` with structure_mode, graph params, training

### Async evaluation (Context-1)

For LLM-based methods requiring a server:
- Use `asyncio` with concurrency control — see `context_1/evaluate_context1.py:114–143`
- Stream results to JSONL for resumability — check existing log on startup to skip completed queries
- Requires `--base_url` and `--model` args matching the vLLM `--served-model-name`

### Scorer/Reranker pattern (GAR, Rerank, QUAM)

Shared across graph-expansion methods:
```python
SCORERS = {
    "monot5": ("seq2seq", "castorini/monot5-base-msmarco"),
    "mt5":    ("seq2seq", "unicamp-dl/mt5-base-mmarco-v2"),
    "bge":    ("cross_encoder", "BAAI/bge-reranker-v2-m3"),
}
```

---

## 6. Codebase Reference

| What | File | Lines |
|------|------|-------|
| BM25 DATASETS + retrieval | `src/evaluate_bm25.py` | 9–42 |
| BM25 argparse + main loop | `src/evaluate_bm25.py` | 55–108 |
| Dense DATASETS + retrieval | `src/evaluate_dense_retrieval.py` | 18–68 |
| Dense argparse + main loop | `src/evaluate_dense_retrieval.py` | 72–220 |
| JNLP DATASETS + pipeline | `src/evaluate_jnlp.py` | 5–72 |
| Para-GNN DATASETS + config | `src/paragnn/__init__.py` | 49–101 |
| Para-GNN evaluation | `src/evaluate_paragnn.py` | 38–200 |
| GAR DATASETS + scorers | `src/evaluate_gar.py` | 39–74 |
| GAR argparse + main loop | `src/evaluate_gar.py` | 405–540 |
| Rerank evaluation | `src/evaluate_rerank.py` | 39–343 |
| QUAM evaluation | `src/evaluate_quam.py` | 40–533 |
| Context-1 agentic eval | `src/context_1/evaluate_context1.py` | 45–368 |
| SAILER legacy eval | `scripts/sailer/evaluate_retrieval.py` | 1–134 |
| `DataLoader` class | `src/util/dataloader.py` | 28–96 |
| `evaluate_ranking` function | `src/util/metrics.py` | 32–69 |
| `Config` dataclass (JNLP) | `src/jnlp/__init__.py` | 36–109 |
| `ParaGNNConfig` dataclass | `src/paragnn/__init__.py` | 49–91 |
| `PipelineOrchestrator` | `src/jnlp/pipeline.py` | 15 |
| `evaluate_stage1_only` | `src/jnlp/pipeline.py` | 379–501 |
