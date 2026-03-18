# KUHPerdata Statute Retrieval

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run BM25 Evaluation

```bash
python src/evaluate_bm25.py --top_k 10
```

Options:
- `--top_k`: Number of top documents (default: 10)
- `--bm25_b`: BM25 b parameter (default: 0.75)
- `--bm25_k1`: BM25 k1 parameter (default: 1.5)
- `--verbose`: Show detailed results
