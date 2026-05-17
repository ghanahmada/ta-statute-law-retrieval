"""OpenSearch Neural Sparse (SPLADE-like) learned sparse retrieval evaluation.

Uses inference-free IDF-based query encoding and MLM-based document encoding
via the sentence-transformers SparseEncoder API.

Usage:
  python src/evaluate_splade.py --dataset kuhperdata-exp --save_embeddings
  python src/evaluate_splade.py --dataset all --save_embeddings
"""
import argparse
import time
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from util.dataloader import DataLoader
from util.metrics import (
    calculate_mrr, calculate_recall_at_k, calculate_precision_at_k,
    save_predictions,
)

MODEL_NAME = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"

DATASETS = {
    "kuhperdata-humanized": "data/kuhperdata-humanized",
    "kuhperdata-summarized": "data/kuhperdata-summarized",
    "kuhperdata-exp": "data/kuhperdata-exp",
    "kuhperdata-summ-exp": "data/kuhperdata-summ-exp",
    "bsard": "data/bsard",
    "coliee": "data/coliee",
    "stard": "data/stard",
}


def sparse_output_to_csr(embeddings, vocab_size: int) -> sp.csr_matrix:
    """Convert SparseEncoder output (list of dicts or sparse tensor) to CSR matrix."""
    import torch

    if isinstance(embeddings, torch.Tensor):
        if embeddings.is_sparse:
            embeddings = embeddings.coalesce().cpu()
            indices = embeddings.indices().numpy()
            values = embeddings.values().numpy()
            shape = (embeddings.shape[0], vocab_size)
            return sp.csr_matrix(
                (values, (indices[0], indices[1])), shape=shape,
            )
        else:
            return sp.csr_matrix(embeddings.cpu().numpy())

    if isinstance(embeddings, np.ndarray):
        return sp.csr_matrix(embeddings)

    if isinstance(embeddings, sp.spmatrix):
        return embeddings.tocsr()

    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], dict):
        rows, cols, vals = [], [], []
        for i, d in enumerate(embeddings):
            for token_id, weight in d.items():
                rows.append(i)
                cols.append(int(token_id))
                vals.append(float(weight))
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(embeddings), vocab_size),
        )

    raise TypeError(f"Unsupported embedding type: {type(embeddings)}")


def evaluate_sparse_retrieval(
    corpus_csr: sp.csr_matrix,
    query_csr: sp.csr_matrix,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict,
    top_k_values: tuple[int, ...] = (10, 50, 100),
):
    scores = (query_csr @ corpus_csr.T).toarray()

    max_k = max(top_k_values)
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :max_k]

    results = {}
    for k in top_k_values:
        per_query_mrr = []
        per_query_recall = []
        per_query_precision = []

        for i, qid in enumerate(query_ids):
            gt = list(qrels.get(qid, {}).keys())
            if not gt:
                continue
            ranked = [doc_ids[idx] for idx in top_indices[i, :k]]
            per_query_mrr.append(calculate_mrr(ranked, gt, k))
            per_query_recall.append(calculate_recall_at_k(ranked, gt, k))
            per_query_precision.append(calculate_precision_at_k(ranked, gt, k))

        results[k] = {
            "mrr": float(np.mean(per_query_mrr)),
            "recall": float(np.mean(per_query_recall)),
            "precision": float(np.mean(per_query_precision)),
            "n_queries": len(per_query_mrr),
            "hit_rate": sum(1 for m in per_query_mrr if m > 0) / len(per_query_mrr),
        }

    return results, scores, top_indices


def run_dataset(args, dataset_name: str, data_dir: str, model=None):
    corpus_path = f"{data_dir}/corpus.jsonl"
    queries_path = f"{data_dir}/queries.jsonl"
    qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

    print("=" * 60)
    print(f"Neural Sparse (SPLADE) — {dataset_name}")
    print("=" * 60)

    loader = DataLoader(corpus_path, queries_path, qrels_path).load()
    if args.max_relevant:
        before = len(loader.qrels)
        loader.filter_max_relevant(args.max_relevant)
        print(f"Filtered queries: {len(loader.qrels)} (from {before}, max_relevant={args.max_relevant})")

    doc_ids, doc_texts = loader.get_corpus_texts()
    test_query_ids = list(loader.qrels.keys())
    test_query_texts = [loader.queries[qid]["text"] for qid in test_query_ids]

    print(f"Corpus: {len(doc_ids)} documents")
    print(f"Test queries: {len(test_query_ids)}")

    emb_dir = Path(args.embeddings_dir)
    corpus_path_cache = emb_dir / f"splade_corpus_{dataset_name}.npz"
    query_path_cache = emb_dir / f"splade_queries_{dataset_name}.npz"
    query_ids_cache = emb_dir / f"splade_query_ids_{dataset_name}.json"

    cache_valid = False
    if corpus_path_cache.exists() and query_path_cache.exists() and query_ids_cache.exists():
        corpus_csr = sp.load_npz(corpus_path_cache)
        query_csr = sp.load_npz(query_path_cache)
        with open(query_ids_cache) as f:
            cached_qids = json.load(f)
        if corpus_csr.shape[0] == len(doc_ids) and cached_qids == test_query_ids:
            cache_valid = True
            print(f"Loading cached sparse embeddings...")
            print(f"  Corpus: {corpus_csr.shape}, nnz={corpus_csr.nnz}")
            print(f"  Queries: {query_csr.shape}, nnz={query_csr.nnz}")

    if not cache_valid:
        if model is None:
            from sentence_transformers.sparse_encoder import SparseEncoder
            print(f"\nLoading {MODEL_NAME}...")
            model = SparseEncoder(MODEL_NAME)

        print("Encoding corpus (MLM forward pass)...")
        t0 = time.time()
        doc_emb = model.encode_document(doc_texts, batch_size=args.batch_size)
        vocab_size = model.tokenizer.vocab_size
        corpus_csr = sparse_output_to_csr(doc_emb, vocab_size)
        print(f"  {corpus_csr.shape}, nnz={corpus_csr.nnz} in {time.time() - t0:.1f}s")
        print(f"  Avg nnz/doc: {corpus_csr.nnz / corpus_csr.shape[0]:.0f}")

        print("Encoding queries (inference-free IDF)...")
        t0 = time.time()
        query_emb = model.encode_query(test_query_texts, batch_size=args.batch_size)
        query_csr = sparse_output_to_csr(query_emb, vocab_size)
        print(f"  {query_csr.shape}, nnz={query_csr.nnz} in {time.time() - t0:.1f}s")

        if args.save_embeddings:
            emb_dir.mkdir(parents=True, exist_ok=True)
            sp.save_npz(corpus_path_cache, corpus_csr)
            sp.save_npz(query_path_cache, query_csr)
            with open(query_ids_cache, "w") as f:
                json.dump(test_query_ids, f)
            print(f"  Cached to {emb_dir}/")

    top_k_values = [10, 50, 100]
    results, scores, top_indices = evaluate_sparse_retrieval(
        corpus_csr, query_csr, doc_ids, test_query_ids, loader.qrels, top_k_values,
    )

    print(f"\n{'K':>6} | {'Recall@K':>10} | {'MRR@K':>8} | {'Precision@K':>12} | {'Hit Rate':>10}")
    print("-" * 60)
    for k in top_k_values:
        r = results[k]
        print(f"{k:>6} | {r['recall']:>10.4f} | {r['mrr']:>8.4f} | {r['precision']:>12.4f} | {r['hit_rate']:>9.1%}")

    save_k = 100
    top_save = np.argsort(scores, axis=1)[:, ::-1][:, :save_k]
    pred_rankings = {}
    pred_scores = {}
    ground_truth = {qid: list(loader.qrels.get(qid, {}).keys()) for qid in test_query_ids}
    for i, qid in enumerate(test_query_ids):
        ranked = [doc_ids[idx] for idx in top_save[i]]
        pred_rankings[qid] = ranked
        pred_scores[qid] = {doc_ids[idx]: float(scores[i, idx]) for idx in top_save[i]}

    save_predictions(
        pred_rankings, ground_truth,
        method="splade", dataset=dataset_name, scores=pred_scores,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenSearch Neural Sparse (SPLADE) evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--embeddings_dir", default="outputs/embeddings")
    parser.add_argument("--max_relevant", type=int, default=0)
    args = parser.parse_args()

    from sentence_transformers.sparse_encoder import SparseEncoder
    print(f"Loading {MODEL_NAME}...")
    model = SparseEncoder(MODEL_NAME)

    if args.dataset == "all":
        for name, path in DATASETS.items():
            run_dataset(args, name, path, model)
            print()
    elif args.dataset:
        run_dataset(args, args.dataset, DATASETS[args.dataset], model)
    else:
        parser.error("--dataset is required")


if __name__ == "__main__":
    main()
