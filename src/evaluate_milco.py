"""MILCO learned sparse retrieval evaluation.

Uses a multilingual connector to map all languages into a shared English
lexical space, with LexEcho head to preserve source-language entities.

Usage:
  python src/evaluate_milco.py --dataset kuhperdata-exp --save_embeddings
  python src/evaluate_milco.py --dataset all --save_embeddings
"""
import argparse
import time
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path

import torch

from util.dataloader import DataLoader
from util.metrics import (
    calculate_mrr, calculate_recall_at_k, calculate_precision_at_k,
    save_predictions,
)

MODEL_NAME = "omai-research/milco-650m"

DATASETS = {
    "kuhperdata-humanized": "data/kuhperdata-humanized",
    "kuhperdata-summarized": "data/kuhperdata-summarized",
    "kuhperdata-exp": "data/kuhperdata-exp",
    "kuhperdata-summ-exp": "data/kuhperdata-summ-exp",
    "bsard": "data/bsard",
    "coliee": "data/coliee",
    "stard": "data/stard",
}


def sparse_coo_to_csr(tensor: torch.Tensor) -> sp.csr_matrix:
    """Convert torch sparse COO tensor to scipy CSR matrix."""
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices().numpy()
    values = tensor.values().numpy()
    shape = tuple(tensor.shape)
    return sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)


def encode_batched(texts: list[str], model, batch_size: int, method: str = "encode_text") -> sp.csr_matrix:
    """Encode texts in batches and stack into a single CSR matrix."""
    encode_fn = getattr(model, method)
    parts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        sparse_tensor = encode_fn(batch)
        parts.append(sparse_coo_to_csr(sparse_tensor))
    return sp.vstack(parts, format="csr")


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


def run_dataset(args, dataset_name: str, data_dir: str, model):
    corpus_path = f"{data_dir}/corpus.jsonl"
    queries_path = f"{data_dir}/queries.jsonl"
    qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

    print("=" * 60)
    print(f"MILCO — {dataset_name}")
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
    corpus_path_cache = emb_dir / f"milco_corpus_{dataset_name}.npz"
    query_path_cache = emb_dir / f"milco_queries_{dataset_name}.npz"
    query_ids_cache = emb_dir / f"milco_query_ids_{dataset_name}.json"

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
        print("Encoding corpus...")
        t0 = time.time()
        corpus_csr = encode_batched(doc_texts, model, args.batch_size, "encode_document")
        print(f"  {corpus_csr.shape}, nnz={corpus_csr.nnz} in {time.time() - t0:.1f}s")
        print(f"  Avg nnz/doc: {corpus_csr.nnz / corpus_csr.shape[0]:.0f}")

        print("Encoding queries...")
        t0 = time.time()
        query_csr = encode_batched(test_query_texts, model, args.batch_size, "encode_query")
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

    print(f"\n{'K':>6} | {'MRR@K':>8} | {'Recall@K':>10} | {'Precision@K':>12} | {'Hit Rate':>10}")
    print("-" * 60)
    for k in top_k_values:
        r = results[k]
        print(f"{k:>6} | {r['mrr']:>8.4f} | {r['recall']:>10.4f} | {r['precision']:>12.4f} | {r['hit_rate']:>9.1%}")

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
        method="milco", dataset=dataset_name, scores=pred_scores,
    )


def main():
    parser = argparse.ArgumentParser(description="MILCO learned sparse retrieval evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--embeddings_dir", default="outputs/embeddings")
    parser.add_argument("--max_relevant", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModel
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
