"""Evaluate SAILER model on kuhperdata test set.

Uses precomputed embeddings to perform FAISS retrieval,
then evaluates against qrels_test.tsv with NDCG@10, Recall@10, MAP.

Usage:
    python src/scripts/sailer/evaluate_retrieval.py
    python src/scripts/sailer/evaluate_retrieval.py --embeddings_dir outputs/sailer_base/embeddings
"""

import argparse
import json
import os

import faiss
import torch
import numpy as np
import pytrec_eval

QRELS_PATH = os.path.join("data", "kuhperdata", "qrels_test.tsv")
TOP_K = 100  # Retrieve top-K for evaluation


def load_embeddings(path: str) -> tuple[list[str], np.ndarray]:
    """Load encoded embeddings from pickle file.

    SAILER's encode driver saves as (tensor, lookup_indices) via torch.save.
    """
    embeddings_tensor, ids = torch.load(path, map_location="cpu")
    embeddings = embeddings_tensor.float().numpy()
    return ids, embeddings


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """Load qrels into pytrec_eval format: {query_id: {doc_id: score}}."""
    qrels = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = score
    return qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        default=os.path.join("outputs", "sailer_kuhperdata", "embeddings"),
        help="Directory containing corpus_emb.pkl and query_emb.pkl",
    )
    args = parser.parse_args()
    embeddings_dir = args.embeddings_dir

    print("Loading embeddings...")
    corpus_ids, corpus_emb = load_embeddings(os.path.join(embeddings_dir, "corpus_emb.pkl"))
    query_ids, query_emb = load_embeddings(os.path.join(embeddings_dir, "query_emb.pkl"))
    print(f"  Corpus: {len(corpus_ids)} docs, dim={corpus_emb.shape[1]}")
    print(f"  Queries: {len(query_ids)} queries")

    # Normalize for cosine similarity
    faiss.normalize_L2(corpus_emb)
    faiss.normalize_L2(query_emb)

    # Build FAISS index
    print("Building FAISS index...")
    dim = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
    index.add(corpus_emb)

    # Search
    print(f"Searching top-{TOP_K}...")
    scores, indices = index.search(query_emb, TOP_K)

    # Build run dict for pytrec_eval
    run = {}
    for i, qid in enumerate(query_ids):
        run[qid] = {}
        for j in range(TOP_K):
            if indices[i][j] >= 0:
                did = corpus_ids[indices[i][j]]
                run[qid][did] = float(scores[i][j])

    # Load qrels and evaluate
    print("Loading qrels...")
    qrels = load_qrels(QRELS_PATH)
    print(f"  {len(qrels)} test queries with relevance judgments")

    # Filter run to only queries in qrels
    run_filtered = {qid: docs for qid, docs in run.items() if qid in qrels}
    print(f"  {len(run_filtered)} queries matched between run and qrels")

    # Evaluate
    metrics = {"ndcg_cut_10", "recall_10", "map", "recip_rank"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run_filtered)

    # Aggregate
    metric_names = ["ndcg_cut_10", "recall_10", "map", "recip_rank"]
    print("\n" + "=" * 50)
    print("SAILER Retrieval Evaluation (kuhperdata test set)")
    print("=" * 50)

    aggregated = {}
    for metric in metric_names:
        values = [results[qid][metric] for qid in results]
        mean = np.mean(values) if values else 0.0
        aggregated[metric] = mean
        label = "mrr_10" if metric == "recip_rank" else metric
        print(f"  {label:20s}: {mean:.4f}")

    print("=" * 50)
    print(f"  Evaluated on {len(results)} queries")

    # Save results
    results_path = os.path.join(embeddings_dir, "eval_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"aggregated": aggregated, "per_query": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()
