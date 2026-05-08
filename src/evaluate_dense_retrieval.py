import argparse
import time
import numpy as np
from pathlib import Path

from util.dataloader import DataLoader
from util.metrics import calculate_mrr, calculate_recall_at_k, calculate_precision_at_k, save_predictions


def encode_with_bge(texts, model, batch_size=64, max_length=1024):
    safe_texts = [t if t.strip() else "empty" for t in texts]
    output = model.encode(safe_texts, batch_size=batch_size, max_length=max_length)
    if isinstance(output, dict):
        return np.array(output["dense_vecs"])
    return np.array(output)


def evaluate_cosine_retrieval(
    corpus_embeddings,
    query_embeddings,
    doc_ids,
    query_ids,
    qrels,
    top_k_values=(10, 50, 100),
):
    """Evaluate raw cosine similarity retrieval at multiple K values."""
    # corpus_embeddings: (N_docs, D), already L2-normalized by BGE-M3
    # query_embeddings: (N_queries, D)
    # cosine sim = dot product for L2-normalized vectors
    scores = query_embeddings @ corpus_embeddings.T  # (N_queries, N_docs)

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

    return results


DATASETS = {
    "kuhperdata-humanized": "data/kuhperdata-humanized",
    "kuhperdata-summarized": "data/kuhperdata-summarized",
    "kuhperdata-exp": "data/kuhperdata-exp",
    "kuhperdata-summ-exp": "data/kuhperdata-summ-exp",
    "bsard": "data/bsard",
    "ilpcsr": "data/ilpcsr",
    "stard": "data/stard",
}


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 cosine similarity retrieval PoC")
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS.keys(),
                        help="Dataset name (overrides --corpus_path/--queries_path/--qrels_test_path)")
    parser.add_argument("--corpus_path", default="data/kuhperdata/corpus.jsonl")
    parser.add_argument("--queries_path", default="data/kuhperdata/queries.jsonl")
    parser.add_argument("--qrels_test_path", default="data/kuhperdata/qrels_test.tsv")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--bge_model", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--save_embeddings", action="store_true", help="Cache embeddings to disk")
    parser.add_argument("--embeddings_dir", default="outputs/embeddings")
    parser.add_argument("--max_relevant", type=int, default=5,
                        help="Max ground-truth docs per query (queries with more are excluded)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Path to save per-query top-100 predictions as JSONL")
    args = parser.parse_args()

    if args.dataset:
        data_dir = DATASETS[args.dataset]
        args.corpus_path = f"{data_dir}/corpus.jsonl"
        args.queries_path = f"{data_dir}/queries.jsonl"
        args.qrels_test_path = f"{data_dir}/qrels_{args.split}.tsv"

    print("=" * 60)
    print(f"BGE-M3 Dense Retrieval{f' — {args.dataset}' if args.dataset else ''}")
    print("=" * 60)

    # --- Load data ---
    loader = DataLoader(args.corpus_path, args.queries_path, args.qrels_test_path).load()
    if args.max_relevant:
        before = len(loader.qrels)
        loader.filter_max_relevant(args.max_relevant)
        print(f"Filtered queries: {len(loader.qrels)} (from {before}, max_relevant={args.max_relevant})")
    doc_ids, doc_texts = loader.get_corpus_texts()
    test_query_ids = list(loader.qrels.keys())
    test_query_texts = [loader.queries[qid]["text"] for qid in test_query_ids]

    print(f"Corpus: {len(doc_ids)} documents")
    print(f"Test queries: {len(test_query_ids)}")
    print(f"Avg query length: {np.mean([len(t) for t in test_query_texts]):.0f} chars")
    print(f"Avg doc length: {np.mean([len(t) for t in doc_texts]):.0f} chars")

    # --- Check for cached embeddings ---
    emb_dir = Path(args.embeddings_dir)
    corpus_emb_path = emb_dir / "bge_m3_corpus.npy"
    query_emb_path = emb_dir / f"bge_m3_test_queries.npy"
    query_ids_path = emb_dir / f"bge_m3_test_query_ids.npy"

    cache_valid = False
    if corpus_emb_path.exists() and query_emb_path.exists():
        corpus_embeddings = np.load(corpus_emb_path)
        query_embeddings = np.load(query_emb_path)
        if corpus_embeddings.shape[0] == len(doc_ids) and query_embeddings.shape[0] == len(test_query_ids):
            cache_valid = True
            print(f"\nLoading cached embeddings...")
            print(f"  Corpus: {corpus_embeddings.shape}, Queries: {query_embeddings.shape}")
        else:
            print(f"\nStale cache (corpus: {corpus_embeddings.shape[0]} vs {len(doc_ids)}, "
                  f"queries: {query_embeddings.shape[0]} vs {len(test_query_ids)}). Re-encoding...")

    if not cache_valid:
        import torch
        devices = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading BGE-M3 ({args.bge_model}) on {devices}...")
        import util.compat  # noqa: F401 — patch transformers for FlagEmbedding compat
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(args.bge_model, use_fp16=True, devices=devices)

        print("Encoding corpus...")
        t0 = time.time()
        corpus_embeddings = encode_with_bge(doc_texts, model, args.batch_size, args.max_length)
        print(f"  {corpus_embeddings.shape} in {time.time() - t0:.1f}s")

        print("Encoding test queries...")
        t0 = time.time()
        query_embeddings = encode_with_bge(test_query_texts, model, args.batch_size, args.max_length)
        print(f"  {query_embeddings.shape} in {time.time() - t0:.1f}s")

        if args.save_embeddings:
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(corpus_emb_path, corpus_embeddings)
            np.save(query_emb_path, query_embeddings)
            print(f"  Saved to {emb_dir}/")

    # --- Check embedding norms (should be ~1.0 if L2-normalized) ---
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norms = np.linalg.norm(query_embeddings, axis=1)
    print(f"\nEmbedding norms (should be ~1.0 if L2-normalized):")
    print(f"  Corpus: mean={corpus_norms.mean():.4f}, std={corpus_norms.std():.6f}")
    print(f"  Queries: mean={query_norms.mean():.4f}, std={query_norms.std():.6f}")

    # --- Check L1 distance distribution (diagnose histogram range) ---
    sample_q = query_embeddings[0]
    sample_l1 = np.abs(corpus_embeddings - sample_q.reshape(1, -1))
    print(f"\nL1 distance stats (sample query vs all docs):")
    print(f"  min={sample_l1.min():.4f}, max={sample_l1.max():.4f}, "
          f"mean={sample_l1.mean():.4f}, std={sample_l1.std():.4f}")
    print(f"  % values in (0, 0.5): {(sample_l1 < 0.5).mean():.1%}")
    print(f"  % values in (0.5, 1): {((sample_l1 >= 0.5) & (sample_l1 < 1)).mean():.1%}")
    print(f"  % values in (1, 2):   {(sample_l1 >= 1).mean():.1%}")

    # --- Evaluate cosine similarity retrieval ---
    print("\n" + "=" * 60)
    print("RESULTS: Raw Cosine Similarity Retrieval")
    print("=" * 60)

    top_k_values = [10, 50, 100]
    results = evaluate_cosine_retrieval(
        corpus_embeddings, query_embeddings,
        doc_ids, test_query_ids, loader.qrels,
        top_k_values=top_k_values,
    )

    print(f"\n{'K':>6} | {'MRR@K':>8} | {'Recall@K':>10} | {'Precision@K':>12} | {'Hit Rate':>10}")
    print("-" * 60)
    for k in top_k_values:
        r = results[k]
        print(f"{k:>6} | {r['mrr']:>8.4f} | {r['recall']:>10.4f} | {r['precision']:>12.4f} | {r['hit_rate']:>9.1%}")

    sim_scores_full = query_embeddings @ corpus_embeddings.T
    save_k = 100
    top_save_indices = np.argsort(sim_scores_full, axis=1)[:, ::-1][:, :save_k]
    pred_rankings = {}
    pred_scores = {}
    ground_truth = {qid: list(loader.qrels.get(qid, {}).keys()) for qid in test_query_ids}
    for i, qid in enumerate(test_query_ids):
        ranked = [doc_ids[idx] for idx in top_save_indices[i]]
        pred_rankings[qid] = ranked
        pred_scores[qid] = {doc_ids[idx]: float(sim_scores_full[i, idx]) for idx in top_save_indices[i]}
    save_predictions(
        pred_rankings, ground_truth,
        args.save_predictions.format(dataset=args.dataset or "dense") if args.save_predictions else None,
        method="dense_bge_m3", dataset=args.dataset or "", scores=pred_scores,
    )

    # --- Cosine similarity score distribution ---
    sim_scores = query_embeddings @ corpus_embeddings.T
    print(f"\nCosine similarity distribution (all query-doc pairs):")
    print(f"  min={sim_scores.min():.4f}, max={sim_scores.max():.4f}, "
          f"mean={sim_scores.mean():.4f}, std={sim_scores.std():.4f}")

    # Relevant vs non-relevant score gap
    rel_scores = []
    nonrel_scores = []
    for i, qid in enumerate(test_query_ids):
        gt = set(loader.qrels.get(qid, {}).keys())
        for j, did in enumerate(doc_ids):
            if did in gt:
                rel_scores.append(sim_scores[i, j])
            # sample some non-relevant for speed
            elif j % 50 == 0:
                nonrel_scores.append(sim_scores[i, j])

    rel_scores = np.array(rel_scores)
    nonrel_scores = np.array(nonrel_scores)
    print(f"\n  Relevant pairs:     mean={rel_scores.mean():.4f}, std={rel_scores.std():.4f}")
    print(f"  Non-relevant pairs: mean={nonrel_scores.mean():.4f}, std={nonrel_scores.std():.4f}")
    print(f"  Score gap:          {rel_scores.mean() - nonrel_scores.mean():.4f}")
    if rel_scores.std() + nonrel_scores.std() > 0:
        separability = (rel_scores.mean() - nonrel_scores.mean()) / (rel_scores.std() + nonrel_scores.std())
        print(f"  Separability index: {separability:.4f} (higher = easier to rank)")


if __name__ == "__main__":
    main()
