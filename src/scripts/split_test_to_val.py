"""Split existing qrels_test.tsv into qrels_val.tsv + qrels_test.tsv using semantic clustering.

Maximizes cosine distance between val and test clusters while keeping
intra-split queries topically similar. Same algorithm as the train/test
split in dataset.py.

Usage:
  python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-humanized
  python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-exp
  python src/scripts/split_test_to_val.py --dataset_dir data/bsard
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import embed_queries_for_splitting

MIN_VAL_SIZE = 50


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0] != "query_id":
                qid, doc_id = parts[0], parts[1]
                qrels.setdefault(qid, []).append(doc_id)
    return qrels


def load_query_texts(queries_path: Path, qids: set[str]) -> dict[str, str]:
    texts = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["_id"] in qids:
                texts[entry["_id"]] = entry["text"]
    return texts


def write_qrels(path: Path, qrels: dict[str, list[str]], qids: set[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid in sorted(qids):
            for doc_id in qrels.get(qid, []):
                f.write(f"{qid}\t{doc_id}\t1\n")


def semantic_val_test_split(
    qids: list[str],
    query_texts: list[str],
    val_ratio: float = 0.5,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[list[str], list[str], dict]:
    n = len(qids)
    n_val = int(n * val_ratio)

    if n_clusters is None:
        n_clusters = max(5, n // 10)
    n_clusters = min(n_clusters, n // 2)

    embeddings = embed_queries_for_splitting(query_texts)

    print(f"Clustering {n} test queries into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_indices: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_indices[label].append(idx)

    cluster_sizes = {i: len(indices) for i, indices in cluster_indices.items()}
    centroid_distances = cosine_distances(kmeans.cluster_centers_)

    # Greedy: start all in val, move clusters to test maximizing distance
    val_clusters = set(range(n_clusters))
    test_clusters: set[int] = set()
    test_count = 0

    while test_count < (n - n_val) and val_clusters:
        best_cluster = None
        best_min_distance = -1

        for candidate in val_clusters:
            remaining_val = val_clusters - {candidate}
            if not remaining_val:
                continue
            min_dist = min(centroid_distances[candidate, v] for v in remaining_val)
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_cluster = candidate

        if best_cluster is None:
            break

        val_clusters.remove(best_cluster)
        test_clusters.add(best_cluster)
        test_count += cluster_sizes[best_cluster]

    val_indices = []
    test_indices = []
    for cid in val_clusters:
        val_indices.extend(cluster_indices[cid])
    for cid in test_clusters:
        test_indices.extend(cluster_indices[cid])

    val_qids = [qids[i] for i in val_indices]
    test_qids = [qids[i] for i in test_indices]

    # Compute split statistics
    val_emb = embeddings[val_indices]
    test_emb = embeddings[test_indices]

    cross_dist = float(np.mean(cosine_distances(val_emb, test_emb)))
    val_self = cosine_distances(val_emb)
    val_dist = float(np.mean(val_self[np.triu_indices(len(val_indices), k=1)]))
    test_self = cosine_distances(test_emb)
    test_dist = float(np.mean(test_self[np.triu_indices(len(test_indices), k=1)]))

    info = {
        "n_val": len(val_qids),
        "n_test": len(test_qids),
        "n_clusters": n_clusters,
        "val_clusters": sorted(val_clusters),
        "test_clusters": sorted(test_clusters),
        "mean_val_cosine_distance": val_dist,
        "mean_test_cosine_distance": test_dist,
        "mean_val_test_cosine_distance": cross_dist,
        "separation_ratio": cross_dist / max(val_dist, 1e-6),
    }

    print(f"\nSemantic Val/Test Split:")
    print(f"  Val:  {len(val_qids)} queries ({len(val_clusters)} clusters)")
    print(f"  Test: {len(test_qids)} queries ({len(test_clusters)} clusters)")
    print(f"  Intra-val distance:  {val_dist:.4f}")
    print(f"  Intra-test distance: {test_dist:.4f}")
    print(f"  Val↔Test distance:   {cross_dist:.4f}")
    print(f"  Separation ratio:    {info['separation_ratio']:.2f}x")

    return val_qids, test_qids, info


def main():
    parser = argparse.ArgumentParser(description="Split qrels_test.tsv into val + test")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    qrels_test_path = dataset_dir / "qrels_test.tsv"
    qrels_val_path = dataset_dir / "qrels_val.tsv"
    queries_path = dataset_dir / "queries.jsonl"
    split_path = dataset_dir / "split_indices.json"

    if not qrels_test_path.exists():
        print(f"ERROR: {qrels_test_path} not found")
        sys.exit(1)
    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found")
        sys.exit(1)
    if qrels_val_path.exists():
        print(f"WARNING: {qrels_val_path} already exists, will overwrite")

    qrels = load_qrels(qrels_test_path)
    all_test_qids = sorted(qrels.keys())
    print(f"Loaded {len(all_test_qids)} test queries from {qrels_test_path}")

    if len(all_test_qids) < MIN_VAL_SIZE * 2:
        print(f"ERROR: Only {len(all_test_qids)} test queries — need at least {MIN_VAL_SIZE * 2} for a meaningful val/test split")
        sys.exit(1)

    query_texts_map = load_query_texts(queries_path, set(all_test_qids))
    query_texts = [query_texts_map[qid] for qid in all_test_qids]

    val_qids, test_qids, info = semantic_val_test_split(
        all_test_qids, query_texts,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    if len(val_qids) < MIN_VAL_SIZE:
        print(f"ERROR: Val split has only {len(val_qids)} queries (minimum {MIN_VAL_SIZE})")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write qrels_val.tsv
    write_qrels(qrels_val_path, qrels, set(val_qids))
    n_val_judgments = sum(len(qrels.get(q, [])) for q in val_qids)
    print(f"\nWrote {qrels_val_path} ({len(val_qids)} queries, {n_val_judgments} judgments)")

    # Overwrite qrels_test.tsv
    write_qrels(qrels_test_path, qrels, set(test_qids))
    n_test_judgments = sum(len(qrels.get(q, [])) for q in test_qids)
    print(f"Wrote {qrels_test_path} ({len(test_qids)} queries, {n_test_judgments} judgments)")

    # Update split_indices.json
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            split_data = json.load(f)
    else:
        split_data = {}

    split_data["val_query_ids"] = val_qids
    split_data["test_query_ids"] = test_qids
    split_data["val_test_split_info"] = info

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)
    print(f"Updated {split_path}")


if __name__ == "__main__":
    main()
