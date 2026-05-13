"""Split a BEIR dataset to add a validation split via semantic clustering.

Two modes:

1. FROM TRAIN (default, --from_test not set):
   Takes val queries from the TRAIN split, scaling val size relative to TEST.
   Leaves qrels_test.tsv untouched (preserves benchmark comparability).
   
   Usage:
     python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-humanized --val_scale 1.0
     python src/scripts/split_test_to_val.py --dataset_dir data/bsard --val_scale 0.5

   --val_scale:  val_size = round(val_scale * n_test_queries)
                 1.0  → val ≈ test  (roughly 6:2:2)
                 0.5  → val ≈ half test  (roughly 7:1:2)

2. FROM TEST (--from_test flag):
   Splits qrels_test.tsv into qrels_val.tsv + reduced qrels_test.tsv.
   Legacy behavior from original script.
   
   Usage:
     python src/scripts/split_test_to_val.py --dataset_dir data/bsard --from_test --val_ratio 0.5
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

MIN_VAL_SIZE = 25


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


def semantic_val_split(
    qids: list[str],
    query_texts: list[str],
    n_val: int,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[list[str], list[str], dict]:
    """Split qids into val + remaining via KMeans + greedy assignment.

    Greedy: start all clusters in val, greedily move clusters to
    remaining to maximize cosine distance between val and remaining.
    """
    n = len(qids)
    if n_clusters is None:
        n_clusters = max(5, n // 10)
    n_clusters = min(n_clusters, n // 2)

    embeddings = embed_queries_for_splitting(query_texts)

    print(f"Clustering {n} queries into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_indices: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_indices[label].append(idx)

    cluster_sizes = {i: len(indices) for i, indices in cluster_indices.items()}
    centroid_distances = cosine_distances(kmeans.cluster_centers_)

    # Greedy: start all in val, move clusters to remaining maximizing distance
    val_clusters = set(range(n_clusters))
    remaining_clusters: set[int] = set()
    remaining_count = 0
    remaining_target = n - n_val

    while remaining_count < remaining_target and val_clusters:
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
        remaining_clusters.add(best_cluster)
        remaining_count += cluster_sizes[best_cluster]

    val_indices = []
    remaining_indices = []
    for cid in val_clusters:
        val_indices.extend(cluster_indices[cid])
    for cid in remaining_clusters:
        remaining_indices.extend(cluster_indices[cid])

    val_qids = [qids[i] for i in val_indices]
    remaining_qids = [qids[i] for i in remaining_indices]

    # Split statistics
    val_emb = embeddings[val_indices]
    remaining_emb = embeddings[remaining_indices]

    cross_dist = float(np.mean(cosine_distances(val_emb, remaining_emb)))
    val_self = cosine_distances(val_emb)
    val_self_dist = float(np.mean(val_self[np.triu_indices(len(val_indices), k=1)])) if len(val_indices) > 1 else 0
    remaining_self = cosine_distances(remaining_emb)
    remaining_self_dist = float(np.mean(remaining_self[np.triu_indices(len(remaining_indices), k=1)])) if len(remaining_indices) > 1 else 0

    info = {
        "n_val": len(val_qids),
        "n_remaining": len(remaining_qids),
        "n_clusters": n_clusters,
        "val_clusters": sorted(val_clusters),
        "remaining_clusters": sorted(remaining_clusters),
        "mean_val_cosine_distance": val_self_dist,
        "mean_remaining_cosine_distance": remaining_self_dist,
        "mean_cross_cosine_distance": cross_dist,
        "separation_ratio": cross_dist / max(val_self_dist, 1e-6),
    }

    print(f"\nSemantic Split:")
    print(f"  Val:       {len(val_qids)} queries ({len(val_clusters)} clusters)")
    print(f"  Remaining: {len(remaining_qids)} queries ({len(remaining_clusters)} clusters)")
    print(f"  Intra-val distance:        {val_self_dist:.4f}")
    print(f"  Intra-remaining distance:  {remaining_self_dist:.4f}")
    print(f"  Cross distance:            {cross_dist:.4f}")
    print(f"  Separation ratio:          {info['separation_ratio']:.2f}x")

    return val_qids, remaining_qids, info


def main():
    parser = argparse.ArgumentParser(
        description="Add a validation split to a BEIR-format dataset"
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Print stats without writing")

    # Mode: FROM TRAIN (default)
    parser.add_argument("--val_scale", type=float, default=None,
                        help="Val size = round(val_scale * n_test). "
                        "Val taken from TRAIN; TEST untouched. "
                        "Example: 1.0 → 6:2:2, 0.5 → 7:1:2. "
                        "Sets --from_train implicitly.")

    # Mode: FROM TEST (legacy)
    parser.add_argument("--from_test", action="store_true",
                        help="Split TEST instead of TRAIN. "
                        "Uses --val_ratio (not --val_scale).")
    parser.add_argument("--val_ratio", type=float, default=0.5,
                        help="Fraction of source split to use as val (default: 0.5). "
                        "Only used with --from_test.")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    queries_path = dataset_dir / "queries.jsonl"
    split_path = dataset_dir / "split_indices.json"

    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found")
        sys.exit(1)

    # Determine mode: from_train if --val_scale is set, else from_test if --from_test is set
    from_train = args.val_scale is not None
    from_test = args.from_test
    if not from_train and not from_test:
        print("ERROR: specify either --val_scale (from train) or --from_test (legacy)")
        sys.exit(1)

    if from_test:
        # --- FROM TEST: legacy mode ---
        qrels_test_path = dataset_dir / "qrels_test.tsv"
        qrels_val_path = dataset_dir / "qrels_val.tsv"

        if not qrels_test_path.exists():
            print(f"ERROR: {qrels_test_path} not found")
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

        n_val = int(len(all_test_qids) * args.val_ratio)
        val_qids, test_qids, info = semantic_val_split(
            all_test_qids, query_texts,
            n_val=n_val,
            random_state=args.seed,
        )

        if len(val_qids) < MIN_VAL_SIZE:
            print(f"ERROR: Val split has only {len(val_qids)} queries (minimum {MIN_VAL_SIZE})")
            sys.exit(1)

        if args.dry_run:
            print("\n[DRY RUN] No files written.")
            return

        write_qrels(qrels_val_path, qrels, set(val_qids))
        n_val_judgments = sum(len(qrels.get(q, [])) for q in val_qids)
        print(f"\nWrote {qrels_val_path} ({len(val_qids)} queries, {n_val_judgments} judgments)")

        write_qrels(qrels_test_path, qrels, set(test_qids))
        n_test_judgments = sum(len(qrels.get(q, [])) for q in test_qids)
        print(f"Wrote {qrels_test_path} ({len(test_qids)} queries, {n_test_judgments} judgments)")

        info["mode"] = "from_test"
        info["source"] = "test"
        info["val_ratio"] = args.val_ratio

    else:
        # --- FROM TRAIN: val taken from train, test untouched ---
        qrels_train_path = dataset_dir / "qrels_train.tsv"
        qrels_test_path = dataset_dir / "qrels_test.tsv"
        qrels_val_path = dataset_dir / "qrels_val.tsv"

        if not qrels_train_path.exists():
            print(f"ERROR: {qrels_train_path} not found")
            sys.exit(1)
        if not qrels_test_path.exists():
            print(f"ERROR: {qrels_test_path} not found")
            sys.exit(1)
        if qrels_val_path.exists():
            print(f"WARNING: {qrels_val_path} already exists, will overwrite")

        train_qrels = load_qrels(qrels_train_path)
        test_qrels = load_qrels(qrels_test_path)
        all_train_qids = sorted(train_qrels.keys())
        n_test = len(test_qrels)
        n_train = len(all_train_qids)

        n_val = round(args.val_scale * n_test)
        max_from_train = int(n_train * 0.8)
        if n_val > max_from_train:
            print(f"WARNING: val_scale={args.val_scale} gives n_val={n_val}, "
                  f"but only {max_from_train} available from train (80% cap). "
                  f"Clipping to {max_from_train}.")
            n_val = max_from_train
        n_val = max(n_val, MIN_VAL_SIZE)
        n_val = min(n_val, n_train - MIN_VAL_SIZE)

        print(f"Train: {n_train} queries | Test: {n_test} queries")
        print(f"Target val: {n_val} queries (={args.val_scale} × {n_test} test)")

        query_texts_map = load_query_texts(queries_path, set(all_train_qids))
        query_texts = [query_texts_map[qid] for qid in all_train_qids]

        val_qids, remaining_train_qids, info = semantic_val_split(
            all_train_qids, query_texts,
            n_val=n_val,
            random_state=args.seed,
        )

        if len(val_qids) < MIN_VAL_SIZE:
            print(f"ERROR: Val split has only {len(val_qids)} queries (minimum {MIN_VAL_SIZE})")
            sys.exit(1)

        if args.dry_run:
            print("\n[DRY RUN] No files written.")
            return

        write_qrels(qrels_val_path, train_qrels, set(val_qids))
        n_val_judgments = sum(len(train_qrels.get(q, [])) for q in val_qids)

        write_qrels(qrels_train_path, train_qrels, set(remaining_train_qids))
        n_new_train_judgments = sum(len(train_qrels.get(q, [])) for q in remaining_train_qids)

        print(f"\nWrote {qrels_val_path} ({len(val_qids)} queries, {n_val_judgments} judgments)")
        print(f"Wrote {qrels_train_path} ({len(remaining_train_qids)} queries, {n_new_train_judgments} judgments)")
        print(f"  (qrels_test.tsv unchanged — {n_test} queries)")

        effective_pct = {
            "train_pct": round(len(remaining_train_qids) / (len(remaining_train_qids) + n_val + n_test) * 100),
            "val_pct": round(n_val / (len(remaining_train_qids) + n_val + n_test) * 100),
            "test_pct": round(n_test / (len(remaining_train_qids) + n_val + n_test) * 100),
        }
        print(f"Effective ratio: {effective_pct['train_pct']}:{effective_pct['val_pct']}:{effective_pct['test_pct']} "
              f"(train:val:test)")

        info["mode"] = "from_train"
        info["source"] = "train"
        info["val_scale"] = args.val_scale
        info["effective_ratio"] = effective_pct

    # Update split_indices.json
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            split_data = json.load(f)
    else:
        split_data = {}

    split_data["val_query_ids"] = val_qids
    if from_train:
        split_data["train_query_ids"] = remaining_train_qids
        split_data["test_query_ids"] = sorted(test_qrels.keys())
        split_data["val_from"] = "train"
    else:
        split_data["test_query_ids"] = test_qids
        split_data["val_from"] = "test"
    split_data["val_split_info"] = info

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)
    print(f"Updated {split_path}")


if __name__ == "__main__":
    main()
