"""Evaluate agentic retrieval from agent_log.jsonl against qrels.

Usage:
  python src/context_1/eval_agent_log.py \
    --log outputs/context_1/kuhperdata-exp/agent_log.jsonl \
    --dataset kuhperdata-exp

  # Or specify qrels directly:
  python src/context_1/eval_agent_log.py \
    --log outputs/context_1/kuhperdata-exp/agent_log.jsonl \
    --qrels datasets/kuhperdata-exp/qrels_test.tsv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.metrics import evaluate_ranking


DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


def load_qrels(path: str) -> dict[str, list[str]]:
    gt = {}
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id, score = parts[0], parts[1], int(parts[2])
                if score > 0:
                    gt.setdefault(qid, []).append(doc_id)
    return gt


def load_agent_log(path: str) -> dict[str, list[str]]:
    rankings = {}
    stats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["qid"]
            rankings[qid] = rec["ranked_doc_ids"]
            stats.append(rec)
    return rankings, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to agent_log.jsonl")
    parser.add_argument("--dataset", help="Dataset name (resolves qrels automatically)")
    parser.add_argument("--qrels", help="Path to qrels_test.tsv (overrides --dataset)")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if args.qrels:
        qrels_path = args.qrels
    elif args.dataset:
        qrels_path = str(DATASET_DIR / args.dataset / "qrels_test.tsv")
    else:
        parser.error("Provide --qrels or --dataset")

    ground_truth = load_qrels(qrels_path)
    rankings, stats = load_agent_log(args.log)

    evaluated_gt = {qid: docs for qid, docs in ground_truth.items() if qid in rankings}

    print(f"Log entries:      {len(rankings)}")
    print(f"Queries w/ qrels: {len(evaluated_gt)}")
    print(f"Total qrels:      {len(ground_truth)}")

    if not evaluated_gt:
        print("No overlapping queries between log and qrels.")
        return

    k = args.top_k
    metrics = evaluate_ranking(rankings, evaluated_gt, k)
    print(f"\n{'='*40}")
    print(f"  MRR@{k}:       {metrics[f'mrr@{k}']:.4f}")
    print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
    print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f"  Hit Rate:      {metrics['hit_rate']:.4f}")
    print(f"  Queries:       {metrics['n_queries']}")
    print(f"{'='*40}")

    if stats:
        errors = [s for s in stats if s.get("error")]
        turns = [s["turns"] for s in stats]
        n_selected = [s["n_selected"] for s in stats]
        n_seen = [s["n_seen"] for s in stats]
        n_read = [s.get("n_read", 0) for s in stats]
        elapsed = [s["elapsed_s"] for s in stats]

        print(f"\nAgent stats ({len(stats)} queries):")
        print(f"  Avg turns:    {np.mean(turns):.1f}")
        print(f"  Avg selected: {np.mean(n_selected):.1f}")
        print(f"  Avg seen:     {np.mean(n_seen):.1f}")
        print(f"  Avg read:     {np.mean(n_read):.1f}")
        print(f"  Avg time:     {np.mean(elapsed):.1f}s")
        if errors:
            print(f"  Errors:       {len(errors)}")

    per_mrr = metrics["per_query_mrr"]
    zero_mrr_qids = [qid for qid, m in zip(evaluated_gt.keys(), per_mrr) if m == 0]
    if zero_mrr_qids:
        print(f"\n{len(zero_mrr_qids)} queries with MRR=0 (no relevant doc in top {k})")


if __name__ == "__main__":
    main()
