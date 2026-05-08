from typing import Dict, List, Any, Optional
import json
import os

import numpy as np


def calculate_mrr(ranked_doc_ids: List[str], ground_truth_ids: List[str], top_k: int) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""
    for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def calculate_recall_at_k(ranked_doc_ids: List[str], ground_truth_ids: List[str], top_k: int) -> float:
    """Calculate Recall@K for a single query."""
    if not ground_truth_ids:
        return 0.0
    top_k_docs = set(ranked_doc_ids[:top_k])
    relevant_found = len(top_k_docs.intersection(set(ground_truth_ids)))
    return relevant_found / len(ground_truth_ids)


def calculate_precision_at_k(ranked_doc_ids: List[str], ground_truth_ids: List[str], top_k: int) -> float:
    """Calculate Precision@K for a single query."""
    top_k_docs = ranked_doc_ids[:top_k]
    if not top_k_docs:
        return 0.0
    relevant_found = len(set(top_k_docs).intersection(set(ground_truth_ids)))
    return relevant_found / len(top_k_docs)


def save_predictions(
    rankings: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    output_path: str,
    method: str = "",
    dataset: str = "",
    scores: Optional[Dict[str, Dict[str, float]]] = None,
    top_k: int = 100,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in ground_truth:
            ranked = rankings.get(qid, [])[:top_k]
            gt = ground_truth.get(qid, [])
            rec = {
                "qid": qid,
                "method": method,
                "dataset": dataset,
                "ranked_doc_ids": ranked,
                "ground_truth": gt,
                "rr@10": calculate_mrr(ranked, gt, 10),
                "recall@10": calculate_recall_at_k(ranked, gt, 10),
            }
            if scores and qid in scores:
                qid_scores = scores[qid]
                rec["doc_scores"] = {d: round(qid_scores.get(d, 0.0), 6) for d in ranked}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Predictions saved: {output_path} ({len(ground_truth)} queries, top-{top_k})")


def evaluate_ranking(
    rankings: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Evaluate ranked retrieval results.

    Args:
        rankings: Dict mapping query_id to ranked list of doc_ids
        ground_truth: Dict mapping query_id to list of relevant doc_ids
        top_k: Cutoff for metrics

    Returns:
        Dict with mean metrics and per-query results
    """
    per_query_mrr = []
    per_query_recall = []
    per_query_precision = []

    for qid in ground_truth.keys():
        gt = ground_truth.get(qid, [])
        ranked = rankings.get(qid, [])

        per_query_mrr.append(calculate_mrr(ranked, gt, top_k))
        per_query_recall.append(calculate_recall_at_k(ranked, gt, top_k))
        per_query_precision.append(calculate_precision_at_k(ranked, gt, top_k))

    return {
        f"mrr@{top_k}": float(np.mean(per_query_mrr)),
        f"recall@{top_k}": float(np.mean(per_query_recall)),
        f"precision@{top_k}": float(np.mean(per_query_precision)),
        "per_query_mrr": per_query_mrr,
        "per_query_recall": per_query_recall,
        "per_query_precision": per_query_precision,
        "n_queries": len(per_query_mrr),
        "hit_rate": sum(1 for m in per_query_mrr if m > 0) / len(per_query_mrr) if per_query_mrr else 0
    }
