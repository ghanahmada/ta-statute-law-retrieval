"""
Statistical significance tests for retrieval method comparisons.

Reads predictions from the unified JSONL format (save_predictions output)
or Para-GNN/StructGNN inference JSONL, computes per-query metrics,
and runs paired statistical tests between methods.

Tests:
  - Wilcoxon signed-rank (non-parametric, preferred for bounded metrics)
  - Paired t-test (parametric, for reference)
  - Bootstrap 95% CI on mean MRR@10
  - Cohen's d effect size

Usage:
  python src/analysis/statistical_test.py \
    --predictions outputs/predictions/bm25_kuhperdata-exp.jsonl \
                  outputs/predictions/jnlp_kuhperdata-exp.jsonl \
                  outputs/predictions/agentic_kuhperdata-exp.jsonl \
    --names bm25 jnlp agentic \
    --output_dir outputs/analysis/stat_tests
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats


def load_predictions(path: str) -> dict[str, list[str]]:
    """Load predictions JSONL → {qid: [ranked_doc_ids]}."""
    rankings = {}
    ground_truth = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["qid"]
            if "ranked_doc_ids" in rec:
                rankings[qid] = rec["ranked_doc_ids"]
            elif "rankings" in rec:
                rankings[qid] = [r["doc_id"] for r in rec["rankings"]]
            if "ground_truth" in rec:
                gt = rec["ground_truth"]
                if isinstance(gt, list) and gt and isinstance(gt[0], str):
                    ground_truth[qid] = gt
                elif isinstance(gt, list) and gt and isinstance(gt[0], dict):
                    ground_truth[qid] = [d["doc_id"] for d in gt]
    return rankings, ground_truth


def compute_per_query_rr(rankings: dict, ground_truth: dict, k: int = 10) -> dict[str, float]:
    """Compute reciprocal rank per query."""
    rr = {}
    for qid in ground_truth:
        gt = set(ground_truth[qid])
        ranked = rankings.get(qid, [])[:k]
        rr[qid] = 0.0
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in gt:
                rr[qid] = 1.0 / rank
                break
    return rr


def compute_per_query_recall(rankings: dict, ground_truth: dict, k: int = 10) -> dict[str, float]:
    recall = {}
    for qid in ground_truth:
        gt = set(ground_truth[qid])
        if not gt:
            recall[qid] = 0.0
            continue
        ranked = set(rankings.get(qid, [])[:k])
        recall[qid] = len(ranked & gt) / len(gt)
    return recall


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def run_comparison(name_a, rr_a, name_b, rr_b, qids):
    a = np.array([rr_a[q] for q in qids])
    b = np.array([rr_b[q] for q in qids])

    mean_a = a.mean()
    mean_b = b.mean()
    ci_a = bootstrap_ci(a)
    ci_b = bootstrap_ci(b)

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    d = cohens_d(a, b)

    return {
        "pair": f"{name_a} vs {name_b}",
        "n_queries": len(qids),
        f"mean_mrr_{name_a}": round(mean_a, 4),
        f"mean_mrr_{name_b}": round(mean_b, 4),
        f"ci95_{name_a}": [round(ci_a[0], 4), round(ci_a[1], 4)],
        f"ci95_{name_b}": [round(ci_b[0], 4), round(ci_b[1], 4)],
        "delta": round(mean_a - mean_b, 4),
        "cohens_d": round(d, 4),
        "ttest_p": round(t_p, 6),
        "wilcoxon_p": round(w_p, 6),
        "significant_005": w_p < 0.05 if not np.isnan(w_p) else None,
        "significant_001": w_p < 0.01 if not np.isnan(w_p) else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical tests on retrieval predictions")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="Paths to prediction JSONL files")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Method names (same order as --predictions)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/stat_tests")
    args = parser.parse_args()

    assert len(args.predictions) == len(args.names), \
        "Must provide same number of --predictions and --names"

    method_rr = {}
    all_gt = {}

    for path, name in zip(args.predictions, args.names):
        rankings, gt = load_predictions(path)
        if gt:
            all_gt.update(gt)
        rr = compute_per_query_rr(rankings, gt or all_gt, k=args.top_k)
        method_rr[name] = rr
        ci = bootstrap_ci(np.array(list(rr.values())))
        print(f"{name}: MRR@{args.top_k} = {np.mean(list(rr.values())):.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] ({len(rr)} queries)")

    shared_qids = set.intersection(*[set(rr.keys()) for rr in method_rr.values()])
    print(f"\nShared queries across all methods: {len(shared_qids)}")

    results = []
    print(f"\n{'=' * 80}")
    print(f"Pairwise comparisons (Wilcoxon signed-rank, k={args.top_k})")
    print(f"{'=' * 80}")

    for (name_a, rr_a), (name_b, rr_b) in combinations(method_rr.items(), 2):
        pair_qids = sorted(set(rr_a.keys()) & set(rr_b.keys()))
        if not pair_qids:
            print(f"\n  {name_a} vs {name_b}: no shared queries, skipping")
            continue

        res = run_comparison(name_a, rr_a, name_b, rr_b, pair_qids)
        results.append(res)

        sig = "***" if res["significant_001"] else ("**" if res["significant_005"] else "ns")
        print(f"\n  {res['pair']} (n={res['n_queries']})")
        print(f"    Δ MRR@{args.top_k} = {res['delta']:+.4f}  Cohen's d = {res['cohens_d']:.3f}")
        print(f"    Wilcoxon p = {res['wilcoxon_p']:.6f}  t-test p = {res['ttest_p']:.6f}  [{sig}]")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/stat_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
