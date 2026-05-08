"""
Statistical significance tests for retrieval method comparisons.

Reads predictions from the unified JSONL format (save_predictions output)
or Para-GNN/StructGNN inference JSONL, computes per-query metrics,
and runs paired statistical tests between methods.

Tests:
  - Friedman test (omnibus, >2 methods) — are any methods significantly different?
  - Wilcoxon signed-rank (pairwise, non-parametric) with Holm-Bonferroni correction
  - Paired t-test (parametric, for reference)
  - Bootstrap 95% CI on mean metric
  - Cohen's d effect size with interpretation (small/medium/large)

Metrics tested: MRR@k, Recall@k

Output:
  - Console: summary table + pairwise results
  - JSONL: outputs/analysis/stat_tests/stat_results.jsonl
  - LaTeX: outputs/analysis/stat_tests/significance_table.tex

Usage:
  python src/analysis/statistical_test.py \\
    --predictions outputs/predictions/bm25_kuhperdata-exp.jsonl \\
                  outputs/predictions/jnlp_stage1_kuhperdata-exp.jsonl \\
                  outputs/predictions/gar_kuhperdata-exp.jsonl \\
    --names bm25 jnlp gar \\
    --output_dir outputs/analysis/stat_tests
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats


def load_predictions(path: str) -> tuple[dict, dict]:
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


def compute_per_query_precision(rankings: dict, ground_truth: dict, k: int = 10) -> dict[str, float]:
    precision = {}
    for qid in ground_truth:
        gt = set(ground_truth[qid])
        ranked = rankings.get(qid, [])[:k]
        if not ranked:
            precision[qid] = 0.0
            continue
        precision[qid] = len(set(ranked) & gt) / len(ranked)
    return precision


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


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


def run_comparison(name_a, vals_a, name_b, vals_b, qids, metric_name):
    a = np.array([vals_a[q] for q in qids])
    b = np.array([vals_b[q] for q in qids])

    mean_a, mean_b = a.mean(), b.mean()
    ci_a, ci_b = bootstrap_ci(a), bootstrap_ci(b)

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    d = cohens_d(a, b)

    return {
        "metric": metric_name,
        "pair": f"{name_a} vs {name_b}",
        "n_queries": len(qids),
        f"mean_{name_a}": round(mean_a, 4),
        f"mean_{name_b}": round(mean_b, 4),
        f"ci95_{name_a}": [round(ci_a[0], 4), round(ci_a[1], 4)],
        f"ci95_{name_b}": [round(ci_b[0], 4), round(ci_b[1], 4)],
        "delta": round(mean_a - mean_b, 4),
        "cohens_d": round(d, 4),
        "effect_size": effect_size_label(d),
        "ttest_p": round(float(t_p), 6),
        "wilcoxon_p": round(float(w_p), 6),
    }


def run_friedman(method_values: dict[str, dict[str, float]], qids: list[str], metric_name: str):
    names = list(method_values.keys())
    if len(names) < 3:
        return None
    samples = [np.array([method_values[n][q] for q in qids]) for n in names]
    try:
        stat, p = stats.friedmanchisquare(*samples)
        return {"metric": metric_name, "methods": names, "n_queries": len(qids),
                "friedman_stat": round(float(stat), 4), "friedman_p": round(float(p), 6)}
    except Exception:
        return None


def generate_latex_table(all_results: list[dict], method_summaries: dict, output_path: str):
    methods = list(method_summaries.keys())
    metrics = sorted(set(r["metric"] for r in all_results))

    sig_map = {}
    for r in all_results:
        key = (r["metric"], r["pair"])
        sig_map[key] = r.get("holm_p", r["wilcoxon_p"])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Retrieval results with statistical significance.}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(m.replace("@", r"@").replace("_", r"\_") for m in metrics) + r" \\",
        r"\midrule",
    ]

    for m in methods:
        row = [m.replace("_", r"\_")]
        for metric in metrics:
            val = method_summaries[m].get(metric, {}).get("mean", 0)
            ci = method_summaries[m].get(metric, {}).get("ci", (0, 0))
            row.append(f"{val:.4f}")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Statistical tests on retrieval predictions")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="Paths to prediction JSONL files")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Method names (same order as --predictions)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/stat_tests")
    args = parser.parse_args()

    assert len(args.predictions) == len(args.names)

    method_rankings = {}
    all_gt = {}

    for path, name in zip(args.predictions, args.names):
        rankings, gt = load_predictions(path)
        if gt:
            all_gt.update(gt)
        method_rankings[name] = (rankings, gt)

    metric_fns = {
        f"mrr@{args.top_k}": compute_per_query_rr,
        f"recall@{args.top_k}": compute_per_query_recall,
        f"precision@{args.top_k}": compute_per_query_precision,
    }

    method_metrics = {}
    method_summaries = {}
    for name in args.names:
        rankings, gt = method_rankings[name]
        gt_used = gt or all_gt
        method_metrics[name] = {}
        method_summaries[name] = {}
        for metric_name, fn in metric_fns.items():
            values = fn(rankings, gt_used, k=args.top_k)
            method_metrics[name][metric_name] = values
            arr = np.array(list(values.values()))
            ci = bootstrap_ci(arr)
            method_summaries[name][metric_name] = {
                "mean": round(float(arr.mean()), 4),
                "ci": [round(ci[0], 4), round(ci[1], 4)],
            }

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"{'Method':<20}", end="")
    for metric_name in metric_fns:
        print(f"  {metric_name:<20}", end="")
    print()
    print("-" * 90)
    for name in args.names:
        print(f"{name:<20}", end="")
        for metric_name in metric_fns:
            s = method_summaries[name][metric_name]
            print(f"  {s['mean']:.4f} [{s['ci'][0]:.4f},{s['ci'][1]:.4f}]", end="")
        print()

    # Friedman omnibus test (>2 methods)
    shared_qids = sorted(set.intersection(
        *[set(method_metrics[n][list(metric_fns.keys())[0]].keys()) for n in args.names]
    ))
    print(f"\nShared queries: {len(shared_qids)}")

    friedman_results = []
    if len(args.names) >= 3:
        print(f"\n{'=' * 90}")
        print("Friedman omnibus test (are any methods significantly different?)")
        print(f"{'=' * 90}")
        for metric_name in metric_fns:
            vals = {n: method_metrics[n][metric_name] for n in args.names}
            fr = run_friedman(vals, shared_qids, metric_name)
            if fr:
                friedman_results.append(fr)
                sig = "***" if fr["friedman_p"] < 0.001 else ("**" if fr["friedman_p"] < 0.01 else ("*" if fr["friedman_p"] < 0.05 else "ns"))
                print(f"  {metric_name}: χ² = {fr['friedman_stat']:.2f}, p = {fr['friedman_p']:.6f} [{sig}]")

    # Pairwise comparisons with Holm-Bonferroni
    all_results = []
    for metric_name in metric_fns:
        pairwise = []
        for (name_a, _), (name_b, _) in combinations(method_rankings.items(), 2):
            vals_a = method_metrics[name_a][metric_name]
            vals_b = method_metrics[name_b][metric_name]
            pair_qids = sorted(set(vals_a.keys()) & set(vals_b.keys()))
            if not pair_qids:
                continue
            res = run_comparison(name_a, vals_a, name_b, vals_b, pair_qids, metric_name)
            pairwise.append(res)

        raw_ps = [r["wilcoxon_p"] for r in pairwise]
        adjusted_ps = holm_bonferroni(raw_ps) if raw_ps else []
        for r, adj_p in zip(pairwise, adjusted_ps):
            r["holm_p"] = round(adj_p, 6)
            r["significant_005"] = adj_p < 0.05 if not np.isnan(adj_p) else None
            r["significant_001"] = adj_p < 0.01 if not np.isnan(adj_p) else None

        all_results.extend(pairwise)

    # Print pairwise results
    for metric_name in metric_fns:
        metric_results = [r for r in all_results if r["metric"] == metric_name]
        if not metric_results:
            continue
        print(f"\n{'=' * 90}")
        print(f"Pairwise: {metric_name} (Wilcoxon + Holm-Bonferroni correction)")
        print(f"{'=' * 90}")
        for res in metric_results:
            sig = "***" if res.get("significant_001") else ("**" if res.get("significant_005") else "ns")
            print(f"\n  {res['pair']} (n={res['n_queries']})")
            print(f"    Δ = {res['delta']:+.4f}  Cohen's d = {res['cohens_d']:.3f} ({res['effect_size']})")
            print(f"    Wilcoxon p = {res['wilcoxon_p']:.6f}  Holm p = {res['holm_p']:.6f}  [{sig}]")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = f"{args.output_dir}/stat_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for fr in friedman_results:
            f.write(json.dumps({"type": "friedman", **fr}, ensure_ascii=False) + "\n")
        for res in all_results:
            f.write(json.dumps({"type": "pairwise", **res}, ensure_ascii=False) + "\n")
    print(f"\nResults saved: {out_path}")

    summary_path = f"{args.output_dir}/method_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(method_summaries, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    latex_path = f"{args.output_dir}/significance_table.tex"
    generate_latex_table(all_results, method_summaries, latex_path)


if __name__ == "__main__":
    main()
