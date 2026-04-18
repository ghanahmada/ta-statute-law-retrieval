"""
StructGNN Error Analysis

Analyzes StructGNN inference results (from infer_paragnn.py output) to understand:
1. What gets retrieved correctly (true positives) and why
2. What gets retrieved incorrectly (false positives) and why
3. What gets missed (false negatives) and why — lexical gap + structural analysis
4. Comparison with BM25-only signal: where does GNN help vs hurt

Input: inference JSON from src/inference/infer_paragnn.py
Output: detailed per-query breakdown + aggregate statistics

Usage:
  python src/analysis_structgnn_errors.py --dataset kuhperdata-humanized
  python src/analysis_structgnn_errors.py --dataset all
  python src/analysis_structgnn_errors.py --dataset bsard --inference_dir outputs/inference
"""

import argparse
import json
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from util.dataloader import DataLoader


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "stard": {"path": "data/stard", "lang": "zh"},
}


def tokenize_simple(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))


def compute_overlap(query_tokens: set, doc_tokens: set) -> dict:
    shared = query_tokens & doc_tokens
    return {
        "shared_tokens": sorted(shared),
        "n_shared": len(shared),
        "query_coverage": len(shared) / len(query_tokens) if query_tokens else 0,
        "doc_coverage": len(shared) / len(doc_tokens) if doc_tokens else 0,
    }


def load_inference(inference_dir: str, dataset: str, method: str = "structgnn_adapted") -> dict:
    path = Path(inference_dir) / dataset / f"{method}.json"
    if not path.exists():
        raise FileNotFoundError(f"No inference file at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_dataset(dataset_name: str, inference_dir: str, top_k: int = 10, max_relevant: int = 5):
    cfg = DATASETS[dataset_name]

    loader = DataLoader(
        f"{cfg['path']}/corpus.jsonl",
        f"{cfg['path']}/queries.jsonl",
        f"{cfg['path']}/qrels_test.tsv",
    ).load()
    if max_relevant:
        loader.filter_max_relevant(max_relevant)

    inference = load_inference(inference_dir, dataset_name)
    inf_config = inference["config"]
    inf_metrics = inference["metrics"]
    inf_results = inference["results"]

    print(f"\n  Model: {inf_config['structure_mode']} (alpha={inf_config['alpha']})")
    print(f"  MRR@10: {inf_metrics['mrr@10']:.4f}  R@10: {inf_metrics['recall@10']:.4f}  Hit: {inf_metrics['hit_rate']:.1%}")
    print(f"  Queries: {inf_metrics['num_queries']}  Candidates: {inf_metrics['num_candidates']}")

    all_results = []
    aggregate = {
        "total_queries": 0,
        "total_relevant": 0,
        "total_retrieved_relevant": 0,
        "total_missed": 0,
        "zero_overlap_misses": 0,
        "gnn_helped": 0,
        "gnn_hurt": 0,
        "gnn_neutral": 0,
        "bm25_rank_of_hits": [],
        "gnn_score_distribution_tp": [],
        "gnn_score_distribution_fp": [],
        "bm25_score_distribution_tp": [],
        "bm25_score_distribution_fp": [],
        "fn_overlap_distribution": [],
        "fp_overlap_distribution": [],
        "missed_docs": [],
    }

    for qid, qresult in inf_results.items():
        if qid not in loader.queries:
            continue

        query_text = loader.queries[qid]["text"]
        query_tokens = tokenize_simple(query_text)
        relevant_docs = set()
        if qid in loader.qrels:
            relevant_docs = {did for did, s in loader.qrels[qid].items() if s > 0}

        ranked = qresult["ranked"]
        true_positives = []
        false_positives = []
        false_negatives = []

        retrieved_ids = set()
        for doc_entry in ranked[:top_k]:
            doc_id = doc_entry["doc_id"]
            retrieved_ids.add(doc_id)

            doc_text = loader.corpus.get(doc_id, {}).get("text", "")
            doc_title = loader.corpus.get(doc_id, {}).get("title", "")
            doc_tokens = tokenize_simple(doc_text)
            overlap = compute_overlap(query_tokens, doc_tokens)

            entry = {
                "rank": doc_entry["rank"],
                "doc_id": doc_id,
                "title": doc_title,
                "score": doc_entry["score"],
                "gnn_score": doc_entry["gnn_score"],
                "bm25_score": doc_entry["bm25_score"],
                "doc_text_preview": doc_text[:300],
                "overlap": overlap,
            }

            if doc_id in relevant_docs:
                true_positives.append(entry)
                aggregate["gnn_score_distribution_tp"].append(doc_entry["gnn_score"])
                aggregate["bm25_score_distribution_tp"].append(doc_entry["bm25_score"])
            else:
                false_positives.append(entry)
                aggregate["gnn_score_distribution_fp"].append(doc_entry["gnn_score"])
                aggregate["bm25_score_distribution_fp"].append(doc_entry["bm25_score"])
                aggregate["fp_overlap_distribution"].append(overlap["n_shared"])

        # Find missed relevant docs
        for doc_id in relevant_docs:
            if doc_id not in retrieved_ids and doc_id in loader.corpus:
                doc_text = loader.corpus[doc_id]["text"]
                doc_title = loader.corpus[doc_id].get("title", "")
                doc_tokens = tokenize_simple(doc_text)
                overlap = compute_overlap(query_tokens, doc_tokens)

                false_negatives.append({
                    "doc_id": doc_id,
                    "title": doc_title,
                    "doc_text_preview": doc_text[:300],
                    "overlap": overlap,
                })
                aggregate["fn_overlap_distribution"].append(overlap["n_shared"])
                if overlap["n_shared"] == 0:
                    aggregate["zero_overlap_misses"] += 1

                aggregate["missed_docs"].append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "title": doc_title,
                    "overlap_tokens": overlap["n_shared"],
                })

        # Classify: did GNN help or hurt vs BM25-only?
        has_hit_blended = len(true_positives) > 0
        bm25_would_hit = any(
            doc_entry["bm25_score"] > 0 and doc_entry["doc_id"] in relevant_docs
            for doc_entry in sorted(ranked, key=lambda x: -x["bm25_score"])[:top_k]
        )

        if has_hit_blended and not bm25_would_hit:
            aggregate["gnn_helped"] += 1
        elif not has_hit_blended and bm25_would_hit:
            aggregate["gnn_hurt"] += 1
        else:
            aggregate["gnn_neutral"] += 1

        result = {
            "qid": qid,
            "query_text": query_text[:500],
            "n_relevant": len(relevant_docs),
            "n_retrieved_relevant": len(true_positives),
            "n_false_positives": len(false_positives),
            "n_missed": len(false_negatives),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        all_results.append(result)

        aggregate["total_queries"] += 1
        aggregate["total_relevant"] += len(relevant_docs)
        aggregate["total_retrieved_relevant"] += len(true_positives)
        aggregate["total_missed"] += len(false_negatives)

    # Compute summary
    total_missed = aggregate["total_missed"]
    summary = {
        "dataset": dataset_name,
        "model": inf_config["structure_mode"],
        "alpha": inf_config["alpha"],
        "top_k": top_k,
        "metrics": inf_metrics,
        "n_queries": aggregate["total_queries"],
        "total_relevant_docs": aggregate["total_relevant"],
        "total_retrieved_relevant": aggregate["total_retrieved_relevant"],
        "total_missed": total_missed,
        "recall": aggregate["total_retrieved_relevant"] / aggregate["total_relevant"] if aggregate["total_relevant"] else 0,
        "zero_overlap_misses": aggregate["zero_overlap_misses"],
        "zero_overlap_miss_rate": aggregate["zero_overlap_misses"] / total_missed if total_missed else 0,
        "gnn_vs_bm25": {
            "gnn_helped": aggregate["gnn_helped"],
            "gnn_hurt": aggregate["gnn_hurt"],
            "gnn_neutral": aggregate["gnn_neutral"],
            "help_rate": aggregate["gnn_helped"] / aggregate["total_queries"] if aggregate["total_queries"] else 0,
        },
        "score_analysis": {
            "tp_gnn_mean": float(np.mean(aggregate["gnn_score_distribution_tp"])) if aggregate["gnn_score_distribution_tp"] else 0,
            "tp_bm25_mean": float(np.mean(aggregate["bm25_score_distribution_tp"])) if aggregate["bm25_score_distribution_tp"] else 0,
            "fp_gnn_mean": float(np.mean(aggregate["gnn_score_distribution_fp"])) if aggregate["gnn_score_distribution_fp"] else 0,
            "fp_bm25_mean": float(np.mean(aggregate["bm25_score_distribution_fp"])) if aggregate["bm25_score_distribution_fp"] else 0,
        },
        "overlap_analysis": {
            "fp_avg_overlap": float(np.mean(aggregate["fp_overlap_distribution"])) if aggregate["fp_overlap_distribution"] else 0,
            "fn_avg_overlap": float(np.mean(aggregate["fn_overlap_distribution"])) if aggregate["fn_overlap_distribution"] else 0,
        },
    }

    return summary, all_results


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  AGGREGATE SUMMARY: {summary['dataset']}")
    print(f"{'='*60}")
    print(f"  Model: {summary['model']} (alpha={summary['alpha']})")
    print(f"  MRR@10: {summary['metrics']['mrr@10']:.4f}  R@10: {summary['metrics']['recall@10']:.4f}  Hit: {summary['metrics']['hit_rate']:.1%}")
    print(f"\n  Queries:             {summary['n_queries']}")
    print(f"  Recall@10:           {summary['recall']:.4f}")
    print(f"  Total relevant:      {summary['total_relevant_docs']}")
    print(f"  Retrieved relevant:  {summary['total_retrieved_relevant']}")
    print(f"  Missed:              {summary['total_missed']}")
    print(f"  Zero-overlap misses: {summary['zero_overlap_misses']} "
          f"({summary['zero_overlap_miss_rate']:.1%} of all misses)")

    gnn = summary["gnn_vs_bm25"]
    print(f"\n  GNN vs BM25-only:")
    print(f"    GNN helped (hit where BM25 missed): {gnn['gnn_helped']} ({gnn['help_rate']:.1%})")
    print(f"    GNN hurt (BM25 hit but blend missed): {gnn['gnn_hurt']}")
    print(f"    Neutral (both hit or both miss): {gnn['gnn_neutral']}")

    sa = summary["score_analysis"]
    print(f"\n  Score analysis (mean):")
    print(f"    True positives:  GNN={sa['tp_gnn_mean']:.3f}  BM25={sa['tp_bm25_mean']:.3f}")
    print(f"    False positives: GNN={sa['fp_gnn_mean']:.3f}  BM25={sa['fp_bm25_mean']:.3f}")

    oa = summary["overlap_analysis"]
    print(f"\n  Lexical overlap (avg shared tokens):")
    print(f"    False positives: {oa['fp_avg_overlap']:.1f}")
    print(f"    False negatives: {oa['fn_avg_overlap']:.1f}")


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def print_examples(all_results, n=10):
    worst = sorted(all_results, key=lambda x: -x["n_missed"])[:n]

    safe_print(f"\n{'='*60}")
    safe_print(f"  WORST QUERIES (top {n} with most misses)")
    safe_print(f"{'='*60}")

    for r in worst:
        safe_print(f"\n--- Query {r['qid']}: {r['query_text'][:200]}")
        safe_print(f"    Relevant: {r['n_relevant']}, Retrieved: {r['n_retrieved_relevant']}, Missed: {r['n_missed']}")

        if r["true_positives"]:
            safe_print(f"  TRUE POSITIVES:")
            for tp in r["true_positives"][:3]:
                safe_print(f"    Rank {tp['rank']}: {tp['title']} "
                           f"(score={tp['score']:.3f}, gnn={tp['gnn_score']:.3f}, bm25={tp['bm25_score']:.3f}, "
                           f"shared={tp['overlap']['n_shared']})")

        if r["false_negatives"]:
            safe_print(f"  FALSE NEGATIVES:")
            for fn in r["false_negatives"]:
                safe_print(f"    {fn['title']} — shared={fn['overlap']['n_shared']} tokens")
                if fn['overlap']['shared_tokens']:
                    safe_print(f"      Shared: {', '.join(fn['overlap']['shared_tokens'][:10])}")

        if r["false_positives"]:
            safe_print(f"  FALSE POSITIVES (top 3):")
            for fp in r["false_positives"][:3]:
                safe_print(f"    Rank {fp['rank']}: {fp['title']} "
                           f"(score={fp['score']:.3f}, gnn={fp['gnn_score']:.3f}, bm25={fp['bm25_score']:.3f}, "
                           f"shared={fp['overlap']['n_shared']})")


def main():
    parser = argparse.ArgumentParser(description="StructGNN Error Analysis")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--inference_dir", default="outputs/inference")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--n_examples", type=int, default=10)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  StructGNN Error Analysis: {ds}")
        print(f"{'='*60}")

        try:
            summary, all_results = analyze_dataset(ds, args.inference_dir, args.top_k, args.max_relevant)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        print_summary(summary)
        print_examples(all_results, args.n_examples)

        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"structgnn_errors_{ds}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "queries": all_results}, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
