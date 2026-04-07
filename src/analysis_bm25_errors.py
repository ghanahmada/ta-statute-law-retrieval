"""
BM25 Error Analysis on KUHPerdata v2

Analyzes BM25 top-k results to understand:
1. What gets retrieved correctly (true positives) and why
2. What gets retrieved incorrectly (false positives) and why
3. What gets missed (false negatives) and why — lexical gap analysis

Output: detailed per-query breakdown + aggregate statistics as JSON
"""

import argparse
import json
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from util.bm25 import BM25
from util.dataloader import DataLoader


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
}


def tokenize_simple(text: str) -> set:
    """Simple whitespace + lowercased tokenization for overlap analysis."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def compute_overlap(query_tokens: set, doc_tokens: set) -> dict:
    """Compute lexical overlap metrics between query and document."""
    shared = query_tokens & doc_tokens
    return {
        "shared_tokens": sorted(shared),
        "n_shared": len(shared),
        "query_coverage": len(shared) / len(query_tokens) if query_tokens else 0,
        "doc_coverage": len(shared) / len(doc_tokens) if doc_tokens else 0,
    }


def analyze_query(
    qid: str,
    query_text: str,
    bm25_ranked: list,  # [(doc_id, score), ...]
    relevant_docs: set,
    corpus: dict,
    top_k: int = 10,
) -> dict:
    """Analyze a single query's BM25 results."""
    query_tokens = tokenize_simple(query_text)
    top_k_ids = [did for did, _ in bm25_ranked[:top_k]]
    top_k_set = set(top_k_ids)

    # Classify results
    true_positives = []  # relevant AND in top-k
    false_positives = []  # in top-k but NOT relevant
    false_negatives = []  # relevant but NOT in top-k

    # Analyze top-k retrieved docs
    for rank, (doc_id, score) in enumerate(bm25_ranked[:top_k]):
        doc_text = corpus[doc_id]["text"] if doc_id in corpus else ""
        doc_title = corpus[doc_id].get("title", "") if doc_id in corpus else ""
        doc_tokens = tokenize_simple(doc_text)
        overlap = compute_overlap(query_tokens, doc_tokens)

        entry = {
            "rank": rank + 1,
            "doc_id": doc_id,
            "title": doc_title,
            "bm25_score": round(score, 4),
            "doc_text_preview": doc_text[:300],
            "overlap": overlap,
        }

        if doc_id in relevant_docs:
            true_positives.append(entry)
        else:
            false_positives.append(entry)

    # Analyze missed relevant docs
    for doc_id in relevant_docs:
        if doc_id not in top_k_set and doc_id in corpus:
            doc_text = corpus[doc_id]["text"]
            doc_title = corpus[doc_id].get("title", "")
            doc_tokens = tokenize_simple(doc_text)
            overlap = compute_overlap(query_tokens, doc_tokens)

            # Find where it actually ranked
            actual_rank = None
            actual_score = None
            for rank, (did, score) in enumerate(bm25_ranked):
                if did == doc_id:
                    actual_rank = rank + 1
                    actual_score = round(score, 4)
                    break

            false_negatives.append({
                "doc_id": doc_id,
                "title": doc_title,
                "actual_rank": actual_rank,
                "bm25_score": actual_score,
                "doc_text_preview": doc_text[:300],
                "overlap": overlap,
            })

    return {
        "qid": qid,
        "query_text": query_text,
        "n_relevant": len(relevant_docs),
        "n_retrieved_relevant": len(true_positives),
        "n_false_positives": len(false_positives),
        "n_missed": len(false_negatives),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def run_analysis(dataset_name: str, top_k: int = 10, max_relevant: int = 5):
    cfg = DATASETS[dataset_name]
    data_dir = cfg["path"]
    lang = cfg["lang"]

    loader = DataLoader(
        f"{data_dir}/corpus.jsonl",
        f"{data_dir}/queries.jsonl",
        f"{data_dir}/qrels_test.tsv",
    ).load()

    if max_relevant:
        loader.filter_max_relevant(max_relevant)

    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()

    print(f"Corpus: {len(doc_ids)} docs, Test queries: {len(loader.qrels)}")

    # Fit BM25
    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang=lang,
                use_stemmer=False, use_stopwords=False)
    bm25.fit(doc_texts)

    # Get full rankings (not just top-k, so we can find where misses rank)
    eval_qids = set(loader.qrels.keys())
    all_results = []
    aggregate = {
        "total_queries": 0,
        "total_relevant": 0,
        "total_retrieved_relevant": 0,
        "total_false_positives": 0,
        "total_missed": 0,
        "missed_rank_distribution": [],
        "fp_overlap_distribution": [],
        "fn_overlap_distribution": [],
        "zero_overlap_misses": 0,
    }

    for qid, query in zip(query_ids, query_texts):
        if qid not in eval_qids:
            continue

        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1]
        bm25_ranked = [(doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

        relevant_docs = set(loader.qrels[qid].keys())

        result = analyze_query(qid, query, bm25_ranked, relevant_docs,
                               loader.corpus, top_k=top_k)
        all_results.append(result)

        # Aggregate stats
        aggregate["total_queries"] += 1
        aggregate["total_relevant"] += result["n_relevant"]
        aggregate["total_retrieved_relevant"] += result["n_retrieved_relevant"]
        aggregate["total_false_positives"] += result["n_false_positives"]
        aggregate["total_missed"] += result["n_missed"]

        for fn in result["false_negatives"]:
            if fn["actual_rank"] is not None:
                aggregate["missed_rank_distribution"].append(fn["actual_rank"])
            aggregate["fn_overlap_distribution"].append(fn["overlap"]["n_shared"])
            if fn["overlap"]["n_shared"] == 0:
                aggregate["zero_overlap_misses"] += 1

        for fp in result["false_positives"]:
            aggregate["fp_overlap_distribution"].append(fp["overlap"]["n_shared"])

    # Compute aggregate summary
    total_rel = aggregate["total_relevant"]
    total_ret_rel = aggregate["total_retrieved_relevant"]
    total_missed = aggregate["total_missed"]
    missed_ranks = aggregate["missed_rank_distribution"]

    summary = {
        "dataset": dataset_name,
        "top_k": top_k,
        "max_relevant": max_relevant,
        "n_queries": aggregate["total_queries"],
        "recall": total_ret_rel / total_rel if total_rel else 0,
        "total_relevant_docs": total_rel,
        "total_retrieved_relevant": total_ret_rel,
        "total_missed": total_missed,
        "zero_overlap_misses": aggregate["zero_overlap_misses"],
        "zero_overlap_miss_rate": aggregate["zero_overlap_misses"] / total_missed if total_missed else 0,
        "missed_rank_stats": {
            "median": float(np.median(missed_ranks)) if missed_ranks else None,
            "mean": float(np.mean(missed_ranks)) if missed_ranks else None,
            "p25": float(np.percentile(missed_ranks, 25)) if missed_ranks else None,
            "p75": float(np.percentile(missed_ranks, 75)) if missed_ranks else None,
            "within_top50": sum(1 for r in missed_ranks if r <= 50),
            "within_top100": sum(1 for r in missed_ranks if r <= 100),
            "beyond_top100": sum(1 for r in missed_ranks if r > 100),
        },
        "fp_avg_overlap_tokens": float(np.mean(aggregate["fp_overlap_distribution"])) if aggregate["fp_overlap_distribution"] else 0,
        "fn_avg_overlap_tokens": float(np.mean(aggregate["fn_overlap_distribution"])) if aggregate["fn_overlap_distribution"] else 0,
    }

    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description="BM25 Error Analysis")
    parser.add_argument("--dataset", type=str, default="kuhperdata-humanized",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    parser.add_argument("--n_examples", type=int, default=20,
                        help="Number of detailed error examples to print")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  BM25 Error Analysis: {args.dataset}")
    print(f"  top_k={args.top_k}, max_relevant={args.max_relevant}")
    print(f"{'=' * 60}\n")

    summary, all_results = run_analysis(args.dataset, args.top_k, args.max_relevant)

    # Print aggregate summary
    print(f"\n{'=' * 60}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Queries:               {summary['n_queries']}")
    print(f"  Recall@{args.top_k}:            {summary['recall']:.4f}")
    print(f"  Total relevant docs:   {summary['total_relevant_docs']}")
    print(f"  Retrieved relevant:    {summary['total_retrieved_relevant']}")
    print(f"  Missed:                {summary['total_missed']}")
    print(f"  Zero-overlap misses:   {summary['zero_overlap_misses']} "
          f"({summary['zero_overlap_miss_rate']:.1%} of all misses)")
    print(f"\n  Missed doc rank distribution (where do misses actually rank?):")
    rs = summary["missed_rank_stats"]
    if rs["median"] is not None:
        print(f"    Median rank: {rs['median']:.0f}")
        print(f"    Mean rank:   {rs['mean']:.0f}")
        print(f"    25th-75th:   {rs['p25']:.0f} - {rs['p75']:.0f}")
        print(f"    Within top-50:  {rs['within_top50']}")
        print(f"    Within top-100: {rs['within_top100']}")
        print(f"    Beyond top-100: {rs['beyond_top100']}")
    print(f"\n  Avg lexical overlap (shared tokens):")
    print(f"    False positives: {summary['fp_avg_overlap_tokens']:.1f} tokens")
    print(f"    False negatives: {summary['fn_avg_overlap_tokens']:.1f} tokens")

    # Print detailed examples: worst misses (most relevant docs missed)
    worst = sorted(all_results, key=lambda x: -x["n_missed"])[:args.n_examples]

    print(f"\n{'=' * 60}")
    print(f"DETAILED EXAMPLES (top {args.n_examples} queries with most misses)")
    print(f"{'=' * 60}")

    for r in worst:
        print(f"\n--- Query {r['qid']}: {r['query_text'][:200]}")
        print(f"    Relevant: {r['n_relevant']}, Retrieved: {r['n_retrieved_relevant']}, "
              f"Missed: {r['n_missed']}")

        if r["true_positives"]:
            print(f"  TRUE POSITIVES:")
            for tp in r["true_positives"][:3]:
                print(f"    Rank {tp['rank']}: {tp['title']} (score={tp['bm25_score']}, "
                      f"shared={tp['overlap']['n_shared']} tokens)")
                print(f"      Shared: {', '.join(tp['overlap']['shared_tokens'][:10])}")

        if r["false_negatives"]:
            print(f"  FALSE NEGATIVES (missed relevant docs):")
            for fn in r["false_negatives"]:
                print(f"    {fn['title']} — actual rank={fn['actual_rank']}, "
                      f"score={fn['bm25_score']}, shared={fn['overlap']['n_shared']} tokens")
                print(f"      Shared: {', '.join(fn['overlap']['shared_tokens'][:10])}")
                print(f"      Doc: {fn['doc_text_preview'][:150]}...")

        if r["false_positives"]:
            print(f"  FALSE POSITIVES (top 3 irrelevant retrieved):")
            for fp in r["false_positives"][:3]:
                print(f"    Rank {fp['rank']}: {fp['title']} (score={fp['bm25_score']}, "
                      f"shared={fp['overlap']['n_shared']} tokens)")
                print(f"      Shared: {', '.join(fp['overlap']['shared_tokens'][:10])}")

    # Save full results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bm25_errors_{args.dataset}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": all_results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
