"""
Generate structured analysis for law expert review.

Purpose: Identify articles that BM25 retrieves in top-k but are NOT in ground truth,
and determine whether they are:
  (a) Actually relevant → ground truth should be expanded
  (b) Truly irrelevant → informs what retrieval needs to improve

Prioritization signals for "likely relevant but unlabeled":
  1. Same KUHPerdata Buku/Bab as a GT article (structural proximity)
  2. High BM25 score (strong lexical match)
  3. Co-occurs as GT for related queries (legal hub article)
  4. High token overlap with query (topical match)

Output:
  - JSON with per-query analysis for expert review
  - Aggregate statistics on candidate categories
  - Sampled review sheets (printable for the meeting)
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


def parse_pasal_number(doc_id: str) -> int:
    """Extract numeric pasal number for proximity calculation."""
    match = re.match(r'(\d+)', doc_id)
    return int(match.group(1)) if match else 0


def get_buku_bab(pasal_num: int) -> dict:
    """Map KUHPerdata pasal number to Buku and approximate Bab.
    KUHPerdata structure:
      Buku I   (Orang):           Pasal 1-498
      Buku II  (Benda):           Pasal 499-1232
      Buku III (Perikatan):       Pasal 1233-1864
      Buku IV  (Pembuktian):      Pasal 1865-1993
    """
    if pasal_num <= 498:
        buku = "I (Orang)"
    elif pasal_num <= 1232:
        buku = "II (Benda)"
    elif pasal_num <= 1864:
        buku = "III (Perikatan)"
    else:
        buku = "IV (Pembuktian)"
    return {"buku": buku, "pasal_num": pasal_num}


def compute_proximity_to_gt(doc_id: str, gt_doc_ids: set) -> dict:
    """Compute structural proximity of a doc to ground truth articles."""
    doc_num = parse_pasal_number(doc_id)
    doc_info = get_buku_bab(doc_num)

    gt_nums = [parse_pasal_number(d) for d in gt_doc_ids]
    gt_infos = [get_buku_bab(n) for n in gt_nums]

    if not gt_nums:
        return {"min_distance": None, "same_buku": False, "nearest_gt": None}

    distances = [abs(doc_num - gn) for gn in gt_nums]
    min_dist = min(distances)
    nearest_idx = distances.index(min_dist)
    nearest_gt = list(gt_doc_ids)[nearest_idx]
    same_buku = doc_info["buku"] == gt_infos[nearest_idx]["buku"]

    return {
        "min_distance": min_dist,
        "same_buku": same_buku,
        "nearest_gt": nearest_gt,
        "doc_buku": doc_info["buku"],
    }


def classify_fp(fp_entry: dict, gt_doc_ids: set, global_gt_docs: set,
                doc_gt_frequency: Counter) -> dict:
    """Classify a false positive by likelihood of being truly relevant."""
    doc_id = fp_entry["doc_id"]
    prox = compute_proximity_to_gt(doc_id, gt_doc_ids)

    # Signals
    is_gt_elsewhere = doc_id in global_gt_docs
    gt_freq = doc_gt_frequency.get(doc_id, 0)
    close_to_gt = prox["min_distance"] is not None and prox["min_distance"] <= 50
    same_buku = prox["same_buku"]
    high_overlap = fp_entry["overlap"]["n_shared"] >= 3

    # Priority score: higher = more likely to be truly relevant
    score = 0
    reasons = []
    if close_to_gt and prox["min_distance"] <= 20:
        score += 3
        reasons.append(f"within 20 articles of GT ({prox['nearest_gt']})")
    elif close_to_gt:
        score += 2
        reasons.append(f"within 50 articles of GT ({prox['nearest_gt']})")
    if same_buku:
        score += 1
        reasons.append(f"same Buku {prox['doc_buku']}")
    if is_gt_elsewhere:
        score += 2
        reasons.append(f"GT for {gt_freq} other queries")
    if high_overlap:
        score += 1
        reasons.append(f"{fp_entry['overlap']['n_shared']} shared tokens")

    category = "high_priority" if score >= 4 else "medium_priority" if score >= 2 else "low_priority"

    return {
        **fp_entry,
        "proximity": prox,
        "is_gt_elsewhere": is_gt_elsewhere,
        "gt_frequency": gt_freq,
        "priority_score": score,
        "priority_reasons": reasons,
        "category": category,
    }


def run_analysis(dataset_name: str, top_k: int = 25, max_relevant: int = 5):
    cfg = DATASETS[dataset_name]
    data_dir = cfg["path"]
    lang = cfg["lang"]

    # Load test data
    loader = DataLoader(
        f"{data_dir}/corpus.jsonl",
        f"{data_dir}/queries.jsonl",
        f"{data_dir}/qrels_test.tsv",
    ).load()
    if max_relevant:
        loader.filter_max_relevant(max_relevant)

    # Load ALL qrels (train + test) to know global GT
    global_gt_docs = set()
    doc_gt_frequency = Counter()
    for split in ["train", "test"]:
        with open(f"{data_dir}/qrels_{split}.tsv", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    global_gt_docs.add(parts[1])
                    doc_gt_frequency[parts[1]] += 1

    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()

    print(f"Corpus: {len(doc_ids)} docs")
    print(f"Test queries: {len(loader.qrels)} (max_relevant={max_relevant})")
    print(f"Global GT docs: {len(global_gt_docs)}")

    # Fit BM25
    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang=lang,
                use_stemmer=False, use_stopwords=False)
    bm25.fit(doc_texts)

    eval_qids = set(loader.qrels.keys())
    all_results = []
    agg = {"high_priority": 0, "medium_priority": 0, "low_priority": 0,
           "total_fp": 0, "total_tp": 0, "total_queries": 0}

    for qid, query in zip(query_ids, query_texts):
        if qid not in eval_qids:
            continue

        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        bm25_ranked = [(doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

        gt_doc_ids = set(loader.qrels[qid].keys())
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

        true_positives = []
        false_positives_classified = []

        for rank, (doc_id, score) in enumerate(bm25_ranked):
            doc_text = loader.corpus.get(doc_id, {}).get("text", "")
            doc_title = loader.corpus.get(doc_id, {}).get("title", "")
            doc_tokens = set(re.findall(r'\b\w+\b', doc_text.lower()))
            shared = query_tokens & doc_tokens

            entry = {
                "rank": rank + 1,
                "doc_id": doc_id,
                "title": doc_title,
                "bm25_score": round(score, 4),
                "doc_text": doc_text,
                "overlap": {
                    "shared_tokens": sorted(shared),
                    "n_shared": len(shared),
                },
            }

            if doc_id in gt_doc_ids:
                true_positives.append(entry)
                agg["total_tp"] += 1
            else:
                classified = classify_fp(entry, gt_doc_ids, global_gt_docs,
                                         doc_gt_frequency)
                false_positives_classified.append(classified)
                agg[classified["category"]] += 1
                agg["total_fp"] += 1

        # Also show where GT docs actually rank (for context)
        gt_ranks = {}
        full_ranked = [(doc_ids[idx], float(scores[idx])) for idx in np.argsort(scores)[::-1]]
        for gt_did in gt_doc_ids:
            for rank, (did, sc) in enumerate(full_ranked):
                if did == gt_did:
                    gt_ranks[gt_did] = {"rank": rank + 1, "bm25_score": round(sc, 4)}
                    break

        all_results.append({
            "qid": qid,
            "query_text": query,
            "case_name": loader.queries[qid].get("metadata", {}).get("case_name", ""),
            "n_gt": len(gt_doc_ids),
            "gt_doc_ids": sorted(gt_doc_ids),
            "gt_ranks": gt_ranks,
            "n_tp_in_topk": len(true_positives),
            "true_positives": true_positives,
            "false_positives": sorted(false_positives_classified,
                                      key=lambda x: -x["priority_score"]),
        })
        agg["total_queries"] += 1

    return agg, all_results


def generate_review_sheet(results: list, n_queries: int = 30, output_path: str = None):
    """Generate a human-readable review sheet for the law expert.

    Samples queries that have the most high-priority FPs — these are the ones
    most worth discussing with the expert.
    """
    # Sort by number of high-priority FPs
    ranked = sorted(results,
                    key=lambda r: -sum(1 for fp in r["false_positives"]
                                       if fp["category"] == "high_priority"))

    lines = []
    lines.append("=" * 80)
    lines.append("REVIEW SHEET FOR LAW EXPERT")
    lines.append("Dataset: KUHPerdata v2 (Humanized)")
    lines.append("Method: BM25 top-25 retrieval")
    lines.append("Task: Verify whether retrieved articles are legally relevant to the query")
    lines.append("=" * 80)
    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("For each query below, we show articles that BM25 retrieved but are NOT")
    lines.append("in our current ground truth (based on court citation). Please mark each as:")
    lines.append("  [R] Relevant  — this article IS applicable to the legal question")
    lines.append("  [P] Partially — this article is related but not directly applicable")
    lines.append("  [I] Irrelevant — this article is NOT applicable")
    lines.append("  [U] Unsure    — needs deeper analysis")
    lines.append("")
    lines.append("For [R] and [P] articles, briefly note WHY it's relevant.")
    lines.append("=" * 80)

    for i, r in enumerate(ranked[:n_queries]):
        lines.append("")
        lines.append(f"{'─' * 80}")
        lines.append(f"QUERY {i+1} ({r['qid']})")
        lines.append(f"{'─' * 80}")
        lines.append(f"Text: {r['query_text']}")
        lines.append(f"Case: {r['case_name']}")
        lines.append(f"Current GT: {', '.join(f'Pasal {d}' for d in r['gt_doc_ids'])}")

        # Show where GT docs ranked
        lines.append(f"GT ranks in BM25:")
        for did, info in sorted(r["gt_ranks"].items(), key=lambda x: x[1]["rank"]):
            lines.append(f"  Pasal {did}: rank {info['rank']} (score={info['bm25_score']})")

        # Show high and medium priority FPs for review
        review_fps = [fp for fp in r["false_positives"]
                      if fp["category"] in ("high_priority", "medium_priority")]

        if not review_fps:
            lines.append("  (No high/medium priority candidates)")
            continue

        lines.append(f"\nCANDIDATES FOR REVIEW ({len(review_fps)} articles):")
        lines.append("")

        for j, fp in enumerate(review_fps):
            lines.append(f"  [{j+1}] Pasal {fp['doc_id']} (BM25 rank {fp['rank']}, "
                         f"score={fp['bm25_score']})")
            lines.append(f"      Priority: {fp['category']} — "
                         f"{'; '.join(fp['priority_reasons'])}")
            lines.append(f"      Shared tokens: {', '.join(fp['overlap']['shared_tokens'][:15])}")
            # Show full article text (truncated to reasonable length for review)
            text = fp["doc_text"][:500]
            lines.append(f"      Article text: {text}")
            lines.append(f"      Judgment: [ R / P / I / U ]  Reason: ________________")
            lines.append("")

    text = "\n".join(lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Review sheet saved to {output_path}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate expert review analysis")
    parser.add_argument("--dataset", default="kuhperdata-humanized",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--top_k", type=int, default=25,
                        help="BM25 top-k to analyze (25 = look deeper than top-10)")
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--n_sample", type=int, default=30,
                        help="Number of queries in the review sheet")
    parser.add_argument("--output_dir", default="outputs/analysis")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Expert Review Analysis: {args.dataset}")
    print(f"  BM25 top-{args.top_k}, max_relevant={args.max_relevant}")
    print(f"{'=' * 60}\n")

    agg, results = run_analysis(args.dataset, args.top_k, args.max_relevant)

    # Print aggregate
    print(f"\n{'=' * 60}")
    print("AGGREGATE")
    print(f"{'=' * 60}")
    print(f"  Queries analyzed: {agg['total_queries']}")
    print(f"  True positives in top-{args.top_k}: {agg['total_tp']}")
    print(f"  False positives in top-{args.top_k}: {agg['total_fp']}")
    print(f"    High priority (likely relevant):   {agg['high_priority']} "
          f"({agg['high_priority']/agg['total_fp']:.1%})")
    print(f"    Medium priority:                   {agg['medium_priority']} "
          f"({agg['medium_priority']/agg['total_fp']:.1%})")
    print(f"    Low priority (likely irrelevant):   {agg['low_priority']} "
          f"({agg['low_priority']/agg['total_fp']:.1%})")

    # Save full JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"expert_review_{args.dataset}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "queries": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {json_path}")

    # Generate review sheet
    sheet_path = output_dir / f"review_sheet_{args.dataset}.txt"
    generate_review_sheet(results, n_queries=args.n_sample,
                          output_path=str(sheet_path))


if __name__ == "__main__":
    main()
