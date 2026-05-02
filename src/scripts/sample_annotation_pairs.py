"""Sample query-article pairs from expansion logs for human annotation study.

Produces a CSV with 100 pairs (50 LLM-RELEVANT, 50 LLM-NOT_RELEVANT),
stratified across KUHPerdata books. Each row includes query text, article text,
LLM verdict, and LLM reasoning for the legal expert to review.
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict


BOOK_RANGES = [
    ("Buku I (Orang)", 1, 498),
    ("Buku II (Benda)", 499, 1232),
    ("Buku III (Perikatan)", 1233, 1864),
    ("Buku IV (Pembuktian)", 1865, 1993),
]


def article_id_to_book(doc_id: str) -> str:
    try:
        num = int(re.match(r"(\d+)", doc_id).group(1))
    except (AttributeError, ValueError):
        return "Unknown"
    for name, lo, hi in BOOK_RANGES:
        if lo <= num <= hi:
            return name
    return "Unknown"


def parse_raw_response(raw_response: str, candidate_doc_ids: list[str]):
    """Parse LLM raw_response into per-article verdicts.

    Returns dict: doc_id -> {verdict: RELEVAN|TIDAK_RELEVAN, reasoning: str}
    """
    results = {}
    lines = raw_response.strip().split("\n")
    for line in lines:
        m = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        rest = m.group(2)
        if idx < 0 or idx >= len(candidate_doc_ids):
            continue
        doc_id = candidate_doc_ids[idx]
        if "RELEVAN" in rest and "TIDAK_RELEVAN" not in rest:
            verdict = "RELEVAN"
        elif "TIDAK_RELEVAN" in rest:
            verdict = "TIDAK_RELEVAN"
        else:
            continue
        results[doc_id] = {"verdict": verdict, "reasoning": rest}
    return results


def load_texts(dataset_dir: str):
    queries = {}
    with open(os.path.join(dataset_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]

    corpus = {}
    with open(os.path.join(dataset_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d["title"], "text": d["text"]}

    return queries, corpus


def load_original_qrels(dataset_dir: str, base_dir: str | None):
    """Load original (unexpanded) qrels to distinguish new vs original judgments."""
    qrels = defaultdict(set)
    search_dir = base_dir or dataset_dir
    for split in ["train", "test"]:
        path = os.path.join(search_dir, f"qrels_{split}.tsv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qrels[parts[0]].add(parts[1])
    return qrels


def main():
    parser = argparse.ArgumentParser(description="Sample annotation pairs from expansion logs")
    parser.add_argument("--expansion_logs", default="data/kuhperdata-exp/expansion_logs.jsonl")
    parser.add_argument("--dataset_dir", default="data/kuhperdata-exp")
    parser.add_argument("--base_dir", default="data/kuhperdata-humanized",
                        help="Original (unexpanded) dataset dir for distinguishing new judgments")
    parser.add_argument("--output", default="data/annotation_study/annotation_pairs.csv")
    parser.add_argument("--n_relevant", type=int, default=50)
    parser.add_argument("--n_not_relevant", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    queries, corpus = load_texts(args.dataset_dir)
    original_qrels = load_original_qrels(args.dataset_dir, args.base_dir)

    relevant_pairs = []
    not_relevant_pairs = []

    with open(args.expansion_logs, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["qid"]
            if entry.get("error"):
                continue

            candidate_doc_ids = entry["candidate_doc_ids"]
            verdicts = parse_raw_response(entry["raw_response"], candidate_doc_ids)

            for doc_id, info in verdicts.items():
                if doc_id in original_qrels.get(qid, set()):
                    continue

                book = article_id_to_book(doc_id)
                pair = {
                    "qid": qid,
                    "doc_id": doc_id,
                    "book": book,
                    "llm_verdict": info["verdict"],
                    "llm_reasoning": info["reasoning"],
                }

                if info["verdict"] == "RELEVAN":
                    relevant_pairs.append(pair)
                else:
                    not_relevant_pairs.append(pair)

    print(f"Total NEW relevant pairs: {len(relevant_pairs)}")
    print(f"Total NEW not-relevant pairs: {len(not_relevant_pairs)}")

    book_counts_rel = defaultdict(int)
    book_counts_notrel = defaultdict(int)
    for p in relevant_pairs:
        book_counts_rel[p["book"]] += 1
    for p in not_relevant_pairs:
        book_counts_notrel[p["book"]] += 1
    print("\nBook distribution (RELEVANT):", dict(book_counts_rel))
    print("Book distribution (NOT RELEVANT):", dict(book_counts_notrel))

    def stratified_sample(pairs, n):
        by_book = defaultdict(list)
        for p in pairs:
            by_book[p["book"]].append(p)

        total = len(pairs)
        sampled = []
        remaining = n

        books_sorted = sorted(by_book.keys())
        for i, book in enumerate(books_sorted):
            if i == len(books_sorted) - 1:
                quota = remaining
            else:
                quota = max(1, round(n * len(by_book[book]) / total))
                quota = min(quota, remaining)
            sample = random.sample(by_book[book], min(quota, len(by_book[book])))
            sampled.extend(sample)
            remaining -= len(sample)

        if remaining > 0:
            pool = [p for p in pairs if p not in sampled]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))

        return sampled[:n]

    sampled_rel = stratified_sample(relevant_pairs, args.n_relevant)
    sampled_notrel = stratified_sample(not_relevant_pairs, args.n_not_relevant)

    all_sampled = sampled_rel + sampled_notrel
    random.shuffle(all_sampled)

    print(f"\nSampled: {len(sampled_rel)} RELEVANT + {len(sampled_notrel)} NOT RELEVANT")
    book_final = defaultdict(lambda: {"R": 0, "NR": 0})
    for p in all_sampled:
        key = "R" if p["llm_verdict"] == "RELEVAN" else "NR"
        book_final[p["book"]][key] += 1
    print("Final distribution by book:")
    for book in sorted(book_final):
        print(f"  {book}: {book_final[book]['R']}R / {book_final[book]['NR']}NR")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id",
            "query_id",
            "query_text",
            "article_id",
            "article_title",
            "article_text",
            "kuhperdata_book",
            "llm_verdict",
            "llm_reasoning",
            "expert_verdict",
            "expert_notes",
        ])

        for i, pair in enumerate(all_sampled, 1):
            qtext = queries.get(pair["qid"], "")
            doc = corpus.get(pair["doc_id"], {"title": "", "text": ""})
            writer.writerow([
                i,
                pair["qid"],
                qtext,
                pair["doc_id"],
                doc["title"],
                doc["text"],
                pair["book"],
                pair["llm_verdict"],
                pair["llm_reasoning"],
                "",  # expert fills this
                "",  # expert fills this
            ])

    print(f"\nWrote {len(all_sampled)} pairs to {args.output}")
    print("Columns for expert to fill: 'expert_verdict' (RELEVAN/TIDAK_RELEVAN), 'expert_notes'")


if __name__ == "__main__":
    main()
