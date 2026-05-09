"""Generate pairs.tsv from humanized and summarized expansion logs.

Samples cases from each expansion log independently, producing one row per
(query, article) pair. Each row is tagged with its variant (humanized/summarized).

Usage:
  python generate_pairs.py
  python generate_pairs.py --n_cases 60 --seed 123
  python generate_pairs.py --hum_logs ../data/kuhperdata-exp/expansion_logs.jsonl \
                           --sum_logs ../data/kuhperdata-summ-exp/expansion_log.jsonl
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
    results = {}
    for line in raw_response.strip().split("\n"):
        m = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        rest = m.group(2)
        if idx < 0 or idx >= len(candidate_doc_ids):
            continue
        doc_id = candidate_doc_ids[idx]
        if "RELEVAN" in rest and "TIDAK_RELEVAN" not in rest:
            results[doc_id] = "RELEVAN"
        elif "TIDAK_RELEVAN" in rest:
            results[doc_id] = "TIDAK_RELEVAN"
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


def load_original_qrels(dataset_dir: str) -> dict[str, set[str]]:
    qrels = defaultdict(set)
    for split in ["train", "test"]:
        path = os.path.join(dataset_dir, f"qrels_{split}.tsv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qrels[parts[0]].add(parts[1])
    return qrels


def collect_pairs(log_path, queries, corpus, original_qrels, variant):
    pairs_by_verdict = defaultdict(list)
    n_skipped = 0

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["qid"]
            if entry.get("error") or qid not in queries:
                continue

            verdicts = parse_raw_response(entry["raw_response"], entry["candidate_doc_ids"])
            for doc_id, verdict in verdicts.items():
                if doc_id not in corpus:
                    continue
                if doc_id in original_qrels.get(qid, set()):
                    n_skipped += 1
                    continue
                pairs_by_verdict[verdict].append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "verdict": verdict,
                    "variant": variant,
                    "book": article_id_to_book(doc_id),
                })

    print(f"  [{variant}] Skipped {n_skipped} already in original qrels")
    print(f"  [{variant}] RELEVAN: {len(pairs_by_verdict.get('RELEVAN', []))}, "
          f"TIDAK_RELEVAN: {len(pairs_by_verdict.get('TIDAK_RELEVAN', []))}")
    return pairs_by_verdict


def main():
    parser = argparse.ArgumentParser(description="Generate annotation pairs from expansion logs")
    parser.add_argument("--hum_logs", default="../data/kuhperdata-exp/expansion_logs.jsonl")
    parser.add_argument("--sum_logs", default="../data/kuhperdata-summ-exp/expansion_log.jsonl")
    parser.add_argument("--hum_dataset", default="../data/kuhperdata-humanized")
    parser.add_argument("--sum_dataset", default="../data/kuhperdata-summarized")
    parser.add_argument("--corpus_dataset", default="../data/kuhperdata-exp")
    parser.add_argument("--output", default="data/pairs.tsv")
    parser.add_argument("--n_cases", type=int, default=40,
                        help="Total cases to sample (split evenly across variants)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    hum_queries, _ = load_texts(args.hum_dataset)
    sum_queries, _ = load_texts(args.sum_dataset)
    _, corpus = load_texts(args.corpus_dataset)

    hum_orig_qrels = load_original_qrels(args.hum_dataset)
    sum_orig_qrels = load_original_qrels(args.sum_dataset)

    print(f"Original qrels: humanized={sum(len(v) for v in hum_orig_qrels.values())}, "
          f"summarized={sum(len(v) for v in sum_orig_qrels.values())}")

    hum_pairs = collect_pairs(args.hum_logs, hum_queries, corpus, hum_orig_qrels, "humanized")
    sum_pairs = collect_pairs(args.sum_logs, sum_queries, corpus, sum_orig_qrels, "summarized")

    # Sample balanced: half from each variant, half RELEVAN / half TIDAK_RELEVAN
    n_per_variant = args.n_cases // 2
    sampled = []

    for variant_pairs, variant_name in [(hum_pairs, "humanized"), (sum_pairs, "summarized")]:
        n_rel = min(n_per_variant // 2, len(variant_pairs.get("RELEVAN", [])))
        n_not = min(n_per_variant - n_rel, len(variant_pairs.get("TIDAK_RELEVAN", [])))
        rel = random.sample(variant_pairs.get("RELEVAN", []), n_rel)
        not_rel = random.sample(variant_pairs.get("TIDAK_RELEVAN", []), n_not)
        sampled.extend(rel + not_rel)
        print(f"  [{variant_name}] Sampled: {n_rel} RELEVAN + {n_not} TIDAK_RELEVAN")

    random.shuffle(sampled)

    query_dicts = {"humanized": hum_queries, "summarized": sum_queries}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "pair_id", "case_id", "variant", "query_text", "article_id",
            "article_title", "article_text", "llm_label", "kuhperdata_book"
        ])

        for i, pair in enumerate(sampled, 1):
            variant = pair["variant"]
            prefix = "H" if variant == "humanized" else "S"
            pair_id = f"{prefix}{i:03d}"
            case_id = f"case_{i:03d}"
            doc = corpus[pair["doc_id"]]
            writer.writerow([
                pair_id,
                case_id,
                variant,
                query_dicts[variant].get(pair["qid"], ""),
                pair["doc_id"],
                doc["title"],
                doc["text"],
                pair["verdict"],
                pair["book"],
            ])

    print(f"\nGenerated {len(sampled)} pairs")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
