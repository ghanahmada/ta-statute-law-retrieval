"""Generate pairs.tsv by sampling 40 cases with humanized + summarized variants."""

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
    """Parse LLM raw_response into per-article verdicts."""
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
        results[doc_id] = verdict
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
    """Load original (unexpanded) qrels to identify pre-existing judgments."""
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


def main():
    parser = argparse.ArgumentParser(description="Generate annotation pairs from expansion logs")
    parser.add_argument("--expansion_logs", default="../data/kuhperdata-exp/expansion_logs.jsonl")
    parser.add_argument("--hum_dataset", default="../data/kuhperdata-humanized",
                        help="Original (unexpanded) humanized dataset for filtering")
    parser.add_argument("--sum_dataset", default="../data/kuhperdata-summarized")
    parser.add_argument("--corpus_dataset", default="../data/kuhperdata-exp")
    parser.add_argument("--output", default="data/pairs.tsv")
    parser.add_argument("--n_cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    hum_queries, _ = load_texts(args.hum_dataset)
    sum_queries, _ = load_texts(args.sum_dataset)
    _, corpus = load_texts(args.corpus_dataset)

    original_qrels = load_original_qrels(args.hum_dataset)
    n_original = sum(len(v) for v in original_qrels.values())
    print(f"Loaded {n_original} original qrels from {args.hum_dataset}")

    pairs_by_verdict = defaultdict(list)
    n_skipped_original = 0

    with open(args.expansion_logs, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["qid"]
            if entry.get("error") or qid not in hum_queries or qid not in sum_queries:
                continue

            candidate_doc_ids = entry["candidate_doc_ids"]
            verdicts = parse_raw_response(entry["raw_response"], candidate_doc_ids)

            for doc_id, verdict in verdicts.items():
                if doc_id not in corpus:
                    continue
                if doc_id in original_qrels.get(qid, set()):
                    n_skipped_original += 1
                    continue
                pairs_by_verdict[verdict].append({
                    "qid": qid,
                    "doc_id": doc_id,
                    "verdict": verdict,
                    "book": article_id_to_book(doc_id),
                })

    print(f"Skipped {n_skipped_original} pairs already in original qrels")

    print(f"Available RELEVAN pairs: {len(pairs_by_verdict.get('RELEVAN', []))}")
    print(f"Available NOT_RELEVAN pairs: {len(pairs_by_verdict.get('TIDAK_RELEVAN', []))}")

    n_rel = min(args.n_cases // 2, len(pairs_by_verdict.get("RELEVAN", [])))
    n_not = min(args.n_cases - n_rel, len(pairs_by_verdict.get("TIDAK_RELEVAN", [])))

    sampled_rel = random.sample(pairs_by_verdict.get("RELEVAN", []), n_rel)
    sampled_not = random.sample(pairs_by_verdict.get("TIDAK_RELEVAN", []), n_not)
    sampled = sampled_rel + sampled_not
    random.shuffle(sampled)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "pair_id", "case_id", "variant", "query_text", "article_id",
            "article_title", "article_text", "llm_label", "kuhperdata_book"
        ])

        case_num = 1
        for pair in sampled:
            qid = pair["qid"]
            doc_id = pair["doc_id"]
            doc = corpus[doc_id]

            for var_suffix, var_name, query_dict in [
                ("H", "humanized", hum_queries),
                ("S", "summarized", sum_queries),
            ]:
                pair_id = f"{var_suffix}{case_num:03d}"
                case_id = f"case_{case_num:03d}"
                writer.writerow([
                    pair_id,
                    case_id,
                    var_name,
                    query_dict.get(qid, ""),
                    doc_id,
                    doc["title"],
                    doc["text"],
                    pair["verdict"],
                    pair["book"],
                ])
            case_num += 1

    print(f"\nGenerated {len(sampled)} cases × 2 variants = {len(sampled) * 2} pairs")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
