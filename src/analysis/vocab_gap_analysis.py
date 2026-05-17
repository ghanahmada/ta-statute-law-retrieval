"""Vocabulary Gap Analysis across all datasets.

Characterizes the lexical gap between queries and their ground-truth documents:
  1. Token overlap statistics (Jaccard, % zero-overlap pairs)
  2. Per-dataset breakdown showing the gap is systematic (not random)
  3. Most discriminative terms: frequent in queries but absent in statutes, and vice versa

Dependencies:
  pip install nltk jieba

Usage:
  python src/analysis/vocab_gap_analysis.py
  python src/analysis/vocab_gap_analysis.py --datasets kuhperdata-exp bsard stard
  python src/analysis/vocab_gap_analysis.py --output_dir outputs/analysis/vocab_gap
"""

import argparse
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/analysis/vocab_gap")

DATASETS = [
    "kuhperdata-exp",
    "kuhperdata-summ-exp",
    "bsard",
    "stard",
    "ilpcsr",
    "coliee",
]

LANG_MAP = {
    "kuhperdata-exp": "id",
    "kuhperdata-summ-exp": "id",
    "bsard": "fr",
    "stard": "zh",
    "ilpcsr": "en",
    "coliee": "en",
}

NLTK_LANG = {
    "id": "indonesian",
    "fr": "french",
    "en": "english",
    "zh": None,
}


def get_stopwords(lang: str) -> set:
    nltk_lang = NLTK_LANG.get(lang)
    if nltk_lang is None:
        return set()
    try:
        return set(stopwords.words(nltk_lang))
    except OSError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words(nltk_lang))


def tokenize(text: str, lang: str, sw: set) -> list[str]:
    text = text.lower()
    if lang == "zh":
        import jieba
        tokens = list(jieba.cut(text))
        # Remove single-char punctuation and whitespace
        return [t.strip() for t in tokens if len(t.strip()) > 1 and re.search(r'[一-鿿]', t)]
    else:
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if t not in sw and len(t) > 1]


def load_queries(dataset: str) -> dict[str, str]:
    path = DATA_DIR / dataset / "queries.jsonl"
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            queries[rec["_id"]] = rec["text"]
    return queries


def load_corpus(dataset: str) -> dict[str, str]:
    path = DATA_DIR / dataset / "corpus.jsonl"
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("title", "") + " " + rec.get("text", "")
            corpus[rec["_id"]] = text.strip()
    return corpus


def load_qrels_test(dataset: str) -> dict[str, list[str]]:
    path = DATA_DIR / dataset / "qrels_test.tsv"
    qrels = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                qrels[parts[0]].append(parts[1])
    return dict(qrels)


def compute_overlap_stats(query_tokens: set, doc_tokens: set) -> dict:
    if not query_tokens or not doc_tokens:
        return {"jaccard": 0.0, "overlap_ratio": 0.0, "query_coverage": 0.0, "n_shared": 0}
    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    return {
        "jaccard": len(intersection) / len(union),
        "overlap_ratio": len(intersection) / min(len(query_tokens), len(doc_tokens)),
        "query_coverage": len(intersection) / len(query_tokens),
        "n_shared": len(intersection),
    }


def analyze_dataset(dataset: str, queries: dict, corpus: dict, qrels: dict) -> dict:
    lang = LANG_MAP[dataset]
    sw = get_stopwords(lang)
    n_queries = len(qrels)

    all_jaccard = []
    all_query_cov = []
    zero_overlap_count = 0
    total_pairs = 0

    query_only_terms = Counter()
    doc_only_terms = Counter()
    shared_terms = Counter()

    for qid, doc_ids in qrels.items():
        if qid not in queries:
            continue
        q_tokens = set(tokenize(queries[qid], lang, sw))
        if not q_tokens:
            continue

        for did in doc_ids:
            if did not in corpus:
                continue
            d_tokens = set(tokenize(corpus[did], lang, sw))
            if not d_tokens:
                continue

            stats = compute_overlap_stats(q_tokens, d_tokens)
            all_jaccard.append(stats["jaccard"])
            all_query_cov.append(stats["query_coverage"])
            total_pairs += 1

            if stats["n_shared"] == 0:
                zero_overlap_count += 1

            intersection = q_tokens & d_tokens
            q_exclusive = q_tokens - d_tokens
            d_exclusive = d_tokens - q_tokens

            for t in q_exclusive:
                query_only_terms[t] += 1
            for t in d_exclusive:
                doc_only_terms[t] += 1
            for t in intersection:
                shared_terms[t] += 1

    if not total_pairs:
        return {"dataset": dataset, "error": "no valid pairs"}

    jaccard_arr = np.array(all_jaccard)
    query_cov_arr = np.array(all_query_cov)

    return {
        "dataset": dataset,
        "language": lang,
        "n_queries": n_queries,
        "n_pairs": total_pairs,
        "zero_overlap_pct": 100 * zero_overlap_count / total_pairs,
        "jaccard_mean": float(jaccard_arr.mean()),
        "jaccard_median": float(np.median(jaccard_arr)),
        "jaccard_std": float(jaccard_arr.std()),
        "query_coverage_mean": float(query_cov_arr.mean()),
        "query_coverage_median": float(np.median(query_cov_arr)),
        "pct_below_10_jaccard": float(100 * (jaccard_arr < 0.10).sum() / len(jaccard_arr)),
        "pct_below_5_jaccard": float(100 * (jaccard_arr < 0.05).sum() / len(jaccard_arr)),
        "top_query_only_terms": query_only_terms.most_common(30),
        "top_doc_only_terms": doc_only_terms.most_common(30),
        "top_shared_terms": shared_terms.most_common(20),
    }


def print_results(results: list[dict]):
    print("\n" + "=" * 90)
    print("  VOCABULARY GAP ANALYSIS — SUMMARY")
    print("=" * 90)

    header = f"  {'Dataset':<22} {'Lang':<5} {'Pairs':>6} {'Zero%':>7} {'Jaccard':>8} {'QCov':>7} {'<5%J':>6} {'<10%J':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        if "error" in r:
            print(f"  {r['dataset']:<22} ERROR: {r['error']}")
            continue
        print(f"  {r['dataset']:<22} {r['language']:<5} {r['n_pairs']:>6} "
              f"{r['zero_overlap_pct']:>6.1f}% {r['jaccard_mean']:>7.4f} "
              f"{r['query_coverage_mean']:>6.3f} {r['pct_below_5_jaccard']:>5.1f}% "
              f"{r['pct_below_10_jaccard']:>5.1f}%")

    print("\n  Legend:")
    print("    Zero%  = % of query-doc pairs sharing zero content tokens")
    print("    Jaccard = mean Jaccard similarity (intersection/union of token sets)")
    print("    QCov   = mean query coverage (fraction of query tokens found in doc)")
    print("    <5%J   = % of pairs with Jaccard < 0.05")
    print("    <10%J  = % of pairs with Jaccard < 0.10")

    for r in results:
        if "error" in r:
            continue
        print(f"\n{'=' * 90}")
        print(f"  {r['dataset']} ({r['language']}) — Top Discriminative Terms")
        print(f"{'=' * 90}")

        print(f"\n  Query-only terms (in queries, absent from GT docs):")
        terms = r["top_query_only_terms"][:15]
        print(f"    {', '.join(f'{t}({c})' for t, c in terms)}")

        print(f"\n  Document-only terms (in GT docs, absent from queries):")
        terms = r["top_doc_only_terms"][:15]
        print(f"    {', '.join(f'{t}({c})' for t, c in terms)}")

        print(f"\n  Shared terms (bridge vocabulary):")
        terms = r["top_shared_terms"][:15]
        print(f"    {', '.join(f'{t}({c})' for t, c in terms)}")


def save_results(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "vocab_gap_summary.json"
    summary = []
    for r in results:
        s = {k: v for k, v in r.items()
             if k not in ("top_query_only_terms", "top_doc_only_terms", "top_shared_terms")}
        summary.append(s)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")

    detail_path = output_dir / "vocab_gap_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            serializable = {}
            for k, v in r.items():
                if k in ("top_query_only_terms", "top_doc_only_terms", "top_shared_terms"):
                    serializable[k] = [[t, c] for t, c in v]
                else:
                    serializable[k] = v
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    print(f"  Detail saved: {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Vocabulary gap analysis")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = []

    for dataset in args.datasets:
        data_path = DATA_DIR / dataset
        if not data_path.exists():
            print(f"  SKIP {dataset}: {data_path} not found")
            continue

        print(f"  Loading {dataset}...", end=" ", flush=True)
        queries = load_queries(dataset)
        corpus = load_corpus(dataset)
        qrels = load_qrels_test(dataset)
        print(f"{len(queries)} queries, {len(corpus)} docs, {len(qrels)} test qrels")

        print(f"  Analyzing {dataset}...", flush=True)
        result = analyze_dataset(dataset, queries, corpus, qrels)
        results.append(result)

    print_results(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
