"""Analysis 03 Part 1: Vocabulary Gap × Structural Benefit Coupling.

Shows that StructGNN's structural features help MOST on queries where the
vocabulary gap is worst, and help LEAST when initial retrieval completely
misses the right neighborhood.

Usage:
  python src/analysis/gap_structure_coupling.py
  python src/analysis/gap_structure_coupling.py --datasets kuhperdata-exp bsard
"""

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

DATA_DIR = Path("data")
PRED_DIR = Path("outputs/predictions")
OUTPUT_DIR = Path("outputs/analysis/gap_structure_coupling")

DATASETS = ["kuhperdata-exp", "kuhperdata-summ-exp", "bsard", "stard"]

LANG_MAP = {
    "kuhperdata-exp": "id",
    "kuhperdata-summ-exp": "id",
    "bsard": "fr",
    "stard": "zh",
    "ilpcsr": "en",
    "coliee": "en",
}

NLTK_LANG = {"id": "indonesian", "fr": "french", "en": "english", "zh": None}

METHODS = ["bm25", "dense_bge_m3", "paragnn", "structgnn"]
METHOD_LABELS = {"bm25": "BM25", "dense_bge_m3": "Dense", "paragnn": "ParaGNN", "structgnn": "StructGNN"}


def get_stopwords(lang: str) -> set:
    nltk_lang = NLTK_LANG.get(lang)
    if nltk_lang is None:
        return set()
    try:
        return set(stopwords.words(nltk_lang))
    except OSError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words(nltk_lang))


def tokenize(text: str, lang: str, sw: set) -> set:
    text = text.lower()
    if lang == "zh":
        import jieba
        tokens = list(jieba.cut(text))
        return {t.strip() for t in tokens if len(t.strip()) > 1 and re.search(r'[一-鿿]', t)}
    else:
        tokens = re.findall(r'\b\w+\b', text)
        return {t for t in tokens if t not in sw and len(t) > 1}


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


def load_predictions(path: Path) -> dict[str, list[str]]:
    rankings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["qid"]
            if "ranked_doc_ids" in rec:
                rankings[qid] = rec["ranked_doc_ids"]
            elif "rankings" in rec:
                rankings[qid] = [r["doc_id"] for r in rec["rankings"]]
    return rankings


def compute_mrr(rankings: dict[str, list[str]], qrels: dict[str, list[str]], qids: list[str], k: int = 10) -> float:
    rrs = []
    for qid in qids:
        gt = set(qrels.get(qid, []))
        ranked = rankings.get(qid, [])[:k]
        rr = 0.0
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in gt:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def compute_per_query_jaccard(queries: dict, corpus: dict, qrels: dict, lang: str) -> dict[str, float]:
    sw = get_stopwords(lang)
    per_query_max_jaccard = {}

    for qid, doc_ids in qrels.items():
        if qid not in queries:
            continue
        q_tokens = tokenize(queries[qid], lang, sw)
        if not q_tokens:
            per_query_max_jaccard[qid] = 0.0
            continue

        max_j = 0.0
        for did in doc_ids:
            if did not in corpus:
                continue
            d_tokens = tokenize(corpus[did], lang, sw)
            if not d_tokens:
                continue
            inter = len(q_tokens & d_tokens)
            union = len(q_tokens | d_tokens)
            j = inter / union if union else 0.0
            max_j = max(max_j, j)
        per_query_max_jaccard[qid] = max_j

    return per_query_max_jaccard


def stratify_queries(per_query_jaccard: dict[str, float]) -> dict[str, list[str]]:
    strata = {"zero": [], "low": [], "moderate": [], "high": []}
    for qid, j in per_query_jaccard.items():
        if j == 0.0:
            strata["zero"].append(qid)
        elif j < 0.05:
            strata["low"].append(qid)
        elif j <= 0.20:
            strata["moderate"].append(qid)
        else:
            strata["high"].append(qid)
    return strata


def analyze_dataset(dataset: str, pred_dir: Path = None) -> dict:
    if pred_dir is None:
        pred_dir = PRED_DIR
    lang = LANG_MAP[dataset]
    print(f"  Loading {dataset}...", flush=True)

    queries = load_queries(dataset)
    corpus = load_corpus(dataset)
    qrels = load_qrels_test(dataset)

    print(f"  Computing per-query Jaccard...", flush=True)
    per_query_jaccard = compute_per_query_jaccard(queries, corpus, qrels, lang)
    strata = stratify_queries(per_query_jaccard)

    # Load predictions for each method
    method_rankings = {}
    for method in METHODS:
        pred_path = pred_dir / f"{method}_{dataset}.jsonl"
        if pred_path.exists():
            method_rankings[method] = load_predictions(pred_path)
        else:
            print(f"    SKIP {method}: {pred_path} not found")

    # Per-stratum MRR
    stratum_results = {}
    for stratum_name, qids in strata.items():
        if not qids:
            continue
        row = {"n": len(qids)}
        for method, rankings in method_rankings.items():
            row[method] = compute_mrr(rankings, qrels, qids)
        if "paragnn" in row and "structgnn" in row:
            row["delta"] = row["structgnn"] - row["paragnn"]
        stratum_results[stratum_name] = row

    # Never-retrieved analysis (StructGNN MRR=0 queries)
    structgnn_rankings = method_rankings.get("structgnn", {})
    dense_rankings = method_rankings.get("dense_bge_m3", {})

    structgnn_fail_qids = []
    for qid in per_query_jaccard:
        gt = set(qrels.get(qid, []))
        ranked = structgnn_rankings.get(qid, [])[:10]
        if not any(d in gt for d in ranked):
            structgnn_fail_qids.append(qid)

    # Of those failures, how many also have no GT in dense top-100?
    also_missed_dense = 0
    for qid in structgnn_fail_qids:
        gt = set(qrels.get(qid, []))
        dense_top100 = dense_rankings.get(qid, [])[:100]
        if not any(d in gt for d in dense_top100):
            also_missed_dense += 1

    return {
        "dataset": dataset,
        "n_queries": len(per_query_jaccard),
        "strata": stratum_results,
        "structgnn_fail_count": len(structgnn_fail_qids),
        "also_missed_dense_100": also_missed_dense,
        "fail_also_missed_pct": also_missed_dense / max(len(structgnn_fail_qids), 1),
    }


def print_results(results: list[dict]):
    for r in results:
        print(f"\n{'=' * 100}")
        print(f"  {r['dataset']} — PERFORMANCE BY VOCABULARY GAP STRATUM")
        print(f"{'=' * 100}")

        available_methods = []
        for method in METHODS:
            # Check if any stratum has this method
            if any(method in s for s in r["strata"].values()):
                available_methods.append(method)

        header = f"  {'Stratum':<10} {'N':>5}"
        for m in available_methods:
            header += f" {METHOD_LABELS.get(m, m):>10}"
        if "paragnn" in available_methods and "structgnn" in available_methods:
            header += f" {'Δ(S-P)':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for stratum_name in ["zero", "low", "moderate", "high"]:
            if stratum_name not in r["strata"]:
                continue
            s = r["strata"][stratum_name]
            row = f"  {stratum_name:<10} {s['n']:>5}"
            for m in available_methods:
                row += f" {s.get(m, 0):>10.4f}"
            if "delta" in s:
                delta = s["delta"]
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:>7.4f}"
            print(row)

        # Never-retrieved
        print(f"\n  NEVER-RETRIEVED ANALYSIS:")
        print(f"    StructGNN MRR=0 queries: {r['structgnn_fail_count']} / {r['n_queries']}")
        print(f"    Of those, also missed in Dense top-100: {r['also_missed_dense_100']} "
              f"({r['fail_also_missed_pct']:.1%})")
        print(f"    → When StructGNN fails, it's {r['fail_also_missed_pct']:.0%} because dense "
              f"retrieval also missed the entire neighborhood")


def save_results(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "coupling_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--pred_dir", type=str, default=str(PRED_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)

    results = []
    for dataset in args.datasets:
        if not (DATA_DIR / dataset).exists():
            print(f"  SKIP {dataset}: not found")
            continue
        result = analyze_dataset(dataset, pred_dir=pred_dir)
        results.append(result)

    print_results(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
