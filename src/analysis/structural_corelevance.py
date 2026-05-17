"""Structural Co-Relevance Analysis.

For queries with multiple relevant articles, measures whether co-relevant
articles cluster structurally (same act, nearby positions) vs random baselines.
Motivates why StructGNN's positional encoding helps.

Dependencies: nltk, jieba, numpy

Usage:
  python src/analysis/structural_corelevance.py
  python src/analysis/structural_corelevance.py --datasets kuhperdata-exp bsard stard
"""

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paragnn.structure import build_structure_metadata

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/analysis/structural_corelevance")

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

NLTK_LANG = {"id": "indonesian", "fr": "french", "en": "english", "zh": None}


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


def pairwise_jaccard(doc_tokens: dict[str, set], doc_ids: list[str]) -> float:
    if len(doc_ids) < 2:
        return 0.0
    jaccards = []
    for d_i, d_j in combinations(doc_ids, 2):
        t_i = doc_tokens.get(d_i, set())
        t_j = doc_tokens.get(d_j, set())
        if not t_i or not t_j:
            jaccards.append(0.0)
            continue
        inter = len(t_i & t_j)
        union = len(t_i | t_j)
        jaccards.append(inter / union if union else 0.0)
    return float(np.mean(jaccards))


def concentration_at_window(positions: list[float], window: float = 0.1) -> float:
    if len(positions) <= 1:
        return 1.0
    sorted_pos = sorted(positions)
    n = len(sorted_pos)
    best = 0
    for i in range(n):
        count = sum(1 for p in sorted_pos[i:] if p - sorted_pos[i] <= window)
        best = max(best, count)
    return best / n


def analyze_dataset(dataset: str, n_random: int = 1000) -> dict:
    lang = LANG_MAP[dataset]
    corpus_path = str(DATA_DIR / dataset / "corpus.jsonl")

    metadata = build_structure_metadata(corpus_path, dataset)
    qrels = load_qrels_test(dataset)
    corpus = load_corpus(dataset)

    all_doc_ids = list(metadata.keys())
    # Get act group sizes for converting position distance to article distance
    act_groups = defaultdict(list)
    for doc_id, meta in metadata.items():
        act_groups[meta["act_name"]].append(doc_id)

    # --- Section A: Query-to-Structure Mapping ---
    n_relevant_list = []
    n_acts_list = []
    pos_span_list = []
    concentration_list = []

    # --- Section B: Co-Relevance Pairs ---
    corelevant_same_act = []
    corelevant_pos_dist = []
    corelevant_article_dist = []

    multi_rel_queries = 0

    for qid, doc_ids in qrels.items():
        valid_docs = [d for d in doc_ids if d in metadata]
        if not valid_docs:
            continue

        n_relevant_list.append(len(valid_docs))
        acts = set(metadata[d]["act_name"] for d in valid_docs)
        n_acts_list.append(len(acts))
        positions = [metadata[d]["position"] for d in valid_docs]
        pos_span_list.append(max(positions) - min(positions))
        concentration_list.append(concentration_at_window(positions, 0.1))

        if len(valid_docs) >= 2:
            multi_rel_queries += 1
            for d_i, d_j in combinations(valid_docs, 2):
                m_i, m_j = metadata[d_i], metadata[d_j]
                same_act = m_i["act_name"] == m_j["act_name"]
                corelevant_same_act.append(same_act)
                pos_dist = abs(m_i["position"] - m_j["position"])
                corelevant_pos_dist.append(pos_dist)
                # Approximate article distance using group size
                if same_act:
                    group_size = len(act_groups[m_i["act_name"]])
                    corelevant_article_dist.append(pos_dist * group_size)
                else:
                    corelevant_article_dist.append(float("nan"))

    # --- Section C: Random Baseline ---
    rng = np.random.default_rng(42)
    random_same_act = []
    random_pos_dist = []
    random_article_dist = []

    for _ in range(n_random):
        i, j = rng.choice(len(all_doc_ids), size=2, replace=False)
        d_i, d_j = all_doc_ids[i], all_doc_ids[j]
        m_i, m_j = metadata[d_i], metadata[d_j]
        same_act = m_i["act_name"] == m_j["act_name"]
        random_same_act.append(same_act)
        random_pos_dist.append(abs(m_i["position"] - m_j["position"]))
        if same_act:
            group_size = len(act_groups[m_i["act_name"]])
            random_article_dist.append(abs(m_i["position"] - m_j["position"]) * group_size)
        else:
            random_article_dist.append(float("nan"))

    # Within-N proximity (using approximate article distance)
    valid_corel_dist = [d for d in corelevant_article_dist if not np.isnan(d)]
    valid_random_dist = [d for d in random_article_dist if not np.isnan(d)]

    within_n = {}
    for n_thresh in [5, 10, 20, 50]:
        corel_within = sum(1 for d in valid_corel_dist if d <= n_thresh) / max(len(valid_corel_dist), 1)
        rand_within = sum(1 for d in valid_random_dist if d <= n_thresh) / max(len(valid_random_dist), 1)
        within_n[n_thresh] = {"corelevant": corel_within, "random": rand_within}

    # --- Lexical Cohesion ---
    sw = get_stopwords(lang)
    doc_tokens = {}
    # Only tokenize docs that appear in multi-relevant queries (for efficiency)
    relevant_doc_set = set()
    for qid, doc_ids in qrels.items():
        if len([d for d in doc_ids if d in metadata]) >= 2:
            relevant_doc_set.update(d for d in doc_ids if d in metadata)

    for doc_id in relevant_doc_set:
        if doc_id in corpus:
            doc_tokens[doc_id] = tokenize(corpus[doc_id], lang, sw)

    # Co-relevant lexical cohesion
    corel_jaccards = []
    for qid, doc_ids in qrels.items():
        valid_docs = [d for d in doc_ids if d in doc_tokens]
        if len(valid_docs) >= 2:
            corel_jaccards.append(pairwise_jaccard(doc_tokens, valid_docs))

    # Random group lexical cohesion (sample groups of same avg size)
    avg_group_size = int(np.mean(n_relevant_list)) if n_relevant_list else 2
    avg_group_size = max(avg_group_size, 2)
    tokenized_doc_ids = list(doc_tokens.keys())

    random_jaccards = []
    if len(tokenized_doc_ids) >= avg_group_size:
        for _ in range(min(200, n_random)):
            sample = rng.choice(len(tokenized_doc_ids), size=min(avg_group_size, len(tokenized_doc_ids)), replace=False)
            group = [tokenized_doc_ids[i] for i in sample]
            random_jaccards.append(pairwise_jaccard(doc_tokens, group))

    return {
        "dataset": dataset,
        "language": lang,
        "n_queries_total": len(qrels),
        "n_queries_multi_rel": multi_rel_queries,
        "n_corelevant_pairs": len(corelevant_same_act),
        # Section A
        "avg_relevant": float(np.mean(n_relevant_list)),
        "median_relevant": float(np.median(n_relevant_list)),
        "max_relevant": int(max(n_relevant_list)),
        "avg_acts": float(np.mean(n_acts_list)),
        "avg_pos_span": float(np.mean(pos_span_list)),
        "avg_concentration_01": float(np.mean(concentration_list)),
        # Section B
        "same_act_rate": float(np.mean(corelevant_same_act)) if corelevant_same_act else 0.0,
        "avg_pos_dist": float(np.mean(corelevant_pos_dist)) if corelevant_pos_dist else 0.0,
        "avg_article_dist": float(np.nanmean(valid_corel_dist)) if valid_corel_dist else 0.0,
        "within_n": within_n,
        "lexical_cohesion": float(np.mean(corel_jaccards)) if corel_jaccards else 0.0,
        # Section C (random)
        "random_same_act_rate": float(np.mean(random_same_act)),
        "random_avg_pos_dist": float(np.mean(random_pos_dist)),
        "random_avg_article_dist": float(np.nanmean(valid_random_dist)) if valid_random_dist else 0.0,
        "random_lexical_cohesion": float(np.mean(random_jaccards)) if random_jaccards else 0.0,
    }


def print_results(results: list[dict]):
    print("\n" + "=" * 100)
    print("  TABLE 1: QUERY-TO-STRUCTURE MAPPING (Prior)")
    print("=" * 100)
    header = f"  {'Dataset':<22} {'Queries':>7} {'MultiRel':>8} {'AvgRel':>7} {'MedRel':>7} {'AvgActs':>7} {'AvgSpan':>7} {'Conc@0.1':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['dataset']:<22} {r['n_queries_total']:>7} {r['n_queries_multi_rel']:>8} "
              f"{r['avg_relevant']:>7.2f} {r['median_relevant']:>7.1f} {r['avg_acts']:>7.2f} "
              f"{r['avg_pos_span']:>7.4f} {r['avg_concentration_01']:>7.1%}")

    print("\n" + "=" * 100)
    print("  TABLE 2: STRUCTURAL CO-RELEVANCE vs RANDOM")
    print("=" * 100)
    header2 = f"  {'Dataset':<22} {'SameAct':>7} {'Rnd':>5} {'PosDist':>7} {'Rnd':>7} {'ArtDist':>7} {'Rnd':>7} {'LexJ':>6} {'Rnd':>6}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for r in results:
        print(f"  {r['dataset']:<22} {r['same_act_rate']:>6.1%} {r['random_same_act_rate']:>5.1%} "
              f"{r['avg_pos_dist']:>7.4f} {r['random_avg_pos_dist']:>7.4f} "
              f"{r['avg_article_dist']:>7.1f} {r['random_avg_article_dist']:>7.1f} "
              f"{r['lexical_cohesion']:>6.4f} {r['random_lexical_cohesion']:>6.4f}")

    print("\n" + "=" * 100)
    print("  TABLE 3: WITHIN-N PROXIMITY (% of same-act co-relevant pairs within N articles)")
    print("=" * 100)
    header3 = f"  {'Dataset':<22} {'W5':>6} {'R5':>5} {'W10':>6} {'R10':>5} {'W20':>6} {'R20':>5} {'W50':>6} {'R50':>5}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))
    for r in results:
        w = r["within_n"]
        print(f"  {r['dataset']:<22} "
              f"{w[5]['corelevant']:>5.1%} {w[5]['random']:>5.1%} "
              f"{w[10]['corelevant']:>5.1%} {w[10]['random']:>5.1%} "
              f"{w[20]['corelevant']:>5.1%} {w[20]['random']:>5.1%} "
              f"{w[50]['corelevant']:>5.1%} {w[50]['random']:>5.1%}")

    # Signal strength ratios
    print("\n" + "=" * 100)
    print("  SIGNAL STRENGTH (observed / random ratio)")
    print("=" * 100)
    for r in results:
        pos_ratio = r['random_avg_pos_dist'] / max(r['avg_pos_dist'], 1e-6)
        lex_ratio = r['lexical_cohesion'] / max(r['random_lexical_cohesion'], 1e-6)
        print(f"  {r['dataset']:<22} position: {pos_ratio:.1f}x closer | lexical: {lex_ratio:.1f}x more cohesive")


def save_results(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "corelevance_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")

    detail_path = output_dir / "corelevance_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Detail saved: {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Structural co-relevance analysis")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--n_random", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = []

    for dataset in args.datasets:
        data_path = DATA_DIR / dataset
        if not data_path.exists():
            print(f"  SKIP {dataset}: {data_path} not found")
            continue

        print(f"  Analyzing {dataset}...", flush=True)
        result = analyze_dataset(dataset, n_random=args.n_random)
        results.append(result)

    print_results(results)
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
