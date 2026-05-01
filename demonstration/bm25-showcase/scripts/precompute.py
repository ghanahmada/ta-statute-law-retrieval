"""
Pre-compute BM25 results split into small static files for fast frontend loading.

Outputs:
  public/corpus.json         — 873 KB, all article texts
  public/index.json          — ~50 KB, sidebar data (qid, preview, gt info)
  public/queries/{qid}.json  — ~50-80 KB each, full ranking + highlights for one query
"""
import sys
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from util.bm25 import BM25
from util.dataloader import DataLoader


def tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def compute_bm25_per_token(query_text: str, doc_text: str, bm25: BM25) -> list:
    query_tokens = tokenize(query_text)
    doc_tokens = tokenize(doc_text)
    doc_token_counts = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    contributions = []
    seen = set()
    for token in query_tokens:
        if token in seen:
            continue
        seen.add(token)

        tf = doc_token_counts.get(token, 0)
        if tf == 0:
            contributions.append({"token": token, "contribution": 0.0, "idf": 0.0, "tf": 0, "in_doc": False})
            continue

        vocab = bm25.vectorizer.vocabulary_
        if token not in vocab:
            contributions.append({"token": token, "contribution": 0.0, "idf": 0.0, "tf": tf, "in_doc": True})
            continue

        token_idx = vocab[token]
        idf = bm25.vectorizer.idf_[token_idx] if hasattr(bm25.vectorizer, 'idf_') else 1.0

        k1, b = bm25.k1, bm25.b
        avgdl = bm25.avgdl if hasattr(bm25, 'avgdl') else doc_len
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
        contribution = idf * numerator / denominator if denominator > 0 else 0.0

        contributions.append({
            "token": token,
            "contribution": round(float(contribution), 4),
            "idf": round(float(idf), 4),
            "tf": tf,
            "in_doc": True,
        })

    return sorted(contributions, key=lambda x: -x["contribution"])


def highlight_positions(text: str, tokens_with_scores: list) -> list:
    token_scores = {t["token"]: t["contribution"] for t in tokens_with_scores if t["contribution"] > 0}
    if not token_scores:
        return []

    max_score = max(token_scores.values())
    highlights = []
    for token, score in token_scores.items():
        pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            highlights.append({
                "start": match.start(),
                "end": match.end(),
                "token": token,
                "intensity": round(score / max_score, 3) if max_score > 0 else 0,
                "score": round(score, 4),
            })

    return sorted(highlights, key=lambda x: x["start"])


def main():
    dataset_name = "kuhperdata-humanized"
    data_dir = f"data/{dataset_name}"
    top_k = 25
    max_relevant = 5

    print(f"Pre-computing BM25 showcase data for {dataset_name}...")

    loader = DataLoader(
        f"{data_dir}/corpus.jsonl",
        f"{data_dir}/queries.jsonl",
        f"{data_dir}/qrels_test.tsv",
    ).load()
    if max_relevant:
        loader.filter_max_relevant(max_relevant)

    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()

    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang="id",
                use_stemmer=False, use_stopwords=False)
    bm25.fit(doc_texts)
    bm25.avgdl = np.mean([len(tokenize(t)) for t in doc_texts])

    eval_qids = set(loader.qrels.keys())

    public_dir = Path(__file__).parent.parent / "public"
    queries_dir = public_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    # Save corpus once
    corpus_path = public_dir / "corpus.json"
    corpus_map = {did: {"title": loader.corpus[did].get("title", ""), "text": loader.corpus[did]["text"]} for did in doc_ids}
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_map, f, ensure_ascii=False)

    index_entries = []
    agg = {"total_queries": 0, "total_gt_docs": 0, "gt_in_top10": 0, "gt_in_top25": 0,
           "gt_rank_distribution": [], "queries_zero_hit": 0}

    for qid, query in zip(query_ids, query_texts):
        if qid not in eval_qids:
            continue

        gt_doc_ids = set(loader.qrels[qid].keys())
        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1]

        # GT ranks across full ranking
        gt_ranks = {}
        for rank, idx in enumerate(ranked_indices):
            did = doc_ids[idx]
            if did in gt_doc_ids:
                gt_ranks[did] = rank + 1

        # Build ranked docs: top_k + any GT docs ranked beyond top_k
        ranked_docs = []
        gt_found_in_topk = set()
        for rank in range(min(top_k, len(ranked_indices))):
            idx = ranked_indices[rank]
            did = doc_ids[idx]
            doc_text = loader.corpus[did]["text"]
            doc_title = loader.corpus[did].get("title", "")
            bm25_score = float(scores[idx])
            per_token = compute_bm25_per_token(query, doc_text, bm25)
            doc_highlights = highlight_positions(doc_text, per_token)
            query_highlights = highlight_positions(query, per_token)

            if did in gt_doc_ids:
                gt_found_in_topk.add(did)

            ranked_docs.append({
                "rank": rank + 1,
                "doc_id": did,
                "title": doc_title,
                "bm25_score": round(bm25_score, 4),
                "is_relevant": did in gt_doc_ids,
                "per_token": per_token[:20],
                "doc_highlights": doc_highlights,
                "query_highlights": query_highlights,
            })

        # Add GT docs that ranked beyond top_k (so frontend can show them)
        for did, gt_rank in sorted(gt_ranks.items(), key=lambda x: x[1]):
            if did in gt_found_in_topk:
                continue
            idx_in_corpus = doc_ids.index(did)
            doc_text = loader.corpus[did]["text"]
            doc_title = loader.corpus[did].get("title", "")
            bm25_score = float(scores[idx_in_corpus])
            per_token = compute_bm25_per_token(query, doc_text, bm25)
            doc_highlights = highlight_positions(doc_text, per_token)
            query_highlights = highlight_positions(query, per_token)

            ranked_docs.append({
                "rank": gt_rank,
                "doc_id": did,
                "title": doc_title,
                "bm25_score": round(bm25_score, 4),
                "is_relevant": True,
                "per_token": per_token[:20],
                "doc_highlights": doc_highlights,
                "query_highlights": query_highlights,
                "beyond_topk": True,
            })

        case_name = loader.queries[qid].get("metadata", {}).get("case_name", "")

        # Save per-query file
        query_data = {
            "qid": qid,
            "query_text": query,
            "case_name": case_name,
            "n_gt": len(gt_doc_ids),
            "gt_doc_ids": sorted(gt_doc_ids),
            "gt_ranks": gt_ranks,
            "ranked_docs": ranked_docs,
        }
        with open(queries_dir / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump(query_data, f, ensure_ascii=False)

        # Index entry (lightweight, for sidebar)
        has_hit = any(r <= 10 for r in gt_ranks.values())
        best_gt_rank = min(gt_ranks.values()) if gt_ranks else None
        worst_gt_rank = max(gt_ranks.values()) if gt_ranks else None
        index_entries.append({
            "qid": qid,
            "preview": query[:100],
            "case_name": case_name,
            "n_gt": len(gt_doc_ids),
            "has_hit": has_hit,
            "best_gt_rank": best_gt_rank,
            "worst_gt_rank": worst_gt_rank,
        })

        # Agg
        agg["total_queries"] += 1
        agg["total_gt_docs"] += len(gt_doc_ids)
        for did, rank in gt_ranks.items():
            agg["gt_rank_distribution"].append(rank)
            if rank <= 10:
                agg["gt_in_top10"] += 1
            if rank <= 25:
                agg["gt_in_top25"] += 1
        if not has_hit:
            agg["queries_zero_hit"] += 1

    gt_ranks_arr = np.array(agg["gt_rank_distribution"]) if agg["gt_rank_distribution"] else np.array([0])
    summary = {
        "dataset": dataset_name,
        "n_queries": agg["total_queries"],
        "n_gt_docs": agg["total_gt_docs"],
        "gt_in_top10": agg["gt_in_top10"],
        "gt_in_top25": agg["gt_in_top25"],
        "recall_at_10": agg["gt_in_top10"] / agg["total_gt_docs"] if agg["total_gt_docs"] else 0,
        "recall_at_25": agg["gt_in_top25"] / agg["total_gt_docs"] if agg["total_gt_docs"] else 0,
        "queries_zero_hit_at_10": agg["queries_zero_hit"],
        "gt_rank_median": float(np.median(gt_ranks_arr)),
        "gt_rank_mean": float(np.mean(gt_ranks_arr)),
    }

    index_path = public_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": index_entries}, f, ensure_ascii=False)

    # Cleanup old monolithic file
    old = public_dir / "rankings.json"
    if old.exists():
        old.unlink()

    total_q_size = sum(f.stat().st_size for f in queries_dir.glob("*.json"))
    print(f"Saved to {public_dir}/")
    print(f"  corpus.json:    {corpus_path.stat().st_size / 1024:.0f} KB")
    print(f"  index.json:     {index_path.stat().st_size / 1024:.0f} KB")
    print(f"  queries/*.json: {total_q_size / 1024 / 1024:.1f} MB ({agg['total_queries']} files)")
    print(f"  Recall@10: {summary['recall_at_10']:.3f}")
    print(f"  Recall@25: {summary['recall_at_25']:.3f}")


if __name__ == "__main__":
    main()
