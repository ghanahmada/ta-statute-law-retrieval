"""Prepare StructGNN inference results for the Next.js demo.

Converts inference JSON into per-query JSON files + index.json
matching the format expected by the StructGNN showcase demo.

Usage:
  python src/inference/prepare_demo_data.py --dataset kuhperdata-humanized
  python src/inference/prepare_demo_data.py --dataset kuhperdata-humanized --output_dir demonstration/structgnn-showcase/public
"""
import argparse
import json
from pathlib import Path


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "stard": {"path": "data/stard", "lang": "zh"},
}


def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}
    return corpus


def load_queries(queries_path):
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = {
                "text": q["text"],
                "metadata": q.get("metadata", {}),
            }
    return queries


def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0] in ("query-id", "query_id"):
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def prepare(dataset_name, inference_dir, output_dir, max_relevant=5):
    cfg = DATASETS[dataset_name]
    data_path = cfg["path"]

    corpus = load_corpus(f"{data_path}/corpus.jsonl")
    queries = load_queries(f"{data_path}/queries.jsonl")
    qrels = load_qrels(f"{data_path}/qrels_test.tsv")

    inf_path = Path(inference_dir) / dataset_name / "structgnn_adapted.json"
    if not inf_path.exists():
        print(f"No inference file at {inf_path}")
        return

    with open(inf_path, "r", encoding="utf-8") as f:
        inference = json.load(f)

    inf_config = inference["config"]
    inf_metrics = inference["metrics"]
    inf_results = inference["results"]

    if max_relevant > 0:
        qrels = {
            qid: docs
            for qid, docs in qrels.items()
            if len(docs) <= max_relevant
        }

    out_dir = Path(output_dir)
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    query_index = []
    total_gt = 0
    gt_in_top10 = 0
    gt_in_top25 = 0
    zero_hit_at_10 = 0
    gnn_helped = 0
    gt_ranks_all = []

    for qid, qresult in inf_results.items():
        if qid not in queries:
            continue
        gt_docs = qrels.get(qid, set())
        if max_relevant > 0 and len(gt_docs) > max_relevant:
            continue

        query_text = queries[qid]["text"]
        case_name = queries[qid].get("metadata", {}).get("case_name", "")

        ranked_docs = []
        retrieved_ids = set()
        has_hit_10 = False
        has_hit_25 = False
        best_gt_rank = None
        worst_gt_rank = None

        for doc_entry in qresult["ranked"]:
            doc_id = doc_entry["doc_id"]
            is_beyond = doc_entry.get("beyond_topk", False)
            retrieved_ids.add(doc_id)

            doc_info = corpus.get(doc_id, {"title": doc_id, "text": ""})
            is_relevant = doc_id in gt_docs

            ranked_docs.append({
                "rank": doc_entry["rank"],
                "doc_id": doc_id,
                "title": doc_info["title"],
                "score": round(doc_entry["score"], 4),
                "gnn_score": round(doc_entry["gnn_score"], 4),
                "bm25_score": round(doc_entry["bm25_score"], 4),
                "is_relevant": is_relevant,
                "beyond_topk": is_beyond,
            })

            if is_relevant and not is_beyond:
                rank = doc_entry["rank"]
                gt_ranks_all.append(rank)
                if rank <= 10:
                    has_hit_10 = True
                    gt_in_top10 += 1
                if rank <= 25:
                    has_hit_25 = True
                    gt_in_top25 += 1
                if best_gt_rank is None or rank < best_gt_rank:
                    best_gt_rank = rank
                if worst_gt_rank is None or rank > worst_gt_rank:
                    worst_gt_rank = rank

        gt_ranks = {}
        for doc in ranked_docs:
            if doc["is_relevant"] and doc["rank"] is not None:
                gt_ranks[doc["doc_id"]] = doc["rank"]

        bm25_top10_hits = sorted(qresult["ranked"], key=lambda x: -x["bm25_score"])[:10]
        bm25_would_hit = any(d["doc_id"] in gt_docs for d in bm25_top10_hits if not d.get("beyond_topk"))
        if has_hit_10 and not bm25_would_hit:
            gnn_helped += 1

        if not has_hit_10:
            zero_hit_at_10 += 1

        total_gt += len(gt_docs)

        query_data = {
            "qid": qid,
            "query_text": query_text,
            "case_name": case_name,
            "n_gt": len(gt_docs),
            "gt_doc_ids": sorted(gt_docs),
            "gt_ranks": gt_ranks,
            "ranked_docs": ranked_docs,
        }

        with open(queries_dir / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump(query_data, f, ensure_ascii=False, indent=2)

        query_index.append({
            "qid": qid,
            "preview": query_text[:100],
            "case_name": case_name,
            "n_gt": len(gt_docs),
            "has_hit": has_hit_10,
            "best_gt_rank": best_gt_rank,
            "worst_gt_rank": worst_gt_rank,
        })

    n_queries = len(query_index)
    gt_rank_median = sorted(gt_ranks_all)[len(gt_ranks_all) // 2] if gt_ranks_all else 0
    gt_rank_mean = sum(gt_ranks_all) / len(gt_ranks_all) if gt_ranks_all else 0

    index_data = {
        "summary": {
            "dataset": dataset_name,
            "model": inf_config.get("structure_mode", "structural"),
            "alpha": inf_config.get("alpha", 0.9),
            "n_queries": n_queries,
            "n_gt_docs": total_gt,
            "gt_in_top10": gt_in_top10,
            "gt_in_top25": gt_in_top25,
            "recall_at_10": gt_in_top10 / total_gt if total_gt else 0,
            "recall_at_25": gt_in_top25 / total_gt if total_gt else 0,
            "mrr_at_10": inf_metrics.get("mrr@10", 0),
            "hit_rate": inf_metrics.get("hit_rate", 0),
            "queries_zero_hit_at_10": zero_hit_at_10,
            "gnn_helped": gnn_helped,
            "gt_rank_median": gt_rank_median,
            "gt_rank_mean": gt_rank_mean,
        },
        "queries": query_index,
    }

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    # Also copy corpus
    corpus_out = {did: {"title": d["title"], "text": d["text"]} for did, d in corpus.items()}
    with open(out_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_out, f, ensure_ascii=False)

    print(f"Prepared {n_queries} queries for {dataset_name}")
    print(f"  Output: {out_dir}")
    print(f"  GT in top-10: {gt_in_top10}/{total_gt} ({gt_in_top10/total_gt:.1%})")
    print(f"  GT in top-25: {gt_in_top25}/{total_gt} ({gt_in_top25/total_gt:.1%})")
    print(f"  GNN helped: {gnn_helped} queries")


def main():
    parser = argparse.ArgumentParser(description="Prepare StructGNN demo data")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=list(DATASETS.keys()))
    parser.add_argument("--inference_dir", default="outputs/inference")
    parser.add_argument("--output_dir", default="demonstration/bm25-showcase/public/structgnn")
    parser.add_argument("--max_relevant", type=int, default=5)
    args = parser.parse_args()

    prepare(args.dataset, args.inference_dir, args.output_dir, args.max_relevant)


if __name__ == "__main__":
    main()
