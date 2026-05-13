"""Compare train/val/test distribution to diagnose why StructGNN degrades.

Four dimensions measured per split:
  1. BM25 signal (can lexical retrieval find relevant docs?)
  2. Label distribution (do splits share the same relevant articles?)
  3. Embedding separation (are queries topically separated?)
  4. Relevance density (how many relevant docs per query?)

Explains why humanized-exp works but summ-exp fails.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import calculate_mrr, calculate_recall_at_k

DATASETS = {
    "humanized-exp":  "data/kuhperdata-exp",
    "summarized-exp": "data/kuhperdata-summ-exp",
}


def load_corpus(data_dir):
    doc_ids, doc_texts = [], []
    with open(f"{data_dir}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_ids.append(d["_id"])
            doc_texts.append(d["text"])
    return doc_ids, doc_texts


def load_split(data_dir, split):
    qp = f"{data_dir}/qrels_{split}.tsv"
    if not Path(qp).exists():
        return None
    loader = DataLoader(f"{data_dir}/corpus.jsonl",
                        f"{data_dir}/queries.jsonl", qp).load()
    return loader


def measure_all(data_dir, doc_ids, doc_texts):
    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang="id")
    bm25.fit(doc_texts)

    results = {}
    for split in ["train", "val", "test"]:
        loader = load_split(data_dir, split)
        if loader is None:
            continue

        qids, qtexts = loader.get_query_texts()
        qrel_counts = [len(list(loader.qrels.get(q, {}).keys())) for q in qids]
        qrel_items = {q: list(loader.qrels.get(q, {}).keys()) for q in qids}

        mrr10, mrr50, rec10, rec50 = [], [], [], []
        hit10, hit50 = 0, 0
        overlaps = []

        for qid, qt in zip(qids, qtexts):
            scores = bm25.transform(qt)
            idx = np.argsort(-np.array(scores))[:50]
            ranked_ids = [doc_ids[i] for i in idx]
            gt = qrel_items[qid]
            if not gt:
                continue

            mr10 = calculate_mrr(ranked_ids[:10], gt, 10)
            mr50 = calculate_mrr(ranked_ids[:50], gt, 50)
            re10 = calculate_recall_at_k(ranked_ids[:10], gt, 10)
            re50 = calculate_recall_at_k(ranked_ids[:50], gt, 50)

            mrr10.append(mr10)
            mrr50.append(mr50)
            rec10.append(re10)
            rec50.append(re50)
            if mr10 > 0:
                hit10 += 1
            if mr50 > 0:
                hit50 += 1

            qtok = set(qt.lower().split())
            for did in gt:
                if did in loader.corpus:
                    dtok = set(loader.corpus[did]["text"].lower().split())
                    overlaps.append(len(qtok & dtok))

        n = len(mrr10) if mrr10 else 1
        results[split] = {
            "n": len(qids),
            "bm25_mrr10": float(np.mean(mrr10)) if mrr10 else 0.0,
            "bm25_mrr50": float(np.mean(mrr50)) if mrr50 else 0.0,
            "bm25_rec10": float(np.mean(rec10)) if rec10 else 0.0,
            "bm25_rec50": float(np.mean(rec50)) if rec50 else 0.0,
            "bm25_hit10": hit10 / n,
            "bm25_hit50": hit50 / n,
            "avg_chars": float(np.mean([len(t) for t in qtexts])),
            "avg_rel": float(np.mean(qrel_counts)),
            "median_rel": float(np.median(qrel_counts)),
            "p90_rel": float(np.percentile(qrel_counts, 90)),
            "pct_ge3_rel": sum(1 for c in qrel_counts if c >= 3) / len(qrel_counts) * 100,
            "avg_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
            "article_set": {did for q in qids for did in loader.qrels.get(q, {})},
            "article_counts": Counter(did for q in qids for did in loader.qrels.get(q, {})),
            "qrels_counts": Counter({qid: len(docs) for qid, docs in loader.qrels.items()}),
            "qids": qids,
            "qtexts": qtexts,
        }
    return results


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def report(results, name):
    print(f"\n{'=' * 75}")
    print(f"  {name}")
    print(f"{'=' * 75}")

    splits = [s for s in ["train", "val", "test"] if s in results]
    if not splits:
        print("  No splits found (run split_test_to_val.py first)")
        return

    # 1. BM25 Signal
    print("\n--- 1. BM25 Signal ---")
    header = (f"  {'Split':>6} | {'n':>5} | {'Chars':>7} | {'Rel':>4} | "
              f"{'Overlap':>7} | {'MRR@10':>7} | {'R@10':>7} | "
              f"{'Hit10':>6} | {'MRR@50':>7} | {'R@50':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for s in splits:
        d = results[s]
        print(f"  {s:>6} | {d['n']:>5} | {d['avg_chars']:>7.0f} | "
              f"{d['avg_rel']:>4.1f} | {d['avg_overlap']:>7.1f} | "
              f"{d['bm25_mrr10']:>7.4f} | {d['bm25_rec10']:>7.4f} | "
              f"{d['bm25_hit10']:>5.1%} | {d['bm25_mrr50']:>7.4f} | "
              f"{d['bm25_rec50']:>7.4f}")

    if "val" in results and "test" in results:
        gap = results["test"]["bm25_mrr10"] - results["val"]["bm25_mrr10"]
        ratio = results["test"]["bm25_mrr10"] / max(results["val"]["bm25_mrr10"], 1e-6)
        gap_overlap = results["test"]["avg_overlap"] - results["val"]["avg_overlap"]
        print(f"\n  BM25 MRR gap (test - val): {gap:+.4f}  "
              f"(test/val = {ratio:.1f}x)")
        print(f"  Overlap gap (test - val):  {gap_overlap:+.1f}")

    # 2. Label Distribution
    print("\n--- 2. Label Distribution ---")
    print(f"  {'Split':>6} | {'n':>5} | {'Art.(set)':>10} | {'Top-5 hub articles':>45}")
    print(f"  {'-' * (28 + 45)}")
    for s in splits:
        r = results[s]
        top5 = r["article_counts"].most_common(5)
        top_str = ", ".join(f"{a}({c})" for a, c in top5)
        print(f"  {s:>6} | {r['n']:>5} | {len(r['article_set']):>10} | {top_str}")

    if all(s in results for s in ["train", "val", "test"]):
        j_tv = jaccard(results["train"]["article_set"], results["val"]["article_set"])
        j_tt = jaccard(results["train"]["article_set"], results["test"]["article_set"])
        j_vt = jaccard(results["val"]["article_set"], results["test"]["article_set"])
        print(f"\n  Article Jaccard overlap:")
        print(f"    train↔val: {j_tv:.4f}   "
              f"train↔test: {j_tt:.4f}   "
              f"val↔test: {j_vt:.4f}")

    if "train" in results and "val" in results:
        val_only = results["val"]["article_set"] - results["train"]["article_set"]
        train_only_val = results["val"]["article_set"] & results["train"]["article_set"]
        print(f"\n  Articles in val NOT in train: {len(val_only)}/{len(results['val']['article_set'])}")
        if len(val_only) > 0 and len(val_only) <= 15:
            print(f"    Unseen articles: {sorted(val_only)}")
    if "train" in results and "test" in results:
        test_only = results["test"]["article_set"] - results["train"]["article_set"]
        print(f"  Articles in test NOT in train: {len(test_only)}/{len(results['test']['article_set'])}")

    # 3. Embedding Separation
    print("\n--- 3. Embedding Separation (BGE-M3) ---")
    try:
        from FlagEmbedding import BGEM3FlagModel
        encoder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    except Exception as e:
        print(f"  (BGE-M3 not available: {e})")
        print(f"  (Skipping embedding analysis)")
        return

    quids = {}
    qembs = {}
    for s in splits:
        texts = results[s]["qtexts"]
        out = encoder.encode(texts, batch_size=64, max_length=512)
        if isinstance(out, dict):
            out = out["dense_vecs"]
        quids[s] = results[s]["qids"]
        qembs[s] = out

    print(f"  {'':>14} | " + " | ".join(f"{s:>8}" for s in splits))
    print(f"  {'-' * 14}-+-" + "-+-".join("-" * 8 for _ in splits))
    for s1 in splits:
        row = []
        for s2 in splits:
            e1, e2 = qembs[s1], qembs[s2]
            n1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8)
            if s1 == s2:
                sim = n1 @ n1.T
                n = sim.shape[0]
                if n > 1:
                    triu = sim[np.triu_indices(n, k=1)]
                    dist = 1 - float(np.mean(triu))
                else:
                    dist = 0.0
            else:
                n2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8)
                sim = n1 @ n2.T
                dist = 1 - float(np.mean(sim))
            row.append(dist)
        print(f"  {s1:>14} | " + " | ".join(f"{d:>8.4f}" for d in row))

    if "val" in qembs and "train" in qembs and "test" in qembs:
        train_centroid = np.mean(qembs["train"], axis=0)
        test_centroid = np.mean(qembs["test"], axis=0)
        val_embs = qembs["val"]
        v1 = val_embs / (np.linalg.norm(val_embs, axis=1, keepdims=True) + 1e-8)
        t1 = train_centroid / (np.linalg.norm(train_centroid) + 1e-8)
        t2 = test_centroid / (np.linalg.norm(test_centroid) + 1e-8)
        sim_train = v1 @ t1
        sim_test = v1 @ t2
        closer_to_test = int(np.sum(sim_test > sim_train))
        pct = closer_to_test / len(val_embs) * 100
        print(f"\n  Val queries closer to test centroid: {closer_to_test}/{len(val_embs)} ({pct:.0f}%)")

    # 4. Relevance Density
    print("\n--- 4. Relevance Density ---")
    print(f"  {'Split':>6} | {'Avg':>6} | {'Median':>6} | {'P75':>6} | "
          f"{'P90':>6} | {'Max':>6} | {'≥3(%)':>7}")
    print(f"  {'-' * (6+2+6+6+6+6+7+1)}")
    for s in splits:
        r = results[s]
        counts = list(r["qrels_counts"].values())
        print(f"  {s:>6} | {r['avg_rel']:>6.1f} | {r['median_rel']:>6.0f} | "
              f"{np.percentile(counts, 75):>6.0f} | {r['p90_rel']:>6.0f} | "
              f"{np.max(counts):>6} | {r['pct_ge3_rel']:>6.1f}%")


def main():
    for name, path in DATASETS.items():
        doc_ids, doc_texts = load_corpus(path)
        results = measure_all(path, doc_ids, doc_texts)
        report(results, name)


if __name__ == "__main__":
    main()
