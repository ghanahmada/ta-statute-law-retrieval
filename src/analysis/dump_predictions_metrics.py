"""Compute MRR/Recall/Precision/Hit at k=10,100 for all predictions files.

Usage:
  python src/analysis/dump_predictions_metrics.py              # all files
  python src/analysis/dump_predictions_metrics.py --path <file> # single file
  python src/analysis/dump_predictions_metrics.py --pred_dir outputs/predictions

Files without 'ground_truth' field (e.g. context1 outputs) are handled by
loading qrels from the dataset directory inferred from the filename.
"""
import json
import argparse
import csv
from pathlib import Path

PRED_DIR = "outputs/predictions"
DATA_DIR = "data"


def load_qrels(dataset_name):
    qrels_path = Path(DATA_DIR) / dataset_name / "qrels_test.tsv"
    if not qrels_path.exists():
        return None
    qrels = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row.get("query_id") or row.get("query-id")
            did = row.get("doc_id") or row.get("corpus-id")
            if not qid or not did:
                continue
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels


def infer_dataset(stem):
    for ds in ["kuhperdata-exp", "kuhperdata-summ-exp", "bsard", "ilpcsr", "stard", "coliee"]:
        if ds in stem:
            return ds
    return None


def score_file(path, fallback_qrels=None):
    m10, r10, p10, h10 = [], [], [], 0
    m100, r100, p100, h100 = [], [], [], 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if "ground_truth" in d:
            gt = set(d["ground_truth"])
        elif fallback_qrels and d.get("qid") in fallback_qrels:
            gt = set(fallback_qrels[d["qid"]])
        else:
            continue
        ranked = d["ranked_doc_ids"]
        if not gt:
            continue

        h10f = [i + 1 for i, did in enumerate(ranked[:10]) if did in gt]
        h100f = [i + 1 for i, did in enumerate(ranked[:100]) if did in gt]

        m10.append(1 / h10f[0] if h10f else 0)
        m100.append(1 / h100f[0] if h100f else 0)
        r10.append(len(h10f) / len(gt))
        r100.append(len(h100f) / len(gt))
        p10.append(len(h10f) / 10)
        p100.append(len(h100f) / 100)
        if h10f:
            h10 += 1
        if h100f:
            h100 += 1

    n = len(m10) or 1
    return (
        n,
        sum(m10) / n, sum(r10) / n, sum(p10) / n, h10 / n,
        sum(m100) / n, sum(r100) / n, sum(p100) / n, h100 / n,
    )


def parse_name(fp):
    stem = Path(fp).stem
    if "_" not in stem:
        return "?", stem
    parts = stem.split("_", 1)
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Dump retrieval metrics for predictions files"
    )
    parser.add_argument("--path", type=str, default=None,
                        help="Path to a single predictions JSONL")
    parser.add_argument("--pred_dir", type=str, default=PRED_DIR,
                        help="Predictions directory (default: outputs/predictions)")
    args = parser.parse_args()

    if args.path:
        files = [Path(args.path)]
    else:
        pred_dir = Path(args.pred_dir)
        if not pred_dir.exists():
            print(f"No predictions directory: {pred_dir}")
            return
        files = sorted(pred_dir.glob("*.jsonl"))
        if not files:
            print(f"No prediction files found in {pred_dir}/")
            return

    fmt = (
        "{dataset:<28} {method:<16} {n:>5}"
        " {r10:>8.4f} {m10:>8.4f} {p10:>8.4f} {h10:>6.3f}"
        " {r100:>8.4f} {m100:>8.4f} {p100:>8.4f} {h100:>6.3f}"
    )
    hdr = (
        f"{'dataset':<28} {'method':<16} {'n':>5}"
        f" {'R@10':>8} {'MRR@10':>8} {'P@10':>8} {'Hit10':>7}"
        f" {'R@100':>8} {'MRR@100':>8} {'P@100':>8} {'Hit100':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    qrels_cache = {}
    for fp in files:
        try:
            method, dataset = parse_name(fp)
            ds_name = infer_dataset(Path(fp).stem)
            fallback = None
            if ds_name:
                if ds_name not in qrels_cache:
                    qrels_cache[ds_name] = load_qrels(ds_name)
                fallback = qrels_cache[ds_name]
            n, m10, r10, p10, h10, m100, r100, p100, h100 = score_file(str(fp), fallback)
            print(fmt.format(dataset=dataset, method=method, n=n,
                             m10=m10, r10=r10, p10=p10, h10=h10,
                             m100=m100, r100=r100, p100=p100, h100=h100))
        except Exception as e:
            print(f"  ERROR {fp.name}: {e}")


if __name__ == "__main__":
    main()
