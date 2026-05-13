"""Compute MRR/Recall/Hit at k=10,100 for all predictions files.

Usage:
  python src/analysis/dump_predictions_metrics.py              # all files
  python src/analysis/dump_predictions_metrics.py --path <file> # single file
"""
import json
import argparse
from pathlib import Path

PRED_DIR = "outputs/predictions"


def score_file(path):
    m10, r10, h10 = [], [], 0
    m100, r100, h100 = [], [], 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        gt = set(d["ground_truth"])
        ranked = d["ranked_doc_ids"]
        if not gt:
            continue

        h10f = [i + 1 for i, did in enumerate(ranked[:10]) if did in gt]
        h100f = [i + 1 for i, did in enumerate(ranked[:100]) if did in gt]

        m10.append(1 / h10f[0] if h10f else 0)
        m100.append(1 / h100f[0] if h100f else 0)
        r10.append(len(h10f) / len(gt))
        r100.append(len(h100f) / len(gt))
        if h10f:
            h10 += 1
        if h100f:
            h100 += 1

    n = len(m10) or 1
    return (
        n,
        sum(m10) / n, sum(r10) / n, h10 / n,
        sum(m100) / n, sum(r100) / n, h100 / n,
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
    args = parser.parse_args()

    if args.path:
        files = [args.path]
    else:
        pred_dir = Path(PRED_DIR)
        if not pred_dir.exists():
            print(f"No predictions directory: {PRED_DIR}")
            return
        files = sorted(pred_dir.glob("*.jsonl"))
        if not files:
            print(f"No prediction files found in {PRED_DIR}/")
            return

    cols = ["dataset", "method", "n", "MRR@10", "R@10", "Hit10",
            "MRR@100", "R@100", "Hit100"]
    fmt = "{dataset:<28} {method:<14} {n:>5} {m10:>8.4f} {r10:>8.4f} {h10:>6.3f} {m100:>8.4f} {r100:>8.4f} {h100:>6.3f}"
    sep = "-" * (28 + 14 + 5 + 8 * 4 + 7 + 6 * 2 + 5)

    print(f"{'dataset':<28} {'method':<14} {'n':>5} "
          f"{'MRR@10':>8} {'R@10':>8} {'Hit10':>7} "
          f"{'MRR@100':>8} {'R@100':>8} {'Hit100':>7}")
    print(sep)

    for fp in files:
        try:
            method, dataset = parse_name(fp)
            n, m10, r10, h10, m100, r100, h100 = score_file(str(fp))
            print(fmt.format(dataset=dataset, method=method, n=n,
                             m10=m10, r10=r10, h10=h10,
                             m100=m100, r100=r100, h100=h100))
        except Exception as e:
            print(f"  ERROR {fp.name}: {e}")


if __name__ == "__main__":
    main()
