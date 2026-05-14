"""Show which docs are most frequently relevant but never retrieved.

Usage:
  python src/context_1/analyze_never_retrieved.py \
    --logs outputs/context_1/kuhperdata-exp_structgnn/agent_log.jsonl \
           outputs/context_1/kuhperdata-exp_flat/agent_log.jsonl \
    --names structgnn_agent dense_flat \
    --preds outputs/predictions/structgnn_kuhperdata-exp.jsonl \
    --pred_names structgnn_solo \
    --top 25
"""

import argparse
import json
from collections import Counter


def never_retrieved(path: str) -> Counter:
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen = set(r.get("ranked_seen_100", r["ranked_doc_ids"]))
            for doc in r.get("ground_truth", []):
                if doc not in seen:
                    c[doc] += 1
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=[],
                        help="agent_log.jsonl files to compare")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each log")
    parser.add_argument("--preds", nargs="+", default=[],
                        help="predictions JSONL files (outputs/predictions/*.jsonl)")
    parser.add_argument("--pred_names", nargs="+", default=None,
                        help="Display names for each predictions file")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top problematic docs to show")
    parser.add_argument("--sort_by", choices=["total", "delta"], default="total",
                        help="Sort by total failures or by delta (log[0] - log[1])")
    args = parser.parse_args()

    if not args.logs and not args.preds:
        parser.error("Provide at least one --logs or --preds file.")

    log_names = args.names or [p.split("/")[-2] for p in args.logs]
    pred_names = args.pred_names or [p.split("/")[-1].replace(".jsonl", "") for p in args.preds]
    names = log_names + pred_names
    counters = [never_retrieved(p) for p in args.logs + args.preds]

    all_docs = set()
    for c in counters:
        all_docs |= set(c)

    if args.sort_by == "delta" and len(counters) >= 2:
        all_docs = sorted(all_docs, key=lambda d: counters[0].get(d, 0) - counters[1].get(d, 0), reverse=True)
    else:
        all_docs = sorted(all_docs, key=lambda d: sum(c.get(d, 0) for c in counters), reverse=True)

    col_w = 12
    name_w = 12
    header = "%-*s" % (col_w, "doc_id")
    for name in names:
        header += " %*s" % (name_w, name[:name_w])
    header += " %*s" % (8, "total")
    if len(counters) >= 2:
        header += " %*s" % (8, "delta")
    print(header)
    print("-" * len(header))

    for doc in all_docs[:args.top]:
        row = "%-*s" % (col_w, doc)
        total = 0
        vals = []
        for c in counters:
            v = c.get(doc, 0)
            row += " %*d" % (name_w, v)
            total += v
            vals.append(v)
        row += " %*d" % (8, total)
        if len(counters) >= 2:
            row += " %*d" % (8, vals[0] - vals[1])
        print(row)


if __name__ == "__main__":
    main()
