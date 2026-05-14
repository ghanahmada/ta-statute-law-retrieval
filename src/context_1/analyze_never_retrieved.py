"""Show which docs are most frequently relevant but never retrieved.

Usage:
  python src/context_1/analyze_never_retrieved.py \
    --logs outputs/context_1/kuhperdata-exp_structgnn/agent_log.jsonl \
           outputs/context_1/kuhperdata-exp_flat/agent_log.jsonl \
    --names structgnn dense_flat \
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
    parser.add_argument("--logs", nargs="+", required=True,
                        help="agent_log.jsonl files to compare")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each log (default: filename)")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top problematic docs to show")
    args = parser.parse_args()

    names = args.names or [p.split("/")[-2] for p in args.logs]
    counters = [never_retrieved(p) for p in args.logs]

    all_docs = set()
    for c in counters:
        all_docs |= set(c)

    all_docs = sorted(all_docs, key=lambda d: sum(c.get(d, 0) for c in counters), reverse=True)

    col_w = 12
    name_w = 12
    header = "%-*s" % (col_w, "doc_id")
    for name in names:
        header += " %*s" % (name_w, name[:name_w])
    header += " %*s" % (8, "total")
    print(header)
    print("-" * len(header))

    for doc in all_docs[:args.top]:
        row = "%-*s" % (col_w, doc)
        total = 0
        for c in counters:
            v = c.get(doc, 0)
            row += " %*d" % (name_w, v)
            total += v
        row += " %*d" % (8, total)
        print(row)


if __name__ == "__main__":
    main()
