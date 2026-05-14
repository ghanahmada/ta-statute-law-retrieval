"""Analyze per-tool-call statistics from agent_log.jsonl.

Produces:
  - Tool call frequency and hit rate per tool
  - Avg docs returned and latency per tool
  - First-hit-turn distribution (which turn first found a relevant doc)
  - Failed query (MRR=0) breakdown — tool sequences, miss reasons
  - Gap analysis: salient patterns suggesting new tool needs

Usage:
  python src/context_1/analyze_tool_calls.py \
    --log outputs/context_1/kuhperdata-exp/agent_log.jsonl \
    --dataset kuhperdata-exp

  python src/context_1/analyze_tool_calls.py \
    --log outputs/context_1/kuhperdata-exp/agent_log.jsonl \
    --qrels data/kuhperdata-exp/qrels_test.tsv
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.metrics import evaluate_ranking

DATASET_QRELS = {
    "kuhperdata-exp":      "data/kuhperdata-exp/qrels_test.tsv",
    "kuhperdata-summ-exp": "data/kuhperdata-summ-exp/qrels_test.tsv",
    "kuhperdata-humanized":"data/kuhperdata-humanized/qrels_test.tsv",
    "bsard":               "data/bsard/qrels_test.tsv",
    "coliee":              "data/coliee/qrels_test.tsv",
    "stard":               "data/stard/qrels_test.tsv",
}


def load_qrels(path: str) -> dict[str, list[str]]:
    gt = {}
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                gt.setdefault(parts[0], []).append(parts[1])
    return gt


def load_log(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _per_query_mrr(rankings: dict, gt: dict, k: int = 10) -> dict[str, float]:
    mrr = {}
    for qid, ranked in rankings.items():
        relevant = set(gt.get(qid, []))
        for rank, doc in enumerate(ranked[:k], 1):
            if doc in relevant:
                mrr[qid] = 1.0 / rank
                break
        else:
            mrr[qid] = 0.0
    return mrr


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--show_failed", type=int, default=10,
                        help="Number of failed queries to show detail for")
    args = parser.parse_args()

    qrels_path = args.qrels
    if not qrels_path and args.dataset:
        qrels_path = DATASET_QRELS.get(args.dataset)
    if not qrels_path:
        sys.exit("Provide --qrels or --dataset")

    gt = load_qrels(qrels_path)
    records = load_log(args.log)
    if not records:
        sys.exit("No records found in log")

    rankings = {r["qid"]: r["ranked_doc_ids"] for r in records}
    evaluated_gt = {qid: docs for qid, docs in gt.items() if qid in rankings}
    mrr_by_qid = _per_query_mrr(rankings, evaluated_gt, args.top_k)

    # ------------------------------------------------------------------ #
    section("OVERALL METRICS")
    metrics = evaluate_ranking(rankings, evaluated_gt, args.top_k)
    k = args.top_k
    print(f"  Queries evaluated : {metrics['n_queries']}")
    print(f"  MRR@{k}            : {metrics[f'mrr@{k}']:.4f}")
    print(f"  Recall@{k}         : {metrics[f'recall@{k}']:.4f}")
    print(f"  Hit Rate          : {metrics['hit_rate']:.4f}")

    # ------------------------------------------------------------------ #
    section("AGENT BEHAVIOUR SUMMARY")
    turns        = [r["turns"]       for r in records]
    n_selected   = [r["n_selected"]  for r in records]
    n_seen       = [r["n_seen"]      for r in records]
    n_read       = [r.get("n_read",0)for r in records]
    elapsed      = [r["elapsed_s"]   for r in records]
    gate_hits    = sum(r.get("n_gate_triggers",0)       for r in records)
    sim_rejs     = sum(r.get("n_similarity_rejections",0) for r in records)
    errors       = [r for r in records if r.get("error")]

    print(f"  Avg turns/query      : {np.mean(turns):.2f}  (max {max(turns)})")
    print(f"  Avg selected docs    : {np.mean(n_selected):.2f}")
    print(f"  Avg seen docs        : {np.mean(n_seen):.2f}")
    print(f"  Avg read docs        : {np.mean(n_read):.2f}")
    print(f"  Avg time/query (s)   : {np.mean(elapsed):.2f}")
    print(f"  Total gate triggers  : {gate_hits}")
    print(f"  Total sim rejections : {sim_rejs}")
    print(f"  Errors               : {len(errors)}")

    # ------------------------------------------------------------------ #
    # Collect per-call stats across all records
    all_calls = []
    for r in records:
        for call in r.get("tool_call_log", []):
            all_calls.append({**call, "_qid": r["qid"]})

    if not all_calls:
        print("\n  No tool_call_log found in records.")
        print("  Re-run evaluate_context1.py with updated code to collect per-call stats.")
        return

    section("PER-TOOL STATISTICS")
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for c in all_calls:
        by_tool[c["tool"]].append(c)

    header = f"  {'Tool':<20} {'Calls':>6} {'Hit%':>6} {'AvgDocs':>8} {'AvgMs':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tool, calls in sorted(by_tool.items()):
        n = len(calls)
        hit_pct = 100 * sum(1 for c in calls if c.get("hit_relevant")) / n
        avg_docs = np.mean([c["n_docs_returned"] for c in calls])
        avg_ms   = np.mean([c["elapsed_s"] for c in calls]) * 1000
        print(f"  {tool:<20} {n:>6} {hit_pct:>5.1f}% {avg_docs:>8.1f} {avg_ms:>7.0f}ms")

    # ------------------------------------------------------------------ #
    section("FIRST-HIT TURN DISTRIBUTION")
    # For each query, which turn first returned a relevant doc?
    first_hit_turns = []
    never_hit = 0
    for r in records:
        qid = r["qid"]
        gt_set = set(gt.get(qid, []))
        found_turn = None
        for call in r.get("tool_call_log", []):
            if set(call["doc_ids_returned"]) & gt_set:
                found_turn = call["turn"]
                break
        if found_turn is not None:
            first_hit_turns.append(found_turn)
        else:
            never_hit += 1

    if first_hit_turns:
        turn_counts = Counter(first_hit_turns)
        print(f"  Never found relevant doc : {never_hit} queries ({100*never_hit/len(records):.1f}%)")
        print(f"  Found at turn 0 (bootstrap): {turn_counts.get(0,0)}")
        for t in sorted(turn_counts):
            if t > 0:
                print(f"  Found at turn {t:<2}         : {turn_counts[t]}")
        print(f"  Avg first-hit turn       : {np.mean(first_hit_turns):.2f}")

    # ------------------------------------------------------------------ #
    section("TOOL CALL SEQUENCE PATTERNS")
    # Most common tool sequences (full trajectory per query)
    seq_counter: Counter = Counter()
    for r in records:
        seq = tuple(c["tool"] for c in r.get("tool_call_log", []))
        seq_counter[seq] += 1
    print("  Top 10 sequences:")
    for seq, cnt in seq_counter.most_common(10):
        print(f"    {cnt:>3}x  {' → '.join(seq)}")

    # ------------------------------------------------------------------ #
    section("FAILED QUERIES (MRR=0)")
    failed_qids = [qid for qid, m in mrr_by_qid.items() if m == 0.0]
    print(f"  {len(failed_qids)} / {len(mrr_by_qid)} queries failed (MRR=0)")

    if failed_qids:
        records_by_qid = {r["qid"]: r for r in records}

        # Aggregate: did ANY tool call in failed queries hit a relevant doc?
        failed_never_retrieved = 0
        failed_retrieved_not_selected = 0
        for qid in failed_qids:
            r = records_by_qid.get(qid, {})
            gt_set = set(gt.get(qid, []))
            any_hit = any(
                set(c["doc_ids_returned"]) & gt_set
                for c in r.get("tool_call_log", [])
            )
            if not any_hit:
                failed_never_retrieved += 1
            else:
                failed_retrieved_not_selected += 1

        print(f"\n  Of failed queries:")
        print(f"    Never retrieved relevant doc : {failed_never_retrieved} "
              f"({100*failed_never_retrieved/len(failed_qids):.1f}%)")
        print(f"    Retrieved but not selected   : {failed_retrieved_not_selected} "
              f"({100*failed_retrieved_not_selected/len(failed_qids):.1f}%)")

        # Detail for first N failed queries
        print(f"\n  Detail (first {min(args.show_failed, len(failed_qids))}):")
        for qid in failed_qids[:args.show_failed]:
            r = records_by_qid.get(qid, {})
            gt_docs_list = gt.get(qid, [])
            gt_set = set(gt_docs_list)
            calls = r.get("tool_call_log", [])
            seq   = " → ".join(c["tool"] for c in calls)
            hits  = [f"t{c['turn']}:{c['tool']}" for c in calls
                     if set(c["doc_ids_returned"]) & gt_set]
            turns = r.get("turns", "?")
            err   = r.get("error", "")
            print(f"\n    qid={qid}  turns={turns}  gt={gt_docs_list}")
            print(f"      sequence : {seq or '(none)'}")
            print(f"      hits     : {hits or 'none'}")
            if err:
                print(f"      error    : {err}")

    # ------------------------------------------------------------------ #
    section("GAP ANALYSIS — TOOL IMPROVEMENT SUGGESTIONS")

    total_q = len(records)
    frac_never = never_hit / total_q if total_q else 0

    # Search hit rate
    search_calls = by_tool.get("search_corpus", [])
    grep_calls   = by_tool.get("grep_corpus",   [])
    read_calls   = by_tool.get("read_document",  [])

    search_hit_pct = (100 * sum(1 for c in search_calls if c.get("hit_relevant")) / len(search_calls)
                      if search_calls else 0)
    grep_hit_pct   = (100 * sum(1 for c in grep_calls if c.get("hit_relevant")) / len(grep_calls)
                      if grep_calls else 0)

    suggestions = []

    if frac_never > 0.15:
        suggestions.append(
            f"  ● {frac_never:.0%} of queries never retrieved a relevant doc via any tool.\n"
            f"    → Consider: query expansion tool (rewrite query using corpus vocab),\n"
            f"      or legal-concept mapping tool to bridge terminology gaps."
        )

    if failed_retrieved_not_selected > 0 and failed_never_retrieved < failed_retrieved_not_selected:
        suggestions.append(
            f"  ● {failed_retrieved_not_selected} failed queries found relevant docs but didn't select them.\n"
            f"    → Agent retrieval ≠ agent selection gap. Consider:\n"
            f"      (1) Strengthening read_document signal so agent recognises relevance after reading.\n"
            f"      (2) A 'compare_documents' tool to help agent distinguish near-miss candidates."
        )

    if search_hit_pct < 40:
        suggestions.append(
            f"  ● search_corpus hit rate = {search_hit_pct:.1f}% — retrieval quality is low.\n"
            f"    → Consider: multi-query search (generate N query variants, union results),\n"
            f"      or article-number lookup tool for structured statute corpora."
        )

    if grep_hit_pct > search_hit_pct + 15 and grep_calls:
        suggestions.append(
            f"  ● grep_corpus hit rate ({grep_hit_pct:.1f}%) > search_corpus ({search_hit_pct:.1f}%).\n"
            f"    → Grep is outperforming semantic search for these queries.\n"
            f"      Consider: exact-term lookup tool or BM25-only fallback tool."
        )

    if sim_rejs > len(records) * 0.5:
        suggestions.append(
            f"  ● High similarity rejections ({sim_rejs} total, "
            f"{sim_rejs/len(records):.1f}/query avg).\n"
            f"    → Agent struggles to diversify queries. Consider:\n"
            f"      a 'suggest_query_variants' tool that generates semantically distinct reformulations."
        )

    avg_read = np.mean(n_read) if n_read else 0
    if avg_read < 0.5 and read_calls:
        suggestions.append(
            f"  ● Agent rarely uses read_document (avg {avg_read:.2f}/query).\n"
            f"    → Consider surfacing more document text in search results, or\n"
            f"      a 'browse_article' tool that returns structured article sections."
        )

    if suggestions:
        for s in suggestions:
            print(s)
    else:
        print("  No major gaps detected from current patterns.")


if __name__ == "__main__":
    main()
