"""Analyze training data distribution: hub vs non-hub positive documents.

Usage:
  python src/analysis/training_distribution.py --dataset kuhperdata-humanized
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paragnn import DATASETS
from util.dataloader import DataLoader

HUB_ARTICLES = {"1365", "1865", "1320", "1337", "1234", "188"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kuhperdata-humanized")
    parser.add_argument("--max_relevant", type=int, default=5)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    train_loader = DataLoader(
        f"{cfg['path']}/corpus.jsonl",
        f"{cfg['path']}/queries.jsonl",
        f"{cfg['path']}/qrels_train.tsv",
    ).load()
    test_loader = DataLoader(
        f"{cfg['path']}/corpus.jsonl",
        f"{cfg['path']}/queries.jsonl",
        f"{cfg['path']}/qrels_test.tsv",
    ).load()

    if args.max_relevant > 0:
        train_loader.filter_max_relevant(args.max_relevant)
        test_loader.filter_max_relevant(args.max_relevant)

    print(f"\n{'='*60}")
    print(f"  Training Distribution: {args.dataset}")
    print(f"{'='*60}")

    # Count training pairs
    hub_pairs = 0
    nonhub_pairs = 0
    hub_doc_counter = Counter()
    nonhub_doc_counter = Counter()
    queries_with_hub = set()
    queries_with_nonhub_only = set()

    for qid, rels in train_loader.qrels.items():
        relevant = [d for d, s in rels.items() if s > 0]
        has_hub = False
        has_nonhub = False
        for did in relevant:
            if did in HUB_ARTICLES:
                hub_pairs += 1
                hub_doc_counter[did] += 1
                has_hub = True
            else:
                nonhub_pairs += 1
                nonhub_doc_counter[did] += 1
                has_nonhub = True
        if has_hub:
            queries_with_hub.add(qid)
        if not has_hub and has_nonhub:
            queries_with_nonhub_only.add(qid)

    total_pairs = hub_pairs + nonhub_pairs
    print(f"\n  Training pairs:")
    print(f"    Hub positive:     {hub_pairs:>5} ({hub_pairs/total_pairs:.1%})")
    print(f"    Non-hub positive: {nonhub_pairs:>5} ({nonhub_pairs/total_pairs:.1%})")
    print(f"    Total:            {total_pairs:>5}")

    print(f"\n  Training queries:")
    print(f"    With hub GT:          {len(queries_with_hub):>5}")
    print(f"    Non-hub only GT:      {len(queries_with_nonhub_only):>5}")
    print(f"    Total:                {len(train_loader.qrels):>5}")

    print(f"\n  Hub article frequency as positive (training):")
    for did, count in sorted(hub_doc_counter.items(), key=lambda x: -x[1]):
        print(f"    Pasal {did}: {count} pairs")

    print(f"\n  Non-hub positive doc distribution:")
    counts = list(nonhub_doc_counter.values())
    print(f"    Unique non-hub docs: {len(counts)}")
    print(f"    Mean appearances:    {sum(counts)/len(counts):.1f}")
    print(f"    Max appearances:     {max(counts)} (Pasal {max(nonhub_doc_counter, key=nonhub_doc_counter.get)})")
    print(f"    Docs with 1 pair:    {sum(1 for c in counts if c == 1)}")
    print(f"    Docs with 2-5 pairs: {sum(1 for c in counts if 2 <= c <= 5)}")
    print(f"    Docs with >5 pairs:  {sum(1 for c in counts if c > 5)}")

    # Same for test set
    print(f"\n  {'='*60}")
    print(f"  Test Distribution")
    print(f"  {'='*60}")

    hub_test = 0
    nonhub_test = 0
    test_hub_queries = set()
    test_nonhub_only = set()

    for qid, rels in test_loader.qrels.items():
        relevant = [d for d, s in rels.items() if s > 0]
        has_hub = False
        for did in relevant:
            if did in HUB_ARTICLES:
                hub_test += 1
                has_hub = True
            else:
                nonhub_test += 1
        if has_hub:
            test_hub_queries.add(qid)
        else:
            test_nonhub_only.add(qid)

    total_test = hub_test + nonhub_test
    print(f"\n  Test pairs:")
    print(f"    Hub positive:     {hub_test:>5} ({hub_test/total_test:.1%})")
    print(f"    Non-hub positive: {nonhub_test:>5} ({nonhub_test/total_test:.1%})")
    print(f"\n  Test queries:")
    print(f"    With hub GT:      {len(test_hub_queries):>5}")
    print(f"    Non-hub only GT:  {len(test_nonhub_only):>5}")

    # How many training examples does the model see per hub article per epoch?
    print(f"\n  {'='*60}")
    print(f"  Training Signal Per Article")
    print(f"  {'='*60}")
    print(f"\n  Each hub article appears as positive in a batch this many times per epoch:")
    print(f"  (with batch_size=256, {total_pairs} total pairs, ~{total_pairs//256} batches)")
    for did, count in sorted(hub_doc_counter.items(), key=lambda x: -x[1]):
        print(f"    Pasal {did}: {count} times → in ~{count * 256 / total_pairs:.0f}% of batches")

    print(f"\n  Average non-hub article appears {sum(counts)/len(counts):.1f} times per epoch")
    print(f"  → in ~{sum(counts)/len(counts) * 256 / total_pairs:.1f}% of batches")


if __name__ == "__main__":
    main()
