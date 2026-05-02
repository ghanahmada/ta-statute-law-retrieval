"""Compute inter-annotator agreement between LLM and legal expert.

Reads the completed annotation CSV and reports:
- Cohen's kappa
- Overall agreement rate
- False positive rate (LLM says RELEVANT, expert says NOT RELEVANT)
- False negative rate (LLM says NOT RELEVANT, expert says RELEVANT)
- Confusion matrix
- Per-book breakdown
"""

import argparse
import csv
from collections import defaultdict


def cohens_kappa(y1: list[int], y2: list[int]) -> float:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 0.0

    tp = sum(a == 1 and b == 1 for a, b in zip(y1, y2))
    tn = sum(a == 0 and b == 0 for a, b in zip(y1, y2))
    fp = sum(a == 1 and b == 0 for a, b in zip(y1, y2))
    fn = sum(a == 0 and b == 1 for a, b in zip(y1, y2))

    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def verdict_to_binary(v: str) -> int | None:
    v = v.strip().upper()
    if v in ("RELEVAN", "1", "YES", "Y", "R"):
        return 1
    if v in ("TIDAK_RELEVAN", "TIDAK RELEVAN", "0", "NO", "N", "NR"):
        return 0
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute LLM-expert agreement")
    parser.add_argument("--input", default="data/annotation_study/annotation_pairs.csv")
    args = parser.parse_args()

    llm_labels = []
    expert_labels = []
    books = []
    skipped = 0

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm = verdict_to_binary(row["llm_verdict"])
            expert = verdict_to_binary(row["expert_verdict"])
            if llm is None or expert is None:
                skipped += 1
                continue
            llm_labels.append(llm)
            expert_labels.append(expert)
            books.append(row["kuhperdata_book"])

    n = len(llm_labels)
    if skipped:
        print(f"Skipped {skipped} rows with missing/invalid verdicts")
    print(f"Evaluated {n} pairs\n")

    if n == 0:
        print("No valid pairs to evaluate.")
        return

    tp = sum(l == 1 and e == 1 for l, e in zip(llm_labels, expert_labels))
    tn = sum(l == 0 and e == 0 for l, e in zip(llm_labels, expert_labels))
    fp = sum(l == 1 and e == 0 for l, e in zip(llm_labels, expert_labels))
    fn = sum(l == 0 and e == 1 for l, e in zip(llm_labels, expert_labels))

    kappa = cohens_kappa(llm_labels, expert_labels)
    agreement = (tp + tn) / n
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    fnr = fn / (fn + tn) if (fn + tn) > 0 else 0.0

    print("=" * 50)
    print("INTER-ANNOTATOR AGREEMENT: LLM vs Expert")
    print("=" * 50)
    print(f"Cohen's kappa:      {kappa:.3f}")
    print(f"Agreement rate:     {agreement:.1%} ({tp + tn}/{n})")
    print()
    print(f"False positive rate: {fpr:.1%} ({fp}/{fp + tp})")
    print(f"  (LLM says RELEVANT, expert says NOT RELEVANT)")
    print(f"False negative rate: {fnr:.1%} ({fn}/{fn + tn})")
    print(f"  (LLM says NOT RELEVANT, expert says RELEVANT)")
    print()
    print("Confusion Matrix (LLM \\ Expert):")
    print(f"                  Expert-REL  Expert-NOTREL")
    print(f"  LLM-REL           {tp:>4}        {fp:>4}")
    print(f"  LLM-NOTREL        {fn:>4}        {tn:>4}")

    kappa_interp = (
        "Almost perfect" if kappa >= 0.81 else
        "Substantial" if kappa >= 0.61 else
        "Moderate" if kappa >= 0.41 else
        "Fair" if kappa >= 0.21 else
        "Slight" if kappa >= 0.0 else
        "Poor"
    )
    print(f"\nInterpretation: {kappa_interp} agreement (Landis & Koch, 1977)")

    by_book = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for l, e, b in zip(llm_labels, expert_labels, books):
        if l == 1 and e == 1:
            by_book[b]["tp"] += 1
        elif l == 0 and e == 0:
            by_book[b]["tn"] += 1
        elif l == 1 and e == 0:
            by_book[b]["fp"] += 1
        else:
            by_book[b]["fn"] += 1

    print("\n" + "=" * 50)
    print("PER-BOOK BREAKDOWN")
    print("=" * 50)
    for book in sorted(by_book):
        d = by_book[book]
        total = d["tp"] + d["tn"] + d["fp"] + d["fn"]
        agree = d["tp"] + d["tn"]
        bl = [1 if k in ("tp", "fp") else 0 for k in ("tp", "fp", "fn", "tn") for _ in range(d[k])]
        be = [1 if k in ("tp", "fn") else 0 for k in ("tp", "fp", "fn", "tn") for _ in range(d[k])]
        bk = cohens_kappa(bl, be) if total >= 5 else float("nan")
        print(f"  {book}: {agree}/{total} agree, κ={bk:.2f}, FP={d['fp']}, FN={d['fn']}")


if __name__ == "__main__":
    main()
