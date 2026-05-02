"""Compute inter-annotator agreement and LLM-vs-human agreement.

Data source: either a local CSV file or the annotation tool's API.

Usage:
  python compute_agreement.py --from-api http://localhost:8000
  python compute_agreement.py --input labels.csv
"""

import argparse
import csv
import io
import urllib.request
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
    if v in ("RELEVAN", "RELEVANT", "1", "YES", "Y", "R"):
        return 1
    if v in ("TIDAK_RELEVAN", "TIDAK RELEVAN", "NOT_RELEVANT", "NOT RELEVANT", "0", "NO", "N", "NR"):
        return 0
    return None


def load_from_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_from_api(base_url: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/admin/export"
    print(f"Fetching labels from {url} ...")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def analyze(rows: list[dict]):
    pair_labels: dict[str, dict] = defaultdict(dict)
    pair_meta: dict[str, dict] = {}
    annotators: set[str] = set()
    skipped = 0

    for row in rows:
        annotator = row["annotator"]
        pair_id = row["pair_id"]
        verdict = verdict_to_binary(row["label"])
        llm_verdict = verdict_to_binary(row.get("llm_label", ""))

        if verdict is None:
            skipped += 1
            continue

        annotators.add(annotator)
        pair_labels[pair_id][annotator] = verdict
        pair_labels[pair_id]["_llm"] = llm_verdict
        pair_meta[pair_id] = {
            "variant": row.get("variant", pair_id[0].lower()),
            "book": row.get("kuhperdata_book", ""),
        }

    n_pairs = len(pair_labels)
    annotators_list = sorted(annotators)
    print(f"Loaded {n_pairs} pairs from {len(annotators_list)} annotator(s)")
    if skipped:
        print(f"Skipped {skipped} invalid rows")
    print()

    # --- Single annotator: LLM vs annotator ---
    if len(annotators_list) == 1:
        ann = annotators_list[0]
        print("=" * 60)
        print(f"LLM vs {ann}")
        print("=" * 60)

        ann_y, llm_y = [], []
        by_variant = defaultdict(lambda: {"ann": [], "llm": []})

        for pair_id, labels in pair_labels.items():
            a = labels.get(ann)
            l = labels.get("_llm")
            if a is None or l is None:
                continue
            ann_y.append(a)
            llm_y.append(l)
            v = pair_meta[pair_id]["variant"]
            by_variant[v]["ann"].append(a)
            by_variant[v]["llm"].append(l)

        n = len(ann_y)
        if n == 0:
            print("No overlapping pairs.")
            return

        kappa = cohens_kappa(llm_y, ann_y)
        agree = sum(1 for a, l in zip(ann_y, llm_y) if a == l)

        tp = sum(1 for l, a in zip(llm_y, ann_y) if l == 1 and a == 1)
        tn = sum(1 for l, a in zip(llm_y, ann_y) if l == 0 and a == 0)
        fp = sum(1 for l, a in zip(llm_y, ann_y) if l == 1 and a == 0)
        fn = sum(1 for l, a in zip(llm_y, ann_y) if l == 0 and a == 1)

        print(f"Cohen's kappa (LLM <-> {ann}): {kappa:.3f}")
        print(f"Agreement: {agree}/{n} ({100*agree/n:.1f}%)")
        print()
        print(f"  LLM says RELEVANT, human says NOT: {fp} (FP rate: {100*fp/max(fp+tp,1):.1f}%)")
        print(f"  LLM says NOT, human says RELEVANT: {fn} (FN rate: {100*fn/max(fn+tn,1):.1f}%)")
        print()
        print("Confusion Matrix (LLM \\ Human):")
        print(f"                Human-REL  Human-NOTREL")
        print(f"  LLM-REL        {tp:>4}        {fp:>4}")
        print(f"  LLM-NOTREL     {fn:>4}        {tn:>4}")

        kappa_interp = (
            "Almost perfect" if kappa >= 0.81 else
            "Substantial" if kappa >= 0.61 else
            "Moderate" if kappa >= 0.41 else
            "Fair" if kappa >= 0.21 else
            "Slight" if kappa >= 0.0 else
            "Poor"
        )
        print(f"\nInterpretation: {kappa_interp} agreement (Landis & Koch, 1977)")

        if by_variant:
            print()
            print("BY VARIANT:")
            for v in sorted(by_variant):
                va, vl = by_variant[v]["ann"], by_variant[v]["llm"]
                vk = cohens_kappa(vl, va)
                vagree = sum(1 for a, l in zip(va, vl) if a == l)
                label = "HUMANIZED" if v in ("h", "humanized") else "SUMMARIZED" if v in ("s", "summarized") else v.upper()
                print(f"  {label:15} kappa={vk:.3f}  agree={vagree}/{len(va)} ({100*vagree/len(va):.1f}%)")
        return

    # --- Multiple annotators ---
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT (Human Annotators)")
    print("=" * 60)

    valid_pairs = []
    for pair_id, labels in pair_labels.items():
        y_list = [labels.get(ann) for ann in annotators_list]
        if any(y is None for y in y_list):
            continue
        valid_pairs.append((pair_id, y_list, labels.get("_llm")))

    if not valid_pairs:
        print("No pairs with labels from all annotators.")
        return

    n = len(valid_pairs)
    print(f"Pairs with all {len(annotators_list)} annotators: {n}")

    for i in range(len(annotators_list)):
        for j in range(i + 1, len(annotators_list)):
            yi = [vp[1][i] for vp in valid_pairs]
            yj = [vp[1][j] for vp in valid_pairs]
            k = cohens_kappa(yi, yj)
            agree = sum(1 for a, b in zip(yi, yj) if a == b)
            print(f"  kappa ({annotators_list[i]} <-> {annotators_list[j]}): {k:.3f}  agree={agree}/{n} ({100*agree/n:.1f}%)")

    if len(annotators_list) >= 2:
        all_kappas = []
        for i in range(len(annotators_list)):
            for j in range(i + 1, len(annotators_list)):
                yi = [vp[1][i] for vp in valid_pairs]
                yj = [vp[1][j] for vp in valid_pairs]
                all_kappas.append(cohens_kappa(yi, yj))
        print(f"  Average kappa: {sum(all_kappas)/len(all_kappas):.3f}")

    print()
    print("=" * 60)
    print("LLM vs HUMAN MAJORITY")
    print("=" * 60)

    valid_with_llm = [(pid, ys, llm) for pid, ys, llm in valid_pairs if llm is not None]
    if not valid_with_llm:
        print("No pairs with LLM labels.")
        return

    majority = []
    llm_y = []
    by_variant = defaultdict(lambda: {"majority": [], "llm": []})

    for pair_id, y_list, llm in valid_with_llm:
        vote = 1 if sum(y_list) > len(y_list) / 2 else 0
        majority.append(vote)
        llm_y.append(llm)
        v = pair_meta[pair_id]["variant"]
        by_variant[v]["majority"].append(vote)
        by_variant[v]["llm"].append(llm)

    kappa = cohens_kappa(llm_y, majority)
    agree = sum(1 for l, m in zip(llm_y, majority) if l == m)
    tp = sum(1 for l, m in zip(llm_y, majority) if l == 1 and m == 1)
    tn = sum(1 for l, m in zip(llm_y, majority) if l == 0 and m == 0)
    fp = sum(1 for l, m in zip(llm_y, majority) if l == 1 and m == 0)
    fn = sum(1 for l, m in zip(llm_y, majority) if l == 0 and m == 1)

    print(f"Cohen's kappa (LLM <-> Majority): {kappa:.3f}")
    print(f"Agreement: {agree}/{len(llm_y)} ({100*agree/len(llm_y):.1f}%)")
    print(f"  FP (LLM too lenient): {fp} ({100*fp/max(fp+tp,1):.1f}%)")
    print(f"  FN (LLM too strict):  {fn} ({100*fn/max(fn+tn,1):.1f}%)")

    if by_variant:
        print()
        print("BY VARIANT:")
        for v in sorted(by_variant):
            vm, vl = by_variant[v]["majority"], by_variant[v]["llm"]
            vk = cohens_kappa(vl, vm)
            vagree = sum(1 for a, l in zip(vm, vl) if a == l)
            label = "HUMANIZED" if v in ("h", "humanized") else "SUMMARIZED" if v in ("s", "summarized") else v.upper()
            print(f"  {label:15} kappa={vk:.3f}  agree={vagree}/{len(vm)} ({100*vagree/len(vm):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compute agreement from labels")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to local labels CSV file")
    source.add_argument("--from-api", help="Base URL of annotation tool (e.g. http://localhost:8000)")
    args = parser.parse_args()

    if args.from_api:
        rows = load_from_api(args.from_api)
    else:
        rows = load_from_csv(args.input)

    analyze(rows)


if __name__ == "__main__":
    main()
