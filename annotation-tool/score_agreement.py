"""Compute inter-annotator agreement (Cohen's Kappa) from the annotation database.

Usage:
  python score_agreement.py                # use local DB
  railway run python score_agreement.py    # use Railway DB

Reports:
  - Pairwise Cohen's Kappa between all annotator pairs
  - Fleiss' Kappa (if 3+ annotators)
  - Each annotator vs LLM label agreement
  - Label distribution per annotator
"""

from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from app.database import SessionLocal
from app.models import Annotator, Label, Pair


def main():
    db = SessionLocal()

    annotators = [a.name for a in db.query(Annotator).all()]
    pairs = {p.pair_id: p.llm_label for p in db.query(Pair).order_by(Pair.pair_id).all()}
    pair_ids = sorted(pairs.keys())

    labels_by_ann: dict[str, dict[str, str]] = {}
    for ann in annotators:
        rows = db.query(Label).filter(Label.annotator == ann).all()
        labels_by_ann[ann] = {r.pair_id: r.label for r in rows}

    db.close()

    # Normalize labels
    def norm(lbl: str) -> str:
        lbl = lbl.upper().strip()
        if lbl in ("RELEVAN", "RELEVANT"):
            return "RELEVANT"
        if lbl in ("TIDAK_RELEVAN", "NOT_RELEVANT"):
            return "NOT_RELEVANT"
        return lbl

    print("=" * 60)
    print("ANNOTATION AGREEMENT REPORT")
    print("=" * 60)

    # Label counts
    print("\n--- Label Distribution ---")
    for ann in annotators:
        ann_labels = labels_by_ann.get(ann, {})
        total = len(ann_labels)
        rel = sum(1 for v in ann_labels.values() if norm(v) == "RELEVANT")
        nrel = total - rel
        print(f"  {ann}: {total}/{len(pair_ids)} labeled "
              f"({rel} RELEVANT, {nrel} NOT_RELEVANT)")

    # Pairwise Cohen's Kappa
    print("\n--- Pairwise Cohen's Kappa ---")
    ann_with_labels = [a for a in annotators if len(labels_by_ann.get(a, {})) > 0]

    if len(ann_with_labels) < 2:
        print("  Need at least 2 annotators with labels to compute kappa.")
    else:
        kappas = []
        for a1, a2 in combinations(ann_with_labels, 2):
            common = [pid for pid in pair_ids
                      if pid in labels_by_ann[a1] and pid in labels_by_ann[a2]]
            if len(common) < 2:
                print(f"  {a1} vs {a2}: insufficient overlap ({len(common)} pairs)")
                continue
            y1 = [norm(labels_by_ann[a1][pid]) for pid in common]
            y2 = [norm(labels_by_ann[a2][pid]) for pid in common]
            k = cohen_kappa_score(y1, y2)
            agree = sum(1 for a, b in zip(y1, y2) if a == b)
            kappas.append(k)
            print(f"  {a1} vs {a2}: κ = {k:.4f}  "
                  f"(agreement: {agree}/{len(common)} = {agree/len(common):.1%})")

        if kappas:
            print(f"\n  Average pairwise κ = {sum(kappas)/len(kappas):.4f}")

    # Annotator vs LLM
    print("\n--- Annotator vs LLM Label ---")
    for ann in ann_with_labels:
        common = [pid for pid in pair_ids
                  if pid in labels_by_ann[ann] and pairs[pid]]
        if len(common) < 2:
            print(f"  {ann} vs LLM: insufficient data")
            continue
        y_ann = [norm(labels_by_ann[ann][pid]) for pid in common]
        y_llm = [norm(pairs[pid]) for pid in common]
        k = cohen_kappa_score(y_ann, y_llm)
        agree = sum(1 for a, b in zip(y_ann, y_llm) if a == b)
        print(f"  {ann} vs LLM: κ = {k:.4f}  "
              f"(agreement: {agree}/{len(common)} = {agree/len(common):.1%})")

    # Interpretation guide
    print("\n--- Interpretation ---")
    print("  κ < 0.20  Poor")
    print("  0.20-0.40 Fair")
    print("  0.40-0.60 Moderate")
    print("  0.60-0.80 Substantial")
    print("  0.80-1.00 Almost perfect")
    print("=" * 60)


if __name__ == "__main__":
    main()
