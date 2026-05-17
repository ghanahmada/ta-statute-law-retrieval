"""Enrich KUHPerdata corpus.jsonl titles with hierarchy from raw statute CSV.

Rewrites titles from "Pasal X" to "Buku Kesatu Orang, Bab I Menikmati dan
Kehilangan Hak Kewargaan, Pasal 1" — giving StructGNN meaningful act grouping.

Usage:
  python src/scripts/enrich_kuhperdata_titles.py --csv data/kuhperdata/kuhperdata_statutes.csv
  python src/scripts/enrich_kuhperdata_titles.py --csv data/kuhperdata/kuhperdata_statutes.csv --dry_run

If the CSV doesn't exist locally, downloads it from HuggingFace.
"""
import argparse
import csv
import json
import os

DATASETS = [
    "data/kuhperdata-humanized",
    "data/kuhperdata-summarized",
    "data/kuhperdata-exp",
    "data/kuhperdata-summ-exp",
]

HF_DATASET_ID = "ghanahmada/kuhperdata"
HF_CSV_PATH = "statute/kuh_perdata.csv"


def title_case_id(s: str) -> str:
    """Convert 'KESATU' -> 'Kesatu', 'ORANG' -> 'Orang', etc."""
    return s.strip().title() if s.strip() else ""


def build_hierarchy_map(csv_path: str) -> dict[str, str]:
    """Parse raw statute CSV and build pasal_nomor -> hierarchical title mapping."""
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pasal_nomor = row["pasal_nomor"].strip()

            buku_label = title_case_id(row.get("buku_label", ""))
            buku_judul = title_case_id(row.get("buku_judul", ""))
            bab_label = row.get("bab_label", "").strip()
            bab_judul = title_case_id(row.get("bab_judul", ""))
            bagian_label = row.get("bagian_label", "").strip()
            bagian_judul = title_case_id(row.get("bagian_judul", ""))

            parts = []
            if buku_label:
                parts.append(f"Buku {buku_label} {buku_judul}".strip())
            if bab_label:
                bab_str = f"Bab {bab_label} {bab_judul}".strip()
                parts.append(bab_str)
            if bagian_label and bagian_judul:
                parts.append(f"Bagian {bagian_label} {bagian_judul}".strip())
            parts.append(f"Pasal {pasal_nomor}")

            mapping[pasal_nomor] = ", ".join(parts)

    return mapping


def enrich_corpus(corpus_path: str, hierarchy: dict[str, str], dry_run: bool = False) -> int:
    """Rewrite corpus.jsonl titles using hierarchy mapping. Returns count of updated docs."""
    if not os.path.exists(corpus_path):
        print(f"  SKIP (not found): {corpus_path}")
        return 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    updated = 0
    for doc in docs:
        doc_id = doc["_id"]
        if doc_id in hierarchy:
            new_title = hierarchy[doc_id]
            if doc["title"] != new_title:
                doc["title"] = new_title
                updated += 1

    if not dry_run and updated > 0:
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return updated


def main():
    parser = argparse.ArgumentParser(description="Enrich KUHPerdata corpus titles with hierarchy")
    parser.add_argument("--csv", default="data/statute/kuh_perdata.csv",
                        help="Path to raw KUHPerdata statute CSV with hierarchy columns")
    parser.add_argument("--dry_run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset directories to update")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found at {args.csv}, downloading from HuggingFace...")
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_DATASET_ID, filename=HF_CSV_PATH, repo_type="dataset"
        )
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        import shutil
        shutil.copy(local, args.csv)
        print(f"  Saved to {args.csv}")

    hierarchy = build_hierarchy_map(args.csv)
    print(f"Loaded hierarchy for {len(hierarchy)} articles")
    print(f"  Example: {hierarchy.get('1320', 'N/A')}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")

    for dataset_dir in args.datasets:
        corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
        updated = enrich_corpus(corpus_path, hierarchy, dry_run=args.dry_run)
        status = "would update" if args.dry_run else "updated"
        print(f"  {corpus_path}: {status} {updated}/{len(hierarchy)} titles")


if __name__ == "__main__":
    main()
