"""Download COLIEE dataset from HuggingFace and convert to BEIR format.

Downloads the 'coliee' config from ghanahmada/kuhperdata and writes:
  data/coliee/corpus.jsonl
  data/coliee/queries.jsonl
  data/coliee/qrels_train.tsv
  data/coliee/qrels_test.tsv

Usage:
  python src/scripts/import_coliee.py
"""
import csv
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

HF_DATASET_ID = "ghanahmada/kuhperdata"
CONFIG = "coliee"
OUTPUT_DIR = "data/coliee"

SPLITS = {
    "corpus":      ("jsonl", f"{OUTPUT_DIR}/corpus.jsonl"),
    "queries":     ("jsonl", f"{OUTPUT_DIR}/queries.jsonl"),
    "qrels_train": ("tsv",   f"{OUTPUT_DIR}/qrels_train.tsv"),
    "qrels_test":  ("tsv",   f"{OUTPUT_DIR}/qrels_test.tsv"),
}


def export_jsonl(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path}")


def export_tsv(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "score"])
        for row in rows:
            writer.writerow([row["query_id"], row["doc_id"], row["score"]])
    print(f"  Wrote {len(rows)} rows -> {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split, (fmt, out_path) in SPLITS.items():
        repo_path = f"{CONFIG}/{split}-00000-of-00001.parquet"
        try:
            local = hf_hub_download(HF_DATASET_ID, repo_path, repo_type="dataset")
        except Exception as e:
            print(f"  SKIP: {repo_path} — {e}")
            continue
        rows = pd.read_parquet(local).to_dict(orient="records")
        if fmt == "jsonl":
            export_jsonl(rows, out_path)
        else:
            export_tsv(rows, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
