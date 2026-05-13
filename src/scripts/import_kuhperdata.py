"""Download kuhperdata-exp and kuhperdata-summ-exp from HuggingFace.

Converts parquet splits back to the expected local format:
  corpus.jsonl, queries.jsonl, qrels_train.tsv, qrels_val.tsv, qrels_test.tsv

Usage:
  python src/scripts/import_kuhperdata.py
  python src/scripts/import_kuhperdata.py --configs humanized-expanded
"""
import argparse
import csv
import json
import os

HF_DATASET_ID = "ghanahmada/kuhperdata"

CONFIGS = {
    "humanized-expanded": "data/kuhperdata-exp",
    "summarized-expanded": "data/kuhperdata-summ-exp",
}


def export_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path}")


def export_tsv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "score"])
        for row in rows:
            writer.writerow([row["query_id"], row["doc_id"], row["score"]])
    print(f"  Wrote {len(rows)} rows -> {path}")


def import_config(config_name, data_dir):
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"Importing {config_name} -> {data_dir}")
    print(f"{'='*50}")
    os.makedirs(data_dir, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID, config_name)

    export_jsonl([dict(r) for r in ds["corpus"]], f"{data_dir}/corpus.jsonl")
    export_jsonl([dict(r) for r in ds["queries"]], f"{data_dir}/queries.jsonl")
    export_tsv([dict(r) for r in ds["qrels_train"]], f"{data_dir}/qrels_train.tsv")
    export_tsv([dict(r) for r in ds["qrels_test"]], f"{data_dir}/qrels_test.tsv")

    if "qrels_val" in ds:
        export_tsv([dict(r) for r in ds["qrels_val"]], f"{data_dir}/qrels_val.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    configs = list(CONFIGS.keys()) if "all" in args.configs else args.configs
    for cfg in configs:
        import_config(cfg, CONFIGS[cfg])

    print("\nDone.")


if __name__ == "__main__":
    main()
