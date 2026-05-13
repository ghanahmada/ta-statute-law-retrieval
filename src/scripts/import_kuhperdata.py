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
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print(f"\n{'='*50}")
    print(f"Importing {config_name} -> {data_dir}")
    print(f"{'='*50}")
    os.makedirs(data_dir, exist_ok=True)

    SPLITS = {
        "corpus":      ("jsonl", f"{data_dir}/corpus.jsonl"),
        "queries":     ("jsonl", f"{data_dir}/queries.jsonl"),
        "qrels_train": ("tsv",   f"{data_dir}/qrels_train.tsv"),
        "qrels_test":  ("tsv",   f"{data_dir}/qrels_test.tsv"),
    }

    for split, (fmt, out_path) in SPLITS.items():
        repo_path = f"{config_name}/{split}-00000-of-00001.parquet"
        try:
            local = hf_hub_download(HF_DATASET_ID, repo_path, repo_type="dataset")
        except Exception:
            print(f"  SKIP: {repo_path} not found on HF")
            continue
        df = pd.read_parquet(local)
        rows = df.to_dict(orient="records")
        if fmt == "jsonl":
            export_jsonl(rows, out_path)
        else:
            export_tsv(rows, out_path)


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
