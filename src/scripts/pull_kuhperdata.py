"""
Pull KUHPerdata dataset configs from HuggingFace and write BEIR-format files.

Downloads parquet splits from ghanahmada/kuhperdata and converts each config
to: corpus.jsonl, queries.jsonl, qrels_train.tsv, qrels_test.tsv

Usage:
  python src/scripts/pull_kuhperdata.py                          # all 4 configs
  python src/scripts/pull_kuhperdata.py --configs humanized      # single config
  python src/scripts/pull_kuhperdata.py --configs humanized summarized-expanded
"""

import argparse
import json
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import pandas as pd
from huggingface_hub import hf_hub_download, login

HF_DATASET_ID = "ghanahmada/kuhperdata"

CONFIGS = {
    "humanized": "data/kuhperdata-humanized",
    "summarized": "data/kuhperdata-summarized",
    "humanized-expanded": "data/kuhperdata-exp",
    "summarized-expanded": "data/kuhperdata-summ-exp",
}

SPLITS = ["corpus", "queries", "qrels_train", "qrels_test"]


def download_config(config_name: str, output_dir: str):
    print(f"\n{'='*50}")
    print(f"Pulling config: {config_name} -> {output_dir}")
    print(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)

    frames = {}
    for split in SPLITS:
        remote_path = f"{config_name}/{split}-00000-of-00001.parquet"
        local_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=remote_path,
            repo_type="dataset",
        )
        frames[split] = pd.read_parquet(local_path)
        print(f"  {split}: {len(frames[split])} rows")

    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _, row in frames["corpus"].iterrows():
            entry = {"_id": str(row["_id"]), "title": row["title"], "text": row["text"]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {corpus_path}")

    queries_path = os.path.join(output_dir, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for _, row in frames["queries"].iterrows():
            entry = {
                "_id": str(row["_id"]),
                "text": row["text"],
                "metadata": {"case_name": row.get("case_name", "")},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {queries_path}")

    for split in ["qrels_train", "qrels_test"]:
        tsv_path = os.path.join(output_dir, f"{split}.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("query_id\tdoc_id\tscore\n")
            for _, row in frames[split].iterrows():
                f.write(f"{row['query_id']}\t{row['doc_id']}\t{row['score']}\n")
        print(f"  Wrote {tsv_path}")


def main():
    parser = argparse.ArgumentParser(description="Pull kuhperdata configs from HuggingFace Hub")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which configs to pull (default: all)",
    )
    args = parser.parse_args()

    login(token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"))

    configs = list(CONFIGS.keys()) if "all" in args.configs else args.configs

    for cfg in configs:
        download_config(cfg, CONFIGS[cfg])

    print("\nDone.")


if __name__ == "__main__":
    main()
