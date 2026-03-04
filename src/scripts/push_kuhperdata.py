import argparse
import csv
import io
import json
import os

from datasets import Dataset
from huggingface_hub import HfApi, login

HF_DATASET_ID = "ghanahmada/kuhperdata"
DEFAULT_DATA_DIR = os.path.join("data", "kuhperdata")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "query_id": row["query_id"],
                    "doc_id": row["doc_id"],
                    "score": int(row["score"]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Push kuhperdata to HuggingFace Hub")
    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR, help="Path to kuhperdata directory"
    )
    args = parser.parse_args()

    login(token=os.environ.get("HF_READ_TOKEN"))

    # corpus
    corpus_rows = load_jsonl(os.path.join(args.data_dir, "corpus.jsonl"))
    corpus_ds = Dataset.from_list(corpus_rows)
    print(f"Corpus: {len(corpus_ds)} documents")

    # queries — flatten metadata.case_name to top-level
    raw_queries = load_jsonl(os.path.join(args.data_dir, "queries.jsonl"))
    queries_rows = []
    for q in raw_queries:
        queries_rows.append(
            {
                "_id": q["_id"],
                "text": q["text"],
                "case_name": q.get("metadata", {}).get("case_name", ""),
            }
        )
    queries_ds = Dataset.from_list(queries_rows)
    print(f"Queries: {len(queries_ds)} queries")

    # qrels
    qrels_train_ds = Dataset.from_list(
        load_tsv(os.path.join(args.data_dir, "qrels_train.tsv"))
    )
    qrels_test_ds = Dataset.from_list(
        load_tsv(os.path.join(args.data_dir, "qrels_test.tsv"))
    )
    print(f"Qrels train: {len(qrels_train_ds)}, test: {len(qrels_test_ds)}")

    # Upload parquet files manually to avoid push_to_hub schema validation
    api = HfApi()

    # Delete existing data and README to avoid stale schema metadata
    for folder in ("data",):
        try:
            api.delete_folder(
                folder_path=folder, repo_id=HF_DATASET_ID, repo_type="dataset"
            )
            print(f"Deleted existing {folder}/ folder from HF repo.")
        except Exception:
            print(f"No existing {folder}/ folder to delete (first push).")

    try:
        api.delete_file(
            path_in_repo="README.md", repo_id=HF_DATASET_ID, repo_type="dataset"
        )
        print("Deleted existing README.md from HF repo.")
    except Exception:
        pass

    splits = {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels_train": qrels_train_ds,
        "qrels_test": qrels_test_ds,
    }

    # Upload each split as a parquet file
    for split_name, split_ds in splits.items():
        buf = io.BytesIO()
        split_ds.to_parquet(buf)
        buf.seek(0)
        path_in_repo = f"data/{split_name}-00000-of-00001.parquet"
        print(f"Uploading {split_name} to {HF_DATASET_ID}/{path_in_repo}...")
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
        )

    # Upload README.md with split-to-path mappings
    readme = """\
---
configs:
- config_name: default
  data_files:
  - split: corpus
    path: data/corpus-*.parquet
  - split: queries
    path: data/queries-*.parquet
  - split: qrels_train
    path: data/qrels_train-*.parquet
  - split: qrels_test
    path: data/qrels_test-*.parquet
---
"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
    )
    print("Uploaded README.md with dataset config.")

    print("Done.")


if __name__ == "__main__":
    main()
