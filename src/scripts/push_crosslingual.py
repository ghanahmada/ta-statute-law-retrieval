"""Push cross-lingual statute retrieval datasets (BSARD, STARD) to HuggingFace.

Each dataset is uploaded to its own HF repo as parquet splits:
  corpus, queries, qrels_train, qrels_val (if exists), qrels_test

Usage:
  python src/scripts/push_crosslingual.py                  # all datasets
  python src/scripts/push_crosslingual.py --datasets bsard  # single dataset
"""

import argparse
import csv
import io
import json
import os

from datasets import Dataset
from huggingface_hub import HfApi, login

DATASETS = {
    "bsard": {
        "path": "data/bsard",
        "repo_id": "ghanahmada/bsard",
        "lang": "fr",
        "description": "Belgian Statutory Article Retrieval Dataset (French)",
    },
    "stard": {
        "path": "data/stard",
        "repo_id": "ghanahmada/stard",
        "lang": "zh",
        "description": "Chinese Statute Article Retrieval Dataset (Mandarin)",
    },
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "score": int(row["score"]),
            })
    return rows


def upload_dataset(api: HfApi, name: str, cfg: dict):
    data_dir = cfg["path"]
    repo_id = cfg["repo_id"]

    print(f"\n{'='*50}")
    print(f"Uploading {name} -> {repo_id}")
    print(f"{'='*50}")

    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries.jsonl")
    qrels_train_path = os.path.join(data_dir, "qrels_train.tsv")
    qrels_val_path = os.path.join(data_dir, "qrels_val.tsv")
    qrels_test_path = os.path.join(data_dir, "qrels_test.tsv")

    for p in [corpus_path, queries_path, qrels_train_path, qrels_test_path]:
        if not os.path.exists(p):
            print(f"  SKIP: {p} not found")
            return False

    # Ensure repo exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"  Warning creating repo: {e}")

    corpus_ds = Dataset.from_list(load_jsonl(corpus_path))
    print(f"  Corpus: {len(corpus_ds)} documents")

    queries_ds = Dataset.from_list([
        {"_id": q["_id"], "text": q["text"]}
        for q in load_jsonl(queries_path)
    ])
    print(f"  Queries: {len(queries_ds)} queries")

    qrels_train_ds = Dataset.from_list(load_tsv(qrels_train_path))
    qrels_test_ds = Dataset.from_list(load_tsv(qrels_test_path))
    print(f"  Qrels train: {len(qrels_train_ds)}, test: {len(qrels_test_ds)}")

    splits = {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels_train": qrels_train_ds,
        "qrels_test": qrels_test_ds,
    }

    if os.path.exists(qrels_val_path):
        qrels_val_ds = Dataset.from_list(load_tsv(qrels_val_path))
        splits["qrels_val"] = qrels_val_ds
        print(f"  Qrels val: {len(qrels_val_ds)}")

    for split_name, split_ds in splits.items():
        buf = io.BytesIO()
        split_ds.to_parquet(buf)
        buf.seek(0)
        path_in_repo = f"{split_name}-00000-of-00001.parquet"
        print(f"  Uploading {split_name} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload README
    split_names = list(splits.keys())
    readme_lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for split_name in split_names:
        readme_lines.append(f"  - split: {split_name}")
        readme_lines.append(f"    path: {split_name}-*.parquet")
    readme_lines += [
        "---",
        "",
        f"# {name.upper()}",
        "",
        cfg["description"],
        "",
        f"Language: `{cfg['lang']}`",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'ds = load_dataset("{repo_id}")',
        "corpus = ds['corpus']",
        "queries = ds['queries']",
        "```",
        "",
    ]
    readme = "\n".join(readme_lines)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded README.md")
    return True


def main():
    parser = argparse.ArgumentParser(description="Push cross-lingual datasets to HuggingFace Hub")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    login(token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"))
    api = HfApi()

    names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    for name in names:
        upload_dataset(api, name, DATASETS[name])

    print("\nDone.")


if __name__ == "__main__":
    main()
