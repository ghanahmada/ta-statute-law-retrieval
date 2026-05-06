import argparse
import csv
import io
import json
import os

from datasets import Dataset
from huggingface_hub import HfApi, login

HF_DATASET_ID = "ghanahmada/kuhperdata"

CONFIGS = {
    "humanized": "data/kuhperdata-humanized",
    "summarized": "data/kuhperdata-summarized",
    "humanized-expanded": "data/kuhperdata-exp",
    "summarized-expanded": "data/kuhperdata-summ-exp",
    "bsard": "data/bsard",
    "stard": "data/stard",
}


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


def upload_config(api, config_name, data_dir):
    print(f"\n{'='*50}")
    print(f"Uploading config: {config_name} from {data_dir}")
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

    corpus_ds = Dataset.from_list(load_jsonl(corpus_path))
    print(f"  Corpus: {len(corpus_ds)} documents")

    raw_queries = load_jsonl(queries_path)
    queries_rows = [{"_id": q["_id"], "text": q["text"]} for q in raw_queries]
    queries_ds = Dataset.from_list(queries_rows)
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
        path_in_repo = f"{config_name}/{split_name}-00000-of-00001.parquet"
        print(f"  Uploading {split_name} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
        )

    return True


def build_readme(uploaded_configs):
    lines = ["---", "configs:"]
    for cfg in uploaded_configs:
        lines.append(f"- config_name: {cfg}")
        lines.append("  data_files:")
        for split in ["corpus", "queries", "qrels_train", "qrels_val", "qrels_test"]:
            lines.append(f"  - split: {split}")
            lines.append(f"    path: {cfg}/{split}-*.parquet")
    if uploaded_configs:
        lines.append(f"default_config_name: {uploaded_configs[0]}")
    lines.append("---")
    lines.append("")
    lines.append("# Statute Law Retrieval Benchmark")
    lines.append("")
    lines.append("Multi-lingual statute article retrieval benchmark covering Indonesian (KUHPerdata), French (BSARD), and Chinese (STARD) legal corpora.")
    lines.append("")
    lines.append("## Configs")
    lines.append("")
    lines.append("| Config | Lang | Description |")
    lines.append("|--------|------|-------------|")
    desc = {
        "humanized": ("id", "KUHPerdata — layperson queries, citation-based qrels"),
        "summarized": ("id", "KUHPerdata — case summary queries, citation-based qrels"),
        "humanized-expanded": ("id", "KUHPerdata — layperson queries, LLM-validated expanded qrels"),
        "summarized-expanded": ("id", "KUHPerdata — case summary queries, LLM-validated expanded qrels"),
        "bsard": ("fr", "Belgian Statutory Article Retrieval Dataset"),
        "stard": ("zh", "Chinese Statute Article Retrieval Dataset"),
    }
    for cfg in uploaded_configs:
        lang, description = desc.get(cfg, ("", ""))
        lines.append(f"| `{cfg}` | `{lang}` | {description} |")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append("")
    lines.append(f'ds = load_dataset("{HF_DATASET_ID}", "humanized")')
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Push kuhperdata configs to HuggingFace Hub")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which configs to push (default: all available)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing data from HF repo before pushing",
    )
    args = parser.parse_args()

    login(token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"))
    api = HfApi()

    if args.clean:
        for folder in list(CONFIGS.keys()) + ["data", "bsard", "stard"]:
            try:
                api.delete_folder(
                    folder_path=folder, repo_id=HF_DATASET_ID, repo_type="dataset"
                )
                print(f"Deleted {folder}/ from HF repo.")
            except Exception:
                pass

    configs_to_push = list(CONFIGS.keys()) if "all" in args.configs else args.configs

    uploaded = []
    for cfg in configs_to_push:
        data_dir = CONFIGS[cfg]
        if upload_config(api, cfg, data_dir):
            uploaded.append(cfg)

    if uploaded:
        readme = build_readme(uploaded)
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
        )
        print(f"\nUploaded README.md with {len(uploaded)} configs: {uploaded}")

    print("\nDone.")


if __name__ == "__main__":
    main()
