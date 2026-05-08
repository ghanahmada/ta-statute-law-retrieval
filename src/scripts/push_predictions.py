"""
Push retrieval prediction results to HuggingFace Hub.

Uploads prediction JSONL files (from --save_predictions) as parquet splits
under ghanahmada/kuhperdata in a `predictions-{dataset}` config.

Usage:
  # Push all prediction files in a directory
  python src/scripts/push_predictions.py --pred_dir outputs/predictions --dataset kuhperdata-exp

  # Push specific files
  python src/scripts/push_predictions.py \
    --files outputs/predictions/bm25_kuhperdata-exp.jsonl \
           outputs/predictions/jnlp_kuhperdata-exp.jsonl \
    --dataset kuhperdata-exp

Pull:
  from datasets import load_dataset
  ds = load_dataset("ghanahmada/kuhperdata", "predictions-kuhperdata-exp")
  # Each split is a method: ds["bm25"], ds["jnlp_stage1"], etc.
"""

import argparse
import glob
import io
import json
import os

from datasets import Dataset
from huggingface_hub import HfApi, login

HF_DATASET_ID = "ghanahmada/kuhperdata"


def load_predictions_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            row = {
                "qid": rec["qid"],
                "method": rec.get("method", ""),
                "dataset": rec.get("dataset", ""),
                "ranked_doc_ids": json.dumps(rec.get("ranked_doc_ids", [])),
                "ground_truth": json.dumps(rec.get("ground_truth", [])),
                "rr_at_10": rec.get("rr@10", 0.0),
                "recall_at_10": rec.get("recall@10", 0.0),
            }
            if "doc_scores" in rec and rec["doc_scores"]:
                row["doc_scores"] = json.dumps(rec["doc_scores"])
            else:
                row["doc_scores"] = "{}"

            if "ranked_seen_100" in rec:
                row["ranked_doc_ids"] = json.dumps(rec["ranked_seen_100"])
            rows.append(row)
    return rows


def load_paragnn_jsonl(path: str) -> list[dict]:
    """Load Para-GNN/StructGNN inference format."""
    rows = []
    method = "paragnn" if "paragnn" in path.lower() else "structgnn" if "structgnn" in path.lower() else "gnn"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ranked = rec.get("rankings", [])
            gt = rec.get("ground_truth", [])
            row = {
                "qid": rec["qid"],
                "method": method,
                "dataset": "",
                "ranked_doc_ids": json.dumps([r["doc_id"] for r in ranked]),
                "ground_truth": json.dumps([g["doc_id"] for g in gt]),
                "rr_at_10": 0.0,
                "recall_at_10": 0.0,
                "doc_scores": json.dumps({r["doc_id"]: r["score"] for r in ranked}),
            }
            rows.append(row)
    return rows


def load_agent_log_jsonl(path: str) -> list[dict]:
    """Load agentic agent_log.jsonl format."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ranked = rec.get("ranked_doc_ids", [])
            seen_100 = rec.get("ranked_seen_100", ranked)
            scores = rec.get("doc_scores", {})
            row = {
                "qid": rec["qid"],
                "method": "agentic",
                "dataset": "",
                "ranked_doc_ids": json.dumps(seen_100[:100]),
                "ground_truth": json.dumps(rec.get("ground_truth", [])),
                "rr_at_10": 0.0,
                "recall_at_10": 0.0,
                "doc_scores": json.dumps(scores),
            }
            rows.append(row)
    return rows


def detect_format(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    if "rankings" in first and isinstance(first["rankings"], list):
        return "paragnn"
    if "conversation" in first or "n_frames_declared" in first:
        return "agent_log"
    return "predictions"


def main():
    parser = argparse.ArgumentParser(description="Push prediction results to HuggingFace")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Directory containing prediction JSONL files")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific prediction JSONL files to push")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name for the HF config (e.g. kuhperdata-exp)")
    parser.add_argument("--split_names", nargs="+", default=None,
                        help="Split names (method names) for each file. Auto-detected if not given.")
    args = parser.parse_args()

    files = args.files or []
    if args.pred_dir:
        files.extend(sorted(glob.glob(os.path.join(args.pred_dir, "*.jsonl"))))

    if not files:
        print("No files to push. Use --pred_dir or --files.")
        return

    login(token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"))
    api = HfApi()

    config_name = f"predictions-{args.dataset}"
    print(f"Pushing to {HF_DATASET_ID} config={config_name}")

    uploaded_splits = []

    for i, path in enumerate(files):
        fmt = detect_format(path)
        if fmt == "paragnn":
            rows = load_paragnn_jsonl(path)
        elif fmt == "agent_log":
            rows = load_agent_log_jsonl(path)
        else:
            rows = load_predictions_jsonl(path)

        if not rows:
            print(f"  SKIP {path}: empty")
            continue

        if args.split_names and i < len(args.split_names):
            split_name = args.split_names[i]
        else:
            split_name = rows[0].get("method", "") or os.path.splitext(os.path.basename(path))[0]
        split_name = split_name.replace("-", "_").replace(" ", "_")

        for row in rows:
            row["dataset"] = args.dataset

        ds = Dataset.from_list(rows)
        buf = io.BytesIO()
        ds.to_parquet(buf)
        buf.seek(0)

        path_in_repo = f"{config_name}/{split_name}-00000-of-00001.parquet"
        print(f"  {path} -> {path_in_repo} ({len(rows)} queries)")
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
        )
        uploaded_splits.append(split_name)

    print(f"\nUploaded {len(uploaded_splits)} splits: {uploaded_splits}")
    print(f"\nPull with:")
    print(f'  ds = load_dataset("{HF_DATASET_ID}", "{config_name}")')
    for s in uploaded_splits:
        print(f'  ds["{s}"]  # {s} predictions')


if __name__ == "__main__":
    main()
