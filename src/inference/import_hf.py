"""Download precomputed data from HuggingFace Hub.

Downloads embedding zips and other precomputed files from the HF dataset
repo, preserving the repo's directory structure locally.

Repo structure on ghanahmada/kuhperdata:
  outputs/paragnn/{dataset}/embeddings/corpus.zip
  outputs/paragnn/{dataset}/embeddings/queries.zip
  outputs/paragnn/{dataset}/*.pt          (BM25 scores, etc.)
  outputs/paragnn/{dataset}/*.json        (ID lists, paragraph maps, etc.)

Usage:
  # Pull all datasets
  python src/inference/import_hf.py

  # Pull specific dataset
  python src/inference/import_hf.py --dataset kuhperdata-humanized

  # List what's on the repo without downloading
  python src/inference/import_hf.py --dry_run
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

DATASETS = ["kuhperdata-humanized", "kuhperdata-summarized", "bsard", "stard"]


def download_dataset_files(repo_id, repo_type, datasets, token=None, dry_run=False):
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    print(f"Listing files in {repo_id} ({repo_type})...")
    try:
        repo_files = list_repo_files(repo_id, repo_type=repo_type, token=token)
    except Exception as e:
        print(f"Error listing repo: {e}")
        sys.exit(1)

    outputs_files = [f for f in repo_files if f.startswith("outputs/")]
    print(f"Found {len(outputs_files)} files under outputs/")

    for ds in datasets:
        prefix = f"outputs/paragnn/{ds}/"
        ds_files = [f for f in outputs_files if f.startswith(prefix)]

        if not ds_files:
            print(f"\n  {ds}: no files found on repo, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Dataset: {ds} ({len(ds_files)} files)")
        print(f"{'='*50}")

        if dry_run:
            for f in ds_files:
                print(f"    {f}")
            continue

        for remote_path in ds_files:
            local_path = Path(remote_path)

            if local_path.suffix == ".zip":
                extract_dir = local_path.parent
                marker = extract_dir / ".downloaded"
                if marker.exists():
                    print(f"  Already extracted: {remote_path}, skipping")
                    continue
            elif local_path.exists():
                print(f"  Already exists: {remote_path}, skipping")
                continue

            print(f"  Downloading {remote_path}...")
            downloaded = hf_hub_download(
                repo_id, remote_path, repo_type=repo_type, token=token,
            )

            if local_path.suffix == ".zip":
                extract_dir = local_path.parent
                extract_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Extracting to {extract_dir}/...")
                with zipfile.ZipFile(downloaded, "r") as zf:
                    zf.extractall(extract_dir)
                    names = zf.namelist()
                    for name in names[:5]:
                        print(f"    + {name}")
                    if len(names) > 5:
                        print(f"    ... and {len(names) - 5} more files")
                marker.write_text("ok")
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(downloaded, local_path)
                print(f"  Saved: {local_path}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Download precomputed data from HuggingFace")
    parser.add_argument("--repo_id", default="ghanahmada/kuhperdata")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    parser.add_argument("--dry_run", action="store_true", help="List files without downloading")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    token = args.token or os.environ.get("HF_TOKEN")

    download_dataset_files(args.repo_id, args.repo_type, datasets, token, args.dry_run)


if __name__ == "__main__":
    main()
