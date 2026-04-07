"""Upload Para-GNN outputs to HuggingFace Hub.

Zips embedding folders (thousands of small .pt files) into single archives
before upload to avoid HuggingFace's 25K file limit per commit.

Usage:
  python experiment/upload_outputs_hf.py
  python experiment/upload_outputs_hf.py --dataset kuhperdata-humanized
"""
import argparse
import os
import zipfile
import shutil

from huggingface_hub import HfApi


def zip_embeddings(dataset_dir):
    """Zip corpus/ and queries/ embedding folders into .zip archives."""
    emb_dir = os.path.join(dataset_dir, "embeddings")
    if not os.path.exists(emb_dir):
        return

    for subfolder in ["corpus", "queries"]:
        folder_path = os.path.join(emb_dir, subfolder)
        zip_path = os.path.join(emb_dir, f"{subfolder}.zip")

        if not os.path.exists(folder_path):
            continue
        if os.path.exists(zip_path):
            print(f"  {zip_path} already exists, skipping")
            continue

        n_files = len(os.listdir(folder_path))
        print(f"  Zipping {folder_path} ({n_files} files)...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in os.listdir(folder_path):
                zf.write(os.path.join(folder_path, f), f)

        zip_size = os.path.getsize(zip_path) / 1024 / 1024
        print(f"  Created {zip_path} ({zip_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Upload Para-GNN outputs to HuggingFace Hub")
    parser.add_argument("--repo", default="ghanahmada/kuhperdata", help="HuggingFace repo ID")
    parser.add_argument("--folder", default="outputs/paragnn", help="Local folder")
    parser.add_argument("--path_in_repo", default="outputs/paragnn", help="Path in repo")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--dataset", default=None, help="Upload single dataset only")
    args = parser.parse_args()

    base_folder = args.folder

    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [d for d in os.listdir(base_folder)
                     if os.path.isdir(os.path.join(base_folder, d))]

    # Step 1: Zip embeddings for each dataset
    print("=== Zipping embeddings ===")
    for ds in datasets:
        ds_dir = os.path.join(base_folder, ds)
        if os.path.exists(ds_dir):
            zip_embeddings(ds_dir)

    # Step 2: Upload per dataset (excluding raw embedding folders, include .zip)
    api = HfApi()
    for ds in datasets:
        ds_dir = os.path.join(base_folder, ds)
        if not os.path.exists(ds_dir):
            print(f"Skipping {ds} (not found)")
            continue

        repo_path = f"{args.path_in_repo}/{ds}"

        # Count uploadable files (exclude raw corpus/ queries/ folders)
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(ds_dir):
            rel = os.path.relpath(root, ds_dir).replace("\\", "/")
            # Skip raw embedding folders (we have the .zip)
            if rel.startswith("embeddings/corpus") or rel.startswith("embeddings/queries"):
                continue
            for f in files:
                total_files += 1
                total_size += os.path.getsize(os.path.join(root, f))

        print(f"\n=== Uploading {ds}: {total_files} files ({total_size / 1024 / 1024:.1f} MB) ===")

        api.upload_folder(
            folder_path=ds_dir,
            repo_id=args.repo,
            path_in_repo=repo_path,
            repo_type=args.repo_type,
            ignore_patterns=["embeddings/corpus/*.pt", "embeddings/queries/*.pt"],
        )
        print(f"  Done: {ds}")


if __name__ == "__main__":
    main()
