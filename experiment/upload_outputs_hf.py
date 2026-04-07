"""Upload Para-GNN outputs to HuggingFace Hub.

Usage:
  python experiment/upload_outputs_hf.py
  python experiment/upload_outputs_hf.py --repo ghanahmada/kuhperdata --folder outputs/paragnn
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload outputs to HuggingFace Hub")
    parser.add_argument("--repo", default="ghanahmada/kuhperdata", help="HuggingFace repo ID")
    parser.add_argument("--folder", default="outputs/paragnn", help="Local folder to upload")
    parser.add_argument("--path_in_repo", default="outputs/paragnn", help="Path in repo")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return

    # Count files
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(args.folder):
        for f in files:
            path = os.path.join(root, f)
            total_files += 1
            total_size += os.path.getsize(path)

    print(f"Uploading {total_files} files ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  From: {args.folder}")
    print(f"  To:   {args.repo}/{args.path_in_repo}")

    api = HfApi()
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
    )
    print("Done!")


if __name__ == "__main__":
    main()
