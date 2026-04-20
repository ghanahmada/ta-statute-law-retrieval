"""Download StructGNN weights and inference results from HuggingFace Hub.

Downloads zip files from the HF repo and extracts model checkpoints
to outputs/paragnn/{dataset}/ and inference results to outputs/inference/{dataset}/.

Usage:
  # Pull everything
  python src/inference/import_hf.py

  # Pull specific dataset only
  python src/inference/import_hf.py --dataset kuhperdata-humanized

  # Pull including precomputed embeddings/BM25 scores
  python src/inference/import_hf.py --include_precomputed
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

DATASETS = ["kuhperdata-humanized", "kuhperdata-summarized", "bsard", "stard"]

METHODS = {
    "adapted_struct": "StructGNN",
    "adapted": "Para-GNN",
    "adapted_prox50": "Prox-GNN",
}


def download_and_extract(repo_id, repo_type, datasets, outputs_dir, inference_dir,
                         include_precomputed, token=None):
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

    zip_files = [f for f in repo_files if f.endswith(".zip")]
    print(f"Found {len(zip_files)} zip files: {', '.join(zip_files)}")

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds}")
        print(f"{'='*50}")

        for method_suffix in METHODS:
            zip_name = f"{ds}_{method_suffix}_model.zip"
            if zip_name not in zip_files:
                continue

            target_dir = Path(outputs_dir) / ds / method_suffix
            target_dir.mkdir(parents=True, exist_ok=True)

            if (target_dir / "best_model.pt").exists():
                print(f"  {METHODS[method_suffix]}: already exists at {target_dir}, skipping")
                continue

            print(f"  Downloading {zip_name}...")
            local_path = hf_hub_download(
                repo_id, zip_name, repo_type=repo_type, token=token,
            )

            print(f"  Extracting to {target_dir}...")
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(target_dir)
                for name in zf.namelist():
                    print(f"    + {name}")

        inference_zip = f"{ds}_inference.zip"
        if inference_zip in zip_files:
            target_dir = Path(inference_dir) / ds
            target_dir.mkdir(parents=True, exist_ok=True)

            existing = list(target_dir.glob("*.json")) + list(target_dir.glob("*.run"))
            if existing:
                print(f"  Inference: already exists at {target_dir}, skipping")
            else:
                print(f"  Downloading {inference_zip}...")
                local_path = hf_hub_download(
                    repo_id, inference_zip, repo_type=repo_type, token=token,
                )
                print(f"  Extracting to {target_dir}...")
                with zipfile.ZipFile(local_path, "r") as zf:
                    zf.extractall(target_dir)
                    for name in zf.namelist():
                        print(f"    + {name}")

        if include_precomputed:
            precomp_zip = f"{ds}_precomputed.zip"
            if precomp_zip in zip_files:
                target_dir = Path(outputs_dir) / ds
                target_dir.mkdir(parents=True, exist_ok=True)

                if (target_dir / "bm25_test_scores.pt").exists():
                    print(f"  Precomputed: already exists at {target_dir}, skipping")
                else:
                    print(f"  Downloading {precomp_zip}...")
                    local_path = hf_hub_download(
                        repo_id, precomp_zip, repo_type=repo_type, token=token,
                    )
                    print(f"  Extracting to {target_dir}...")
                    with zipfile.ZipFile(local_path, "r") as zf:
                        zf.extractall(target_dir)
                    print(f"    Extracted {len(zf.namelist())} files")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Download StructGNN from HuggingFace")
    parser.add_argument("--repo_id", default="ghanahmada/kuhperdata")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    parser.add_argument("--outputs_dir", default="outputs/paragnn")
    parser.add_argument("--inference_dir", default="outputs/inference")
    parser.add_argument("--include_precomputed", action="store_true",
                        help="Also download precomputed embeddings and BM25 scores")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    token = args.token or os.environ.get("HF_TOKEN")

    download_and_extract(
        args.repo_id, args.repo_type, datasets, args.outputs_dir, args.inference_dir,
        args.include_precomputed, token,
    )


if __name__ == "__main__":
    main()
