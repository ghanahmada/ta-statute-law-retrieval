"""Export trained StructGNN weights and inference results to HuggingFace Hub.

Zips model checkpoints and inference outputs per dataset, then uploads
to a HuggingFace repository. Uses ZIP_STORED to avoid accuracy loss
and stay under HF's 25K file limit.

Usage:
  # Export all datasets (after running inference)
  python src/inference/export_hf.py --repo_id ghanahmada/structgnn-statute-retrieval

  # Export specific dataset
  python src/inference/export_hf.py --repo_id ghanahmada/structgnn-statute-retrieval --dataset bsard

  # Dry run (zip only, no upload)
  python src/inference/export_hf.py --dry_run
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

DATASETS = [
    "kuhperdata-humanized", "kuhperdata-summarized",
    "kuhperdata-exp", "kuhperdata-summ-exp",
    "bsard", "stard",
]

METHODS = {
    "adapted_struct": "StructGNN",
    "adapted": "Para-GNN",
    "adapted_prox50": "Prox-GNN",
}

MODEL_FILES = [
    "best_model.pt",
    "resume_checkpoint.pt",
    "training_log.json",
    "rankings_top100.jsonl",
    "rankings_top100_original.jsonl",
    "rankings_top100_debiased.jsonl",
]

STAGING_DIR = "outputs/export_staging"


def zip_model_weights(dataset, method_dir, staging_dir):
    """Zip trained model files for a dataset+method."""
    method_name = Path(method_dir).name
    zip_name = f"{dataset}_{method_name}_model.zip"
    zip_path = Path(staging_dir) / zip_name

    files_to_zip = []
    for fname in MODEL_FILES:
        fpath = Path(method_dir) / fname
        if fpath.exists():
            files_to_zip.append((fpath, fname))

    if not files_to_zip:
        print(f"  No model files found in {method_dir}, skipping")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath, arcname in files_to_zip:
            zf.write(fpath, arcname)
            print(f"    + {arcname} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"  Zipped: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return zip_path


def zip_inference_results(dataset, inference_dir, staging_dir):
    """Zip inference result files for a dataset."""
    src_dir = Path(inference_dir) / dataset
    if not src_dir.exists():
        print(f"  No inference results in {src_dir}, skipping")
        return None

    zip_name = f"{dataset}_inference.zip"
    zip_path = Path(staging_dir) / zip_name

    files_to_zip = []
    for f in src_dir.iterdir():
        if f.is_file():
            files_to_zip.append((f, f.name))

    if not files_to_zip:
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath, arcname in files_to_zip:
            zf.write(fpath, arcname)
            print(f"    + {arcname}")

    print(f"  Zipped: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return zip_path


def zip_precomputed(dataset, precompute_dir, staging_dir):
    """Zip precomputed BM25 scores and embeddings (ZIP_STORED for .pt files)."""
    src_dir = Path(precompute_dir)
    if not src_dir.exists():
        print(f"  No precomputed data in {src_dir}, skipping")
        return None

    zip_name = f"{dataset}_precomputed.zip"
    zip_path = Path(staging_dir) / zip_name

    files_to_zip = []

    # Top-level files (BM25 scores, IDs, etc.)
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix in (".pt", ".json"):
            files_to_zip.append((f, f.name))

    # Embeddings directory
    emb_dir = src_dir / "embeddings"
    if emb_dir.exists():
        for root, dirs, files in os.walk(emb_dir):
            for fname in files:
                fpath = Path(root) / fname
                arcname = str(fpath.relative_to(src_dir))
                files_to_zip.append((fpath, arcname))

    if not files_to_zip:
        return None

    print(f"  Zipping {len(files_to_zip)} precomputed files...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath, arcname in files_to_zip:
            zf.write(fpath, arcname)

    print(f"  Zipped: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return zip_path


def upload_to_hf(repo_id, staging_dir, token=None):
    """Upload all staged zip files to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"  Repo creation note: {e}")

    staging = Path(staging_dir)
    files = sorted(staging.glob("*.zip")) + sorted(staging.glob("*.json"))

    print(f"\nUploading {len(files)} files to {repo_id}...")
    for f in files:
        print(f"  Uploading {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"  Done: {f.name}")

    print(f"\nAll uploads complete: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export StructGNN to HuggingFace")
    parser.add_argument("--repo_id", default="ghanahmada/structgnn-statute-retrieval")
    parser.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    parser.add_argument("--outputs_dir", default="outputs/paragnn")
    parser.add_argument("--inference_dir", default="outputs/inference")
    parser.add_argument("--include_precomputed", action="store_true", help="Also export precomputed embeddings+BM25")
    parser.add_argument("--dry_run", action="store_true", help="Zip only, don't upload")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)

    all_zips = []

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds}")
        print(f"{'='*50}")

        # Zip model weights for each method that has a trained model
        for method_suffix in METHODS:
            method_dir = f"{args.outputs_dir}/{ds}/{method_suffix}"
            if Path(method_dir).exists() and Path(f"{method_dir}/best_model.pt").exists():
                print(f"\n  Model: {METHODS[method_suffix]} ({method_suffix})")
                z = zip_model_weights(ds, method_dir, staging)
                if z:
                    all_zips.append(z)

        # Zip inference results
        print(f"\n  Inference results:")
        z = zip_inference_results(ds, args.inference_dir, staging)
        if z:
            all_zips.append(z)

        # Optionally zip precomputed data
        if args.include_precomputed:
            print(f"\n  Precomputed data:")
            z = zip_precomputed(ds, f"{args.outputs_dir}/{ds}", staging)
            if z:
                all_zips.append(z)

    # Write manifest
    manifest = {
        "datasets": datasets,
        "files": [str(z.name) for z in all_zips],
        "methods": METHODS,
    }
    manifest_path = staging / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    if all_zips:
        print(f"\nStaged {len(all_zips)} zip files in {staging}/")
        for z in all_zips:
            print(f"  {z.name} ({z.stat().st_size / 1024 / 1024:.1f} MB)")

    if not args.dry_run:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("\nNo HF token provided. Set HF_TOKEN or use --token.")
            print("To get a token: https://huggingface.co/settings/tokens")
            sys.exit(1)
        upload_to_hf(args.repo_id, staging, token)
    else:
        print("\nDry run — skipping upload. Files staged in outputs/export_staging/")


if __name__ == "__main__":
    main()
