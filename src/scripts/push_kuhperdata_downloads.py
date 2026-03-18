import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, login

HF_DATASET_ID = "ghanahmada/kuhperdata"
REMOTE_DOWNLOADS_DIR = "cleaned_downloads"
SOURCE_DIR_CANDIDATES = [
    "cleaned_downloads",
    os.path.join("experiment", "cleaned_downloads"),
    os.path.join("data", "kuhperdata", "cleaned_downloads"),
]


def resolve_source_dir(user_source_dir: str | None) -> Path:
    if user_source_dir:
        source_dir = Path(user_source_dir)
        if source_dir.exists() and source_dir.is_dir():
            return source_dir
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for candidate in SOURCE_DIR_CANDIDATES:
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path

    joined = ", ".join(SOURCE_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not auto-detect source downloads directory. "
        f"Checked: {joined}. Use --source_dir explicitly."
    )


def collect_pdf_paths(source_dir: Path) -> list[Path]:
    return sorted(
        [path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"]
    )


def to_rel_posix(path: Path, source_dir: Path) -> str:
    return path.relative_to(source_dir).as_posix()


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def get_existing_remote_pdfs(api: HfApi, repo_id: str) -> set[str]:
    remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return {
        path[len(f"{REMOTE_DOWNLOADS_DIR}/") :]
        for path in remote_files
        if path.startswith(f"{REMOTE_DOWNLOADS_DIR}/") and path.lower().endswith(".pdf")
    }


def upload_batch_with_retry(
    api: HfApi,
    *,
    source_dir: Path,
    repo_id: str,
    rel_paths_batch: list[str],
    batch_index: int,
    total_batches: int,
    max_retries: int,
    retry_wait_seconds: int,
) -> None:
    attempt = 0
    while True:
        try:
            api.upload_folder(
                folder_path=str(source_dir),
                path_in_repo=REMOTE_DOWNLOADS_DIR,
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=rel_paths_batch,
                commit_message=(
                    f"Upload raw PDFs to {REMOTE_DOWNLOADS_DIR}/ "
                    f"(batch {batch_index}/{total_batches}, files={len(rel_paths_batch)})"
                ),
            )
            return
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not is_rate_limit_error(exc):
                raise

            wait_seconds = retry_wait_seconds * (2 ** (attempt - 1))
            print(
                f"Batch {batch_index}/{total_batches} hit rate limit. "
                f"Retry {attempt}/{max_retries} in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload local raw PDFs to Hugging Face dataset repo under downloads/."
        )
    )
    parser.add_argument(
        "--repo_id",
        default=HF_DATASET_ID,
        help="Target Hugging Face dataset repo ID",
    )
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Local directory that contains raw PDF files",
    )
    parser.add_argument(
        "--clear_remote_downloads",
        action="store_true",
        help="Delete existing downloads/ folder in remote repo before upload",
    )
    parser.add_argument(
        "--token_env",
        default="HF_WRITE_TOKEN",
        help="Environment variable name containing Hugging Face write token",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without pushing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=150,
        help="Number of PDFs to upload per batch (default: 150)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=6,
        help="Max retries for one batch when rate limited (default: 6)",
    )
    parser.add_argument(
        "--retry_wait_seconds",
        type=int,
        default=60,
        help="Initial wait before retry after 429; doubles each retry (default: 60)",
    )
    parser.add_argument(
        "--sleep_between_batches",
        type=int,
        default=30,
        help="Cooldown between successful batches in seconds (default: 30)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Stop after this many batches (0 means all pending batches)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip files that already exist in remote downloads/",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max_retries must be >= 0")
    if args.retry_wait_seconds <= 0:
        raise ValueError("--retry_wait_seconds must be > 0")
    if args.sleep_between_batches < 0:
        raise ValueError("--sleep_between_batches must be >= 0")
    if args.max_batches < 0:
        raise ValueError("--max_batches must be >= 0")

    token = os.environ.get(args.token_env) or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        login()

    source_dir = resolve_source_dir(args.source_dir)
    pdf_paths = collect_pdf_paths(source_dir)

    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in source directory: {source_dir}")

    print(f"Source directory: {source_dir}")
    print(f"Target repo: {args.repo_id}")
    print(f"Target path in repo: {REMOTE_DOWNLOADS_DIR}/")
    print(f"Local PDF files found: {len(pdf_paths)}")

    api = HfApi()

    if args.clear_remote_downloads:
        try:
            api.delete_folder(
                folder_path=REMOTE_DOWNLOADS_DIR,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print("Deleted existing downloads/ folder from remote dataset repo.")
        except Exception:
            print("No existing downloads/ folder to delete (or no permission).")

    local_rel_paths = [to_rel_posix(path, source_dir) for path in pdf_paths]

    if args.no_skip_existing:
        pending_rel_paths = local_rel_paths
        print("Skipping existing remote check: disabled by --no_skip_existing")
    else:
        existing_remote = get_existing_remote_pdfs(api, args.repo_id)
        pending_rel_paths = [rel for rel in local_rel_paths if rel not in existing_remote]
        print(f"Remote PDFs already present: {len(existing_remote)}")
        print(f"Pending PDFs to upload: {len(pending_rel_paths)}")

    if not pending_rel_paths:
        print("No pending files to upload. Done.")
        return

    if args.dry_run:
        preview = pending_rel_paths[:20]
        for rel in preview:
            print(f"- {source_dir / Path(rel)} -> {REMOTE_DOWNLOADS_DIR}/{rel}")
        if len(pending_rel_paths) > 20:
            print(f"... and {len(pending_rel_paths) - 20} more pending files")
        return

    batches = chunked(pending_rel_paths, args.batch_size)
    if args.max_batches > 0:
        batches = batches[: args.max_batches]

    print(f"Uploading {sum(len(batch) for batch in batches)} files across {len(batches)} batch(es)")

    for batch_index, rel_paths_batch in enumerate(batches, start=1):
        print(
            f"Starting batch {batch_index}/{len(batches)} "
            f"({len(rel_paths_batch)} files)"
        )

        upload_batch_with_retry(
            api,
            source_dir=source_dir,
            repo_id=args.repo_id,
            rel_paths_batch=rel_paths_batch,
            batch_index=batch_index,
            total_batches=len(batches),
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
        )

        print(f"Batch {batch_index}/{len(batches)} uploaded.")

        if batch_index < len(batches) and args.sleep_between_batches > 0:
            print(f"Sleeping {args.sleep_between_batches}s before next batch...")
            time.sleep(args.sleep_between_batches)

    print("Upload completed.")


if __name__ == "__main__":
    main()