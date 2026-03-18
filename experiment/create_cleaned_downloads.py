import argparse
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class ScanStats:
    scanned_files: int
    scanned_pdfs: int
    copied_pdfs: int
    unreadable_pdfs: int


def process_filename(file_path: str | Path) -> str:
    """Return a path that is safe for long Windows paths."""
    path_obj = Path(file_path).resolve()
    if str(path_obj).startswith("\\\\?\\"):
        return str(path_obj)
    return f"\\\\?\\{path_obj}"


def normalize_for_match(text: str) -> str:
    """Normalize text so keyword matching is tolerant to spacing and punctuation."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def extract_pdf_text(pdf_path: Path) -> str | None:
    safe_path = process_filename(pdf_path)
    try:
        with open(safe_path, "rb") as f:
            file_bytes = f.read()

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n\n".join(page.get_text("text") for page in doc)
    except OSError as e:
        print(f"Error opening {pdf_path.name}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing {pdf_path.name}: {e}")
        return None


def file_matches_keyword(pdf_path: Path, keyword: str) -> tuple[bool, bool]:
    """Return (is_match, is_unreadable)."""
    extracted = extract_pdf_text(pdf_path)
    if extracted is None:
        return False, True

    normalized_keyword = normalize_for_match(keyword)
    normalized_text = normalize_for_match(extracted)
    return normalized_keyword in normalized_text, False


def collect_pdf_candidates(source_dir: Path) -> list[Path]:
    return sorted(
        [path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"]
    )


def _handle_remove_readonly(func, path, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child, onerror=_handle_remove_readonly)
        else:
            child.unlink(missing_ok=True)


def build_cleaned_folder(source_dir: Path, output_dir: Path, keyword: str) -> ScanStats:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_dir_resolved = source_dir.resolve()
    output_dir_resolved = output_dir.resolve()
    if source_dir_resolved == output_dir_resolved:
        raise ValueError("source_dir and output_dir must be different folders")

    scanned_files = sum(1 for path in source_dir.rglob("*") if path.is_file())
    candidate_pdfs = collect_pdf_candidates(source_dir)

    clear_directory_contents(output_dir)

    copied_files = 0
    unreadable_files = 0

    for src_path in candidate_pdfs:
        is_match, is_unreadable = file_matches_keyword(src_path, keyword)
        if is_unreadable:
            unreadable_files += 1
            continue
        if not is_match:
            continue

        rel_path = src_path.relative_to(source_dir)
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_files += 1

    return ScanStats(
        scanned_files=scanned_files,
        scanned_pdfs=len(candidate_pdfs),
        copied_pdfs=copied_files,
        unreadable_pdfs=unreadable_files,
    )


def main() -> None:
    default_source = Path(__file__).resolve().parent / "downloads"
    default_output = Path(__file__).resolve().parent / "cleaned_downloads"

    parser = argparse.ArgumentParser(
        description=(
            "Create cleaned_downloads containing only PDF files whose extracted text "
            "contains a case-insensitive keyword."
        )
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=default_source,
        help=f"Source downloads directory (default: {default_source})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output,
        help=f"Output cleaned directory (default: {default_output})",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="kuhperdata",
        help="Keyword to match in extracted PDF text, case-insensitive (default: kuhperdata)",
    )
    args = parser.parse_args()

    stats = build_cleaned_folder(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        keyword=args.keyword,
    )

    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Keyword: {args.keyword}")
    print(f"Scanned files: {stats.scanned_files}")
    print(f"Scanned PDF files: {stats.scanned_pdfs}")
    print(f"Copied PDF files: {stats.copied_pdfs}")
    print(f"Unreadable PDF files: {stats.unreadable_pdfs}")


if __name__ == "__main__":
    main()