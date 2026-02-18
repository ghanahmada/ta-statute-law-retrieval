import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import (
    export_ir_dataset,
    load_queries,
    load_statute_documents,
)

# TODO: replace with my HuggingFace dataset ID once uploaded
HF_DATASET_ID = "ghanahmada/kuhperdata"

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_STATUTE = PROJECT_ROOT / "data" / "statute" / "kuh_perdata.csv"
RAW_JUDGEMENT = PROJECT_ROOT / "data" / "judgement" / "judgement_to_content.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kuhperdata"


def download_raw_data():
    """Download raw data from HuggingFace if not present locally."""
    from datasets import load_dataset

    print(f"Downloading raw data from {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID)

    RAW_STATUTE.parent.mkdir(parents=True, exist_ok=True)
    RAW_JUDGEMENT.parent.mkdir(parents=True, exist_ok=True)

    # TODO: adapt save logic based on my HuggingFace dataset structure
    raise NotImplementedError(
        f"Upload your dataset to {HF_DATASET_ID} first, then implement the save logic here. "
        f"For now, place raw files manually:\n"
        f"  - {RAW_STATUTE}\n"
        f"  - {RAW_JUDGEMENT}"
    )


def main():
    if not RAW_STATUTE.exists() or not RAW_JUDGEMENT.exists():
        download_raw_data()

    print("Loading statute documents...")
    documents, doc_ids = load_statute_documents(str(RAW_STATUTE))
    print(f"Loaded {len(documents)} documents")

    print("\nLoading queries (filtered for KUHPerdata only)...")
    queries, ground_truths, case_names = load_queries(str(RAW_JUDGEMENT))
    print(f"Loaded {len(queries)} queries with KUHPerdata ground truth")

    created_files = export_ir_dataset(
        str(OUTPUT_DIR),
        documents,
        doc_ids,
        queries,
        ground_truths,
        case_names,
        test_ratio=0.2,
        n_clusters=50,
        random_state=42,
    )

    print("\nCreated files:")
    for name, path in created_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
