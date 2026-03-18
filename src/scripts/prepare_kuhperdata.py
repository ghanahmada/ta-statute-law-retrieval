import json
import os
import re
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, login

HF_DATASET_ID = "ghanahmada/kuhperdata"
OUTPUT_DIR = os.path.join("data", "kuhperdata")

# Parquet file paths in the HuggingFace repo
PARQUET_FILES = {
    "corpus": "data/corpus-00000-of-00001.parquet",
    "queries": "data/queries-00000-of-00001.parquet",
    "qrels_train": "data/qrels_train-00000-of-00001.parquet",
    "qrels_test": "data/qrels_test-00000-of-00001.parquet",
}

RAW_DOWNLOADS_PREFIX = "cleaned_downloads/"


def strip_statute_references(text: str) -> str:
    """Remove statute references from query text to prevent data leakage.

    Strips patterns like:
      - "Pasal 1266 KUHPerdata" (with law name)
      - "Pasal 1266 dan 1267 KUHPerdata" (chained with law name)
      - "(Pasal 1320)" bare references without law name
      - "Kitab Undang-Undang Hukum Perdata" / "KUHPerdata" / "KUH Perdata"
    Preserves references to other laws: "Pasal 283 RBg", "Pasal 372 KUHP", etc.
    """
    _KUHPERDATA = r'(KUHPerdata|KUH\s*Perdata|Kitab\s+Undang-Undang\s+Hukum\s+Perdata)'
    _OTHER_LAWS = r'\s+(?:RBg|HIR|KUHP|KUHAP|UU|UUD|PP|Perpres|Perma|KUHPidana|KUHDagang)\b'
    _PASAL_CHAIN = r'Pasal\s+\d+[a-zA-Z]?(\s*(,|dan|dan\s+Pasal)\s+\d+[a-zA-Z]?)*'

    text = re.sub(_PASAL_CHAIN + r'\s+' + _KUHPERDATA, '', text, flags=re.IGNORECASE)
    text = re.sub(_KUHPERDATA, '', text, flags=re.IGNORECASE)

    def _keep_other_law(m):
        after = m.string[m.end():]
        if re.match(_OTHER_LAWS, after, re.IGNORECASE):
            return m.group(0)
        return ''

    text = re.sub(_PASAL_CHAIN, _keep_other_law, text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def download_raw_pdfs(output_dir: str) -> int:
    """Download raw PDF files from the HuggingFace dataset downloads/ folder."""
    repo_files = list_repo_files(repo_id=HF_DATASET_ID, repo_type="dataset")
    raw_pdf_files = [
        remote_path
        for remote_path in repo_files
        if remote_path.startswith(RAW_DOWNLOADS_PREFIX) and remote_path.lower().endswith(".pdf")
    ]

    if not raw_pdf_files:
        print("No raw PDFs found under downloads/ in dataset repository.")
        return 0

    for remote_path in raw_pdf_files:
        cached_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=remote_path,
            repo_type="dataset",
        )
        local_path = os.path.join(output_dir, remote_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(cached_path, local_path)

    print(f"Wrote {len(raw_pdf_files)} raw PDFs to {os.path.join(output_dir, 'downloads')}")
    return len(raw_pdf_files)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    login()

    # Download parquet files directly to avoid schema unification issues
    def load_parquet(split_name: str) -> pd.DataFrame:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=PARQUET_FILES[split_name],
            repo_type="dataset",
        )
        return pd.read_parquet(local_path)

    corpus_df = load_parquet("corpus")
    queries_df = load_parquet("queries")
    qrels_train_df = load_parquet("qrels_train")
    qrels_test_df = load_parquet("qrels_test")

    # --- downloads/*.pdf (raw source files) ---
    num_raw_pdfs = download_raw_pdfs(OUTPUT_DIR)

    # --- corpus.jsonl ---
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _, doc in corpus_df.iterrows():
            entry = {
                "_id": str(doc["_id"]),
                "title": doc["title"],
                "text": doc["text"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus_df)} documents to {corpus_path}")

    # --- queries.jsonl (with statute reference stripping) ---
    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for _, q in queries_df.iterrows():
            entry = {
                "_id": str(q["_id"]),
                "text": strip_statute_references(q["text"]),
                "metadata": {"case_name": q.get("case_name", "")},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(queries_df)} queries to {queries_path}")

    # --- qrels_train.tsv / qrels_test.tsv ---
    for split, qrels_df in [("train", qrels_train_df), ("test", qrels_test_df)]:
        qrels_path = os.path.join(OUTPUT_DIR, f"qrels_{split}.tsv")
        with open(qrels_path, "w", encoding="utf-8") as f:
            f.write("query_id\tdoc_id\tscore\n")
            for _, row in qrels_df.iterrows():
                f.write(f"{row['query_id']}\t{row['doc_id']}\t{row['score']}\n")
        print(f"Wrote {len(qrels_df)} judgments to {qrels_path}")

    # --- dataset_stats.json ---
    n_train = len(qrels_train_df)
    n_test = len(qrels_test_df)
    stats = {
        "dataset": "kuhperdata",
        "language": "id",
        "num_documents": len(corpus_df),
        "num_queries": len(queries_df),
        "num_relevance_judgments": n_train + n_test,
        "num_train_judgments": n_train,
        "num_test_judgments": n_test,
        "num_raw_pdfs": num_raw_pdfs,
        "avg_relevant_docs_per_query": (n_train + n_test) / len(queries_df) if len(queries_df) > 0 else 0,
    }
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
