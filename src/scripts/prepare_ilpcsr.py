import json
import os

from datasets import load_dataset
from huggingface_hub import login

OUTPUT_DIR = os.path.join("data", "ilpcsr")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    login()

    ds_queries = load_dataset("Exploration-Lab/IL-PCSR", name="queries")
    ds_statutes = load_dataset("Exploration-Lab/IL-PCSR", name="statutes")

    statutes = ds_statutes["statute_candidates"]
    train_qs = list(ds_queries["train_queries"])
    dev_qs = list(ds_queries["dev_queries"])
    test_qs = list(ds_queries["test_queries"])

    # --- corpus.jsonl (statutes only) ---
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in statutes:
            text = doc["text"]
            if isinstance(text, list):
                text = " ".join(text)
            entry = {
                "_id": str(doc["id"]),
                "title": doc.get("provision_name", ""),
                "text": text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(statutes)} statutes to {corpus_path}")

    # --- queries.jsonl + qrels ---
    corpus_ids = {str(doc["id"]) for doc in statutes}
    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    qrels_train_path = os.path.join(OUTPUT_DIR, "qrels_train.tsv")
    qrels_val_path = os.path.join(OUTPUT_DIR, "qrels_val.tsv")
    qrels_test_path = os.path.join(OUTPUT_DIR, "qrels_test.tsv")

    n_train_judgments = 0
    n_val_judgments = 0
    n_test_judgments = 0
    n_missing_docs = 0

    def process_queries(queries_list, qrels_file, split_name):
        nonlocal n_train_judgments, n_val_judgments, n_test_judgments, n_missing_docs
        entries = []
        for q in queries_list:
            qid = str(q["id"])
            text = q["text"]
            if isinstance(text, list):
                text = " ".join(text)

            query_entry = {
                "_id": qid,
                "text": text,
                "metadata": {
                    "case_title": q.get("case_title", ""),
                    "jurisdiction": q.get("jurisdiction", ""),
                },
            }
            entries.append(query_entry)

            statute_ids = q.get("relevant_statute_ids", [])
            if statute_ids is None:
                statute_ids = []
            for sid in statute_ids:
                sid = str(sid)
                if sid not in corpus_ids:
                    n_missing_docs += 1
                    continue
                qrels_file.write(f"{qid}\t{sid}\t1\n")
                if split_name == "train":
                    n_train_judgments += 1
                elif split_name == "val":
                    n_val_judgments += 1
                else:
                    n_test_judgments += 1
        return entries

    with (
        open(queries_path, "w", encoding="utf-8") as fq,
        open(qrels_train_path, "w", encoding="utf-8") as ft,
        open(qrels_val_path, "w", encoding="utf-8") as fv,
        open(qrels_test_path, "w", encoding="utf-8") as fte,
    ):
        ft.write("query_id\tdoc_id\tscore\n")
        fv.write("query_id\tdoc_id\tscore\n")
        fte.write("query_id\tdoc_id\tscore\n")

        all_entries = []
        all_entries.extend(process_queries(train_qs, ft, "train"))
        all_entries.extend(process_queries(dev_qs, fv, "val"))
        all_entries.extend(process_queries(test_qs, fte, "test"))

        for entry in all_entries:
            fq.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_entries)} queries to {queries_path}")
    print(f"Train judgments: {n_train_judgments}")
    print(f"Val judgments (dev): {n_val_judgments}")
    print(f"Test judgments: {n_test_judgments}")
    if n_missing_docs:
        print(f"Missing doc references: {n_missing_docs}")

    # --- dataset_stats.json ---
    total_judgments = n_train_judgments + n_val_judgments + n_test_judgments
    n_queries = len(all_entries)

    stats = {
        "dataset": "ilpcsr",
        "language": "en",
        "task": "statute_retrieval_only",
        "num_documents": len(statutes),
        "num_queries": n_queries,
        "num_train_queries": len(train_qs),
        "num_val_queries": len(dev_qs),
        "num_test_queries": len(test_qs),
        "num_relevance_judgments": total_judgments,
        "num_train_judgments": n_train_judgments,
        "num_val_judgments": n_val_judgments,
        "num_test_judgments": n_test_judgments,
        "avg_relevant_docs_per_query": total_judgments / n_queries if n_queries > 0 else 0,
        "note": "dev split → qrels_val.tsv, test split → qrels_test.tsv; statute retrieval only",
    }

    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
