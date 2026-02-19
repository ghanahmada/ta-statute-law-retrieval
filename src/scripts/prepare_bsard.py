import json
import os

from datasets import load_dataset

OUTPUT_DIR = os.path.join("data", "bsard")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load from HuggingFace
    corpus_ds = load_dataset("maastrichtlawtech/bsard", "corpus")
    questions_ds = load_dataset("maastrichtlawtech/bsard", "questions")

    corpus = corpus_ds["corpus"]
    train_questions = list(questions_ds["train"])
    test_questions = list(questions_ds["test"])

    # --- corpus.jsonl ---
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            entry = {
                "_id": str(doc["id"]),
                "title": doc["reference"],
                "text": doc["article"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} documents to {corpus_path}")

    # --- queries.jsonl + qrels ---
    corpus_ids = {str(doc["id"]) for doc in corpus}
    all_questions = train_questions + test_questions

    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    qrels_train_path = os.path.join(OUTPUT_DIR, "qrels_train.tsv")
    qrels_test_path = os.path.join(OUTPUT_DIR, "qrels_test.tsv")

    n_train_judgments = 0
    n_test_judgments = 0
    n_missing_docs = 0

    with (
        open(queries_path, "w", encoding="utf-8") as fq,
        open(qrels_train_path, "w", encoding="utf-8") as ft,
        open(qrels_test_path, "w", encoding="utf-8") as fte,
    ):
        ft.write("query_id\tdoc_id\tscore\n")
        fte.write("query_id\tdoc_id\tscore\n")

        for idx, q in enumerate(all_questions):
            is_train = idx < len(train_questions)
            qid = f"q{q['id']}"
            query_entry = {
                "_id": qid,
                "text": q["question"],
                "metadata": {
                    "category": q.get("category", ""),
                    "subcategory": q.get("subcategory", ""),
                },
            }
            fq.write(json.dumps(query_entry, ensure_ascii=False) + "\n")

            # parse article_ids (comma-separated string)
            article_ids_raw = q["article_ids"]
            if isinstance(article_ids_raw, str):
                doc_ids = [aid.strip() for aid in article_ids_raw.split(",") if aid.strip()]
            elif isinstance(article_ids_raw, list):
                doc_ids = [str(aid) for aid in article_ids_raw]
            else:
                doc_ids = []

            qrels_file = ft if is_train else fte
            for doc_id in doc_ids:
                if doc_id not in corpus_ids:
                    n_missing_docs += 1
                    continue
                qrels_file.write(f"{qid}\t{doc_id}\t1\n")
                if is_train:
                    n_train_judgments += 1
                else:
                    n_test_judgments += 1

    print(f"Wrote {len(all_questions)} queries to {queries_path}")
    print(f"Train judgments: {n_train_judgments}")
    print(f"Test judgments: {n_test_judgments}")
    if n_missing_docs:
        print(f"Missing doc references: {n_missing_docs}")

    # --- dataset_stats.json ---
    total_judgments = n_train_judgments + n_test_judgments
    n_queries = len(all_questions)

    stats = {
        "dataset": "bsard",
        "language": "fr",
        "num_documents": len(corpus),
        "num_queries": n_queries,
        "num_train_queries": len(train_questions),
        "num_test_queries": len(test_questions),
        "num_relevance_judgments": total_judgments,
        "num_train_judgments": n_train_judgments,
        "num_test_judgments": n_test_judgments,
        "avg_relevant_docs_per_query": total_judgments / n_queries if n_queries > 0 else 0,
    }

    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
