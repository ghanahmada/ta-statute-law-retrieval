import json
import os
import subprocess

OUTPUT_DIR = os.path.join("data", "stard")
STARD_DIR = os.path.join("data", "_raw", "STARD")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # clone repo if needed
    if not os.path.exists(STARD_DIR):
        os.makedirs(os.path.join("data", "_raw"), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/oneal2000/STARD.git", STARD_DIR],
            check=True,
        )
        print("Cloned STARD repo")
    else:
        print(f"STARD already exists at {STARD_DIR}")

    # load queries.json
    with open(os.path.join(STARD_DIR, "data", "queries.json"), encoding="utf-8") as f:
        queries_raw = json.load(f)
    print(f"Loaded {len(queries_raw)} queries")

    # load corpus.jsonl
    corpus_raw = []
    with open(os.path.join(STARD_DIR, "data", "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            corpus_raw.append(json.loads(line))
    print(f"Loaded {len(corpus_raw)} corpus documents")

    # load train/dev split IDs
    def load_query_ids(filepath):
        ids = set()
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    ids.add(int(parts[0]))
        return ids

    example_dir = os.path.join(STARD_DIR, "data", "example")
    train_ids = load_query_ids(os.path.join(example_dir, "train.query.txt"))
    dev_ids = load_query_ids(os.path.join(example_dir, "dev.query.txt"))
    print(f"Train IDs: {len(train_ids)}, Dev IDs (→ test): {len(dev_ids)}")

    # --- corpus.jsonl ---
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus_raw:
            entry = {
                "_id": str(doc["id"]),
                "title": doc["name"],
                "text": doc["content"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus_raw)} documents to {corpus_path}")

    # --- queries.jsonl + qrels ---
    corpus_ids = {str(doc["id"]) for doc in corpus_raw}
    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    qrels_train_path = os.path.join(OUTPUT_DIR, "qrels_train.tsv")
    qrels_test_path = os.path.join(OUTPUT_DIR, "qrels_test.tsv")

    n_train_judgments = 0
    n_test_judgments = 0
    n_missing_docs = 0
    n_unassigned = 0

    QUESTION_KEY = "\u95ee\u9898"  # 问题

    with (
        open(queries_path, "w", encoding="utf-8") as fq,
        open(qrels_train_path, "w", encoding="utf-8") as ft,
        open(qrels_test_path, "w", encoding="utf-8") as fte,
    ):
        ft.write("query_id\tdoc_id\tscore\n")
        fte.write("query_id\tdoc_id\tscore\n")

        for q in queries_raw:
            qid_num = q["query_id"]
            qid = f"q{qid_num}"

            query_entry = {
                "_id": qid,
                "text": q[QUESTION_KEY],
                "metadata": {},
            }
            fq.write(json.dumps(query_entry, ensure_ascii=False) + "\n")

            if qid_num in train_ids:
                qrels_file = ft
                is_train = True
            elif qid_num in dev_ids:
                qrels_file = fte
                is_train = False
            else:
                qrels_file = ft
                is_train = True
                n_unassigned += 1

            for doc_id in q.get("match_id", []):
                doc_id_str = str(doc_id)
                if doc_id_str not in corpus_ids:
                    n_missing_docs += 1
                    continue
                qrels_file.write(f"{qid}\t{doc_id_str}\t1\n")
                if is_train:
                    n_train_judgments += 1
                else:
                    n_test_judgments += 1

    print(f"Wrote {len(queries_raw)} queries to {queries_path}")
    print(f"Train judgments: {n_train_judgments}")
    print(f"Test judgments: {n_test_judgments}")
    if n_missing_docs:
        print(f"Missing doc references: {n_missing_docs}")
    if n_unassigned:
        print(f"Unassigned queries (added to train): {n_unassigned}")

    # --- dataset_stats.json ---
    total_judgments = n_train_judgments + n_test_judgments
    n_queries = len(queries_raw)

    stats = {
        "dataset": "stard",
        "language": "zh",
        "num_documents": len(corpus_raw),
        "num_queries": n_queries,
        "num_train_queries": len(train_ids) + n_unassigned,
        "num_test_queries": len(dev_ids),
        "num_relevance_judgments": total_judgments,
        "num_train_judgments": n_train_judgments,
        "num_test_judgments": n_test_judgments,
        "avg_relevant_docs_per_query": total_judgments / n_queries if n_queries > 0 else 0,
        "note": "dev split used as test; queries not in train/dev assigned to train",
    }

    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
