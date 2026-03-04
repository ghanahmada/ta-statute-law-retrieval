"""Convert kuhperdata to SAILER encoding format for inference/evaluation.

Input:  data/kuhperdata/{corpus.jsonl, queries.jsonl}
Output: data/sailer/encode/corpus.jsonl  ({"text_id": ..., "text": ...})
        data/sailer/encode/queries.jsonl ({"text_id": ..., "text": ...})
"""

import json
import os

DATA_DIR = os.path.join("data", "kuhperdata")
OUTPUT_DIR = os.path.join("data", "sailer", "encode")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert corpus
    corpus_in = os.path.join(DATA_DIR, "corpus.jsonl")
    corpus_out = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    n_corpus = 0
    with open(corpus_in, encoding="utf-8") as fin, open(corpus_out, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            entry = {"text_id": doc["_id"], "text": doc["text"]}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_corpus += 1
    print(f"Wrote {n_corpus} documents to {corpus_out}")

    # Convert queries
    queries_in = os.path.join(DATA_DIR, "queries.jsonl")
    queries_out = os.path.join(OUTPUT_DIR, "queries.jsonl")
    n_queries = 0
    with open(queries_in, encoding="utf-8") as fin, open(queries_out, "w", encoding="utf-8") as fout:
        for line in fin:
            q = json.loads(line)
            entry = {"text_id": q["_id"], "text": q["text"]}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_queries += 1
    print(f"Wrote {n_queries} queries to {queries_out}")


if __name__ == "__main__":
    main()
