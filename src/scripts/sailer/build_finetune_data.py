"""Convert kuhperdata to SAILER finetuning format with BM25 hard negatives.

Input:  data/kuhperdata/{corpus.jsonl, queries.jsonl, qrels_train.tsv}
Output: data/sailer/finetune/train.json

Each output line: {"query": "...", "positives": ["...", ...], "negatives": ["...", ...]}
"""

import json
import os
import random
from collections import defaultdict

from rank_bm25 import BM25Okapi
from tqdm import tqdm

DATA_DIR = os.path.join("data", "kuhperdata")
OUTPUT_DIR = os.path.join("data", "sailer", "finetune")
NUM_HARD_NEGATIVES = 30
SEED = 42


def load_corpus(path: str) -> dict[str, str]:
    """Load corpus.jsonl into {doc_id: text} dict."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["text"]
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """Load queries.jsonl into {query_id: text} dict."""
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(path: str) -> dict[str, list[str]]:
    """Parse qrels TSV into {query_id: [doc_ids]} mapping."""
    qrels = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], parts[2]
                if int(score) > 0:
                    qrels[query_id].append(doc_id)
    return dict(qrels)


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization for BM25."""
    return text.lower().split()


def build_bm25_index(corpus: dict[str, str]) -> tuple[BM25Okapi, list[str]]:
    """Build BM25 index over corpus documents."""
    doc_ids = list(corpus.keys())
    tokenized_corpus = [tokenize(corpus[did]) for did in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids


def mine_hard_negatives(
    query_text: str,
    positive_ids: set[str],
    bm25: BM25Okapi,
    doc_ids: list[str],
    n: int = NUM_HARD_NEGATIVES,
) -> list[str]:
    """Get top-N BM25 results that are NOT in the positive set."""
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)

    # Sort by score descending
    scored_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

    negatives = []
    for did, score in scored_docs:
        if did not in positive_ids:
            negatives.append(did)
            if len(negatives) >= n:
                break

    return negatives


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading corpus...")
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    print(f"  {len(corpus)} documents")

    print("Loading queries...")
    queries = load_queries(os.path.join(DATA_DIR, "queries.jsonl"))
    print(f"  {len(queries)} queries")

    print("Loading train qrels...")
    qrels = load_qrels(os.path.join(DATA_DIR, "qrels_train.tsv"))
    print(f"  {len(qrels)} queries with relevance judgments")

    print("Building BM25 index...")
    bm25, doc_ids = build_bm25_index(corpus)
    all_doc_ids = set(doc_ids)

    output_path = os.path.join(OUTPUT_DIR, "train.json")
    skipped = 0

    print("Mining hard negatives and building training data...")
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in tqdm(sorted(qrels.keys()), desc="Processing queries"):
            if qid not in queries:
                skipped += 1
                continue

            query_text = queries[qid]
            positive_ids = set(qrels[qid])

            # Get positive texts
            positives = []
            for did in qrels[qid]:
                if did in corpus:
                    positives.append(corpus[did])

            if not positives:
                skipped += 1
                continue

            # Mine BM25 hard negatives
            neg_ids = mine_hard_negatives(query_text, positive_ids, bm25, doc_ids)

            # Fallback to random negatives if BM25 yields too few
            if len(neg_ids) < NUM_HARD_NEGATIVES:
                available = list(all_doc_ids - positive_ids - set(neg_ids))
                extra = random.sample(available, min(NUM_HARD_NEGATIVES - len(neg_ids), len(available)))
                neg_ids.extend(extra)

            negatives = [corpus[did] for did in neg_ids if did in corpus]

            entry = {
                "query": query_text,
                "positives": positives,
                "negatives": negatives,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_written = len(qrels) - skipped
    print(f"\nWrote {n_written} training examples to {output_path}")
    if skipped:
        print(f"  Skipped {skipped} queries (missing query text or corpus docs)")

    # Validation
    print("\nValidating output...")
    with open(output_path, encoding="utf-8") as f:
        lines = f.readlines()

    n_valid = 0
    for i, line in enumerate(lines):
        entry = json.loads(line)
        assert "query" in entry, f"Line {i}: missing 'query'"
        assert "positives" in entry and len(entry["positives"]) >= 1, f"Line {i}: need >=1 positive"
        assert "negatives" in entry and len(entry["negatives"]) >= 7, f"Line {i}: need >=7 negatives"

        # Check no overlap between positives and negatives
        pos_set = set(entry["positives"])
        neg_set = set(entry["negatives"])
        assert not pos_set & neg_set, f"Line {i}: overlap between positives and negatives"
        n_valid += 1

    print(f"  All {n_valid} examples valid!")
    print(f"  Avg positives: {sum(len(json.loads(l)['positives']) for l in lines) / len(lines):.1f}")
    print(f"  Avg negatives: {sum(len(json.loads(l)['negatives']) for l in lines) / len(lines):.1f}")


if __name__ == "__main__":
    main()
