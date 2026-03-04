"""Extend SAILER_en vocabulary with frequent Indonesian tokens from training data.

Leakage-safe: only uses corpus + train queries (from qrels_train.tsv).
Test queries are never used.

New token embeddings are initialized as the mean of their existing subword embeddings,
giving the model a warm start for Indonesian legal terms.

Usage:
    python src/scripts/sailer/extend_vocab.py
    python src/scripts/sailer/extend_vocab.py --max_new_tokens 10000 --min_freq 3
"""

import argparse
import json
import re
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModel

BASE_MODEL = "CSHaitao/SAILER_en"
OUTPUT_DIR = "./outputs/sailer_en_extended"

CORPUS_PATH = "data/kuhperdata/corpus.jsonl"
QUERIES_PATH = "data/kuhperdata/queries.jsonl"
QRELS_TRAIN_PATH = "data/kuhperdata/qrels_train.tsv"
QRELS_TEST_PATH = "data/kuhperdata/qrels_test.tsv"


def load_train_query_ids(qrels_train_path: str, qrels_test_path: str) -> set[str]:
    """Load train query IDs and verify no overlap with test set."""
    train_ids = set()
    with open(qrels_train_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                train_ids.add(parts[0])

    test_ids = set()
    with open(qrels_test_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                test_ids.add(parts[0])

    overlap = train_ids & test_ids
    assert not overlap, f"Leakage detected: {len(overlap)} query IDs in both train and test!"

    print(f"Train query IDs: {len(train_ids)}")
    print(f"Test query IDs:  {len(test_ids)} (excluded from vocab extraction)")
    return train_ids


def collect_word_frequencies(
    corpus_path: str,
    queries_path: str,
    train_query_ids: set[str],
) -> Counter:
    """Collect word frequencies from corpus + train queries only."""
    word_re = re.compile(r"[a-zA-Z\u00C0-\u024F]+")  # latin + extended latin (Indonesian)
    counter: Counter = Counter()

    # All corpus documents
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            words = word_re.findall(doc.get("text", "").lower())
            counter.update(words)

    # Train queries only
    n_train_queries = 0
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in train_query_ids:
                words = word_re.findall(q.get("text", "").lower())
                counter.update(words)
                n_train_queries += 1

    print(f"Collected frequencies from {n_train_queries} train queries + all corpus docs")
    print(f"Unique words found: {len(counter)}")
    return counter


def select_new_tokens(
    word_freq: Counter,
    tokenizer,
    min_freq: int,
    max_new_tokens: int,
    min_subwords: int = 3,
) -> list[str]:
    """Select words that are frequent but poorly covered by existing tokenizer."""
    candidates = []
    for word, freq in word_freq.most_common():
        if freq < min_freq:
            break
        # Skip if already a single token in vocabulary
        tokens = tokenizer.tokenize(word)
        if len(tokens) >= min_subwords:
            candidates.append((word, freq, len(tokens)))

    candidates = candidates[:max_new_tokens]
    new_tokens = [w for w, _, _ in candidates]

    if candidates:
        avg_subwords_before = sum(n for _, _, n in candidates) / len(candidates)
        print(f"\nTop 10 candidates (word, freq, #subwords before):")
        for word, freq, n in candidates[:10]:
            print(f"  {word:30s} freq={freq:5d}  subwords={n}")
        print(f"\nAvg subwords before extension: {avg_subwords_before:.2f}")

    return new_tokens


def precompute_subword_ids(
    new_tokens: list[str],
    tokenizer,
) -> dict[str, list[int]]:
    """Pre-compute subword IDs for each new token BEFORE adding them to the tokenizer.

    Must be called before tokenizer.add_tokens() so the tokenizer still fragments
    these words into existing subwords rather than returning the new token ID.
    """
    subword_map = {}
    for token in new_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        subword_map[token] = ids
    return subword_map


def init_new_embeddings(
    model,
    new_tokens: list[str],
    old_vocab_size: int,
    subword_map: dict[str, list[int]],
) -> None:
    """Initialize new token embeddings as mean of their pre-computed subword embeddings."""
    embeddings = model.embeddings.word_embeddings.weight.data

    initialized = 0
    for i, token in enumerate(new_tokens):
        new_idx = old_vocab_size + i
        orig_ids = [sid for sid in subword_map[token] if sid < old_vocab_size]
        if orig_ids:
            embeddings[new_idx] = embeddings[orig_ids].mean(dim=0)
            initialized += 1

    print(f"Initialized {initialized}/{len(new_tokens)} new embeddings from subword means")
    print(f"Remaining {len(new_tokens) - initialized} initialized randomly (no subword coverage)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--min_subwords", type=int, default=3,
                        help="Min subwords a word must fragment into to be a candidate")
    args = parser.parse_args()

    # --- Leakage-safe data loading ---
    print("=" * 60)
    print("Step 1: Loading train/test split (leakage check)")
    print("=" * 60)
    train_query_ids = load_train_query_ids(QRELS_TRAIN_PATH, QRELS_TEST_PATH)

    # --- Word frequency collection ---
    print("\n" + "=" * 60)
    print("Step 2: Collecting word frequencies")
    print("=" * 60)
    word_freq = collect_word_frequencies(CORPUS_PATH, QUERIES_PATH, train_query_ids)

    # --- Load tokenizer ---
    print("\n" + "=" * 60)
    print(f"Step 3: Loading tokenizer from {args.base_model}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    old_vocab_size = len(tokenizer)
    print(f"Original vocab size: {old_vocab_size}")

    # --- Select new tokens ---
    print("\n" + "=" * 60)
    print("Step 4: Selecting new tokens")
    print("=" * 60)
    new_tokens = select_new_tokens(
        word_freq, tokenizer,
        min_freq=args.min_freq,
        max_new_tokens=args.max_new_tokens,
        min_subwords=args.min_subwords,
    )
    print(f"\nSelected {len(new_tokens)} new tokens to add")

    if not new_tokens:
        print("No new tokens to add. Exiting.")
        return

    # --- Pre-compute subword IDs BEFORE modifying the tokenizer ---
    print("\nPre-computing subword IDs for new tokens...")
    subword_map = precompute_subword_ids(new_tokens, tokenizer)

    # --- Add tokens to tokenizer ---
    n_added = tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)
    print(f"Added {n_added} tokens. New vocab size: {new_vocab_size}")

    # Verify: avg fragmentation after
    sample = new_tokens[:100]
    avg_after = sum(len(tokenizer.tokenize(w)) for w in sample) / len(sample)
    print(f"Avg subwords after extension (sample of 100): {avg_after:.2f} (target: 1.0)")

    # --- Load model and resize embeddings ---
    print("\n" + "=" * 60)
    print("Step 5: Loading model and resizing embeddings")
    print("=" * 60)
    model = AutoModel.from_pretrained(args.base_model)
    model.resize_token_embeddings(new_vocab_size)

    # --- Initialize new embeddings from subword means ---
    print("\n" + "=" * 60)
    print("Step 6: Initializing new embeddings from subword means")
    print("=" * 60)
    init_new_embeddings(model, new_tokens, old_vocab_size, subword_map)

    # --- Save ---
    print("\n" + "=" * 60)
    print(f"Step 7: Saving to {args.output_dir}")
    print("=" * 60)
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    print(f"\nDone. Extended model saved to {args.output_dir}")
    print(f"  Original vocab: {old_vocab_size}")
    print(f"  New vocab:      {new_vocab_size} (+{new_vocab_size - old_vocab_size})")
    print(f"\nNext: bash src/scripts/sailer/run_finetune.sh")


if __name__ == "__main__":
    main()
