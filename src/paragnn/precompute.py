"""Pre-computation for Para-GNN: BM25 scores, paragraph embeddings, RR constants.

Usage:
  python -m paragnn.precompute --dataset kuhperdata-humanized
  python -m paragnn.precompute --dataset all
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paragnn import DATASETS, RR_LABELS, FACT_TYPES, ParaGNNConfig
from util.dataloader import DataLoader
from util.bm25 import BM25


def split_into_sentences(text: str, lang: str = "en") -> list[str]:
    """Split text into sentences using simple regex (language-aware)."""
    if lang == "zh":
        # Chinese: split on Chinese punctuation
        sents = re.split(r'[。！？\n]+', text)
    else:
        # ID/FR/EN: split on period, exclamation, question mark followed by space or newline
        sents = re.split(r'(?<=[.!?;])\s+|\n+', text)

    sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 10]
    # If no split found (very short text), return whole text as single sentence
    if not sents:
        sents = [text.strip()]
    return sents


def precompute_bm25_scores(config: ParaGNNConfig):
    """Compute BM25 score matrices for train and test queries."""
    output_dir = f"{config.output_dir}/{config.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_loader = DataLoader(
        f"{config.data_path}/corpus.jsonl",
        f"{config.data_path}/queries.jsonl",
        f"{config.data_path}/qrels_train.tsv",
    ).load()
    test_loader = DataLoader(
        f"{config.data_path}/corpus.jsonl",
        f"{config.data_path}/queries.jsonl",
        f"{config.data_path}/qrels_test.tsv",
    ).load()

    if config.max_relevant > 0:
        train_loader.filter_max_relevant(config.max_relevant)
        test_loader.filter_max_relevant(config.max_relevant)

    doc_ids, doc_texts = train_loader.get_corpus_texts()

    # Handle Chinese tokenization
    if config.lang == "zh":
        import jieba
        jieba.setLogLevel(20)
        bm25_texts = [" ".join(jieba.cut(t)) for t in doc_texts]
    else:
        bm25_texts = doc_texts

    bm25 = BM25(b=config.bm25_b, k1=config.bm25_k1, n_gram=config.bm25_ngram,
                 lang=config.lang, use_stemmer=False, use_stopwords=False)
    bm25.fit(bm25_texts)

    # Save doc_ids ordering (needed for graph construction)
    with open(f"{output_dir}/corpus_doc_ids.json", "w") as f:
        json.dump(doc_ids, f)

    # Train BM25 scores
    train_qids = sorted(train_loader.qrels.keys())
    print(f"Computing BM25 for {len(train_qids)} train queries...")
    train_scores = []
    for qid in tqdm(train_qids, desc="BM25 train"):
        q_text = train_loader.queries[qid]["text"]
        if config.lang == "zh":
            q_text = " ".join(jieba.cut(q_text))
        train_scores.append(bm25.transform(q_text))
    train_scores = torch.tensor(np.array(train_scores), dtype=torch.float32)
    torch.save(train_scores, f"{output_dir}/bm25_train_scores.pt")
    print(f"  Saved: {output_dir}/bm25_train_scores.pt, shape={train_scores.shape}")

    # Save train query IDs ordering
    with open(f"{output_dir}/train_query_ids.json", "w") as f:
        json.dump(train_qids, f)

    # Test BM25 scores
    test_qids = sorted(test_loader.qrels.keys())
    print(f"Computing BM25 for {len(test_qids)} test queries...")
    test_scores = []
    for qid in tqdm(test_qids, desc="BM25 test"):
        q_text = test_loader.queries[qid]["text"]
        if config.lang == "zh":
            q_text = " ".join(jieba.cut(q_text))
        test_scores.append(bm25.transform(q_text))
    test_scores = torch.tensor(np.array(test_scores), dtype=torch.float32)
    torch.save(test_scores, f"{output_dir}/bm25_test_scores.pt")
    print(f"  Saved: {output_dir}/bm25_test_scores.pt, shape={test_scores.shape}")

    with open(f"{output_dir}/test_query_ids.json", "w") as f:
        json.dump(test_qids, f)

    # Also save BM25 hard negative rankings for training
    print("Computing BM25 hard negative rankings...")
    hard_negs = {}
    for i, qid in enumerate(train_qids):
        ranked_idx = np.argsort(train_scores[i].numpy())[::-1]
        hard_negs[qid] = [doc_ids[idx] for idx in ranked_idx[:400]]
    with open(f"{output_dir}/bm25_hard_negatives.json", "w") as f:
        json.dump(hard_negs, f)
    print(f"  Saved: {output_dir}/bm25_hard_negatives.json")


def precompute_paragraph_embeddings(config: ParaGNNConfig):
    """Encode each document/query at the sentence level with BGE-M3."""
    output_dir = f"{config.output_dir}/{config.dataset}"
    emb_dir = f"{output_dir}/embeddings"
    os.makedirs(f"{emb_dir}/corpus", exist_ok=True)
    os.makedirs(f"{emb_dir}/queries", exist_ok=True)

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(config.bge_model_name, use_fp16=True)

    # Load corpus
    loader = DataLoader(
        f"{config.data_path}/corpus.jsonl",
        f"{config.data_path}/queries.jsonl",
        f"{config.data_path}/qrels_train.tsv",
    ).load()

    # Encode corpus paragraphs (statutes split into sentences)
    print(f"Encoding corpus paragraph embeddings ({len(loader.corpus)} docs)...")
    para_counts = []
    for doc_id, doc in tqdm(loader.corpus.items(), desc="Corpus embeddings"):
        emb_path = f"{emb_dir}/corpus/{doc_id}.pt"
        if os.path.exists(emb_path):
            para_counts.append(torch.load(emb_path).shape[0])
            continue

        sentences = split_into_sentences(doc["text"], config.lang)
        embeddings = model.encode(sentences, batch_size=32, max_length=config.encode_max_length)
        emb_tensor = torch.tensor(embeddings["dense_vecs"], dtype=torch.float32)
        torch.save(emb_tensor, emb_path)
        para_counts.append(len(sentences))

    print(f"  Avg paragraphs per statute: {np.mean(para_counts):.1f}")

    # Load RR labels if method=full (to use LLM sentence splits for encoding)
    rr_labels = None
    if config.method == "full":
        rr_path = f"{output_dir}/rr_labels.json"
        if os.path.exists(rr_path):
            with open(rr_path, "r", encoding="utf-8") as f:
                rr_labels = json.load(f)
            print(f"  Loaded RR labels for {len(rr_labels)} queries")
        else:
            print(f"  WARNING: RR labels not found at {rr_path}, falling back to sentence splitting")

    # Encode query paragraphs (sentences)
    print(f"Encoding query paragraph embeddings ({len(loader.queries)} queries)...")
    para_counts = []
    for qid, query in tqdm(loader.queries.items(), desc="Query embeddings"):
        emb_path = f"{emb_dir}/queries/{qid}.pt"

        if config.method == "adapted":
            sentences = [query["text"]]
        elif rr_labels and qid in rr_labels:
            # Method 1: use LLM sentence splits
            sentences = [p["sentence"] for p in rr_labels[qid] if isinstance(p, dict) and "sentence" in p]
            if not sentences:
                sentences = [query["text"]]
        else:
            sentences = split_into_sentences(query["text"], config.lang)

        # Re-encode if paragraph count doesn't match (e.g., switching from adapted to full)
        if os.path.exists(emb_path):
            existing = torch.load(emb_path)
            if existing.shape[0] == len(sentences):
                para_counts.append(len(sentences))
                continue

        embeddings = model.encode(sentences, batch_size=32, max_length=config.encode_max_length)
        emb_tensor = torch.tensor(embeddings["dense_vecs"], dtype=torch.float32)
        torch.save(emb_tensor, emb_path)
        para_counts.append(len(sentences))

    print(f"  Avg paragraphs per query: {np.mean(para_counts):.1f}")

    del model  # Free GPU

    # Encode RR constant label strings
    print("Encoding RR label embeddings...")
    model = BGEM3FlagModel(config.bge_model_name, use_fp16=True)
    rr_output = model.encode(RR_LABELS, batch_size=32, max_length=64)
    rr_tensor = torch.tensor(rr_output["dense_vecs"], dtype=torch.float32)
    torch.save(rr_tensor, f"{emb_dir}/EMBD_CONST.pt")
    print(f"  Saved: {emb_dir}/EMBD_CONST.pt, shape={rr_tensor.shape}")

    # Encode fact type labels for query edge features
    print("Encoding fact type embeddings...")
    ft_output = model.encode(FACT_TYPES, batch_size=32, max_length=64)
    ft_tensor = torch.tensor(ft_output["dense_vecs"], dtype=torch.float32)
    torch.save(ft_tensor, f"{emb_dir}/EMBD_FACT_TYPES.pt")
    print(f"  Saved: {emb_dir}/EMBD_FACT_TYPES.pt, shape={ft_tensor.shape}")
    del model

    # Save sentence splits for queries (needed by graph builder to get RR labels)
    print("Saving paragraph decomposition metadata...")
    query_paras = {}
    for qid, query in loader.queries.items():
        if config.method == "adapted":
            query_paras[qid] = [{"sentence": query["text"], "role": "NONE"}]
        elif rr_labels and qid in rr_labels:
            # Use LLM-labeled sentence splits directly
            query_paras[qid] = rr_labels[qid]
        else:
            sents = split_into_sentences(query["text"], config.lang)
            query_paras[qid] = [{"sentence": s, "role": "NONE"} for s in sents]

    with open(f"{output_dir}/query_paragraphs.json", "w", encoding="utf-8") as f:
        json.dump(query_paras, f, ensure_ascii=False, indent=2)

    corpus_paras = {}
    for doc_id, doc in loader.corpus.items():
        sents = split_into_sentences(doc["text"], config.lang)
        corpus_paras[doc_id] = [{"sentence": s, "role": "NONE"} for s in sents]

    with open(f"{output_dir}/corpus_paragraphs.json", "w", encoding="utf-8") as f:
        json.dump(corpus_paras, f, ensure_ascii=False, indent=2)

    print("Done!")


def precompute_fact_type_embeddings(config: ParaGNNConfig):
    """Re-encode queries using fact annotations (one embedding per fact).

    Reads fact annotations from outputs/subsumption/{dataset}/query_facts.jsonl
    and encodes each fact as a separate paragraph embedding.

    Saves to separate files so the original pipeline is untouched:
      - embeddings/queries_facts/{qid}.pt  (N facts × 1024d per query)
      - embeddings/EMBD_FACT_TYPES.pt      (5 × 1024d label embeddings)
      - query_paragraphs_facts.json        (fact texts + types as "role")
    """
    output_dir = f"{config.output_dir}/{config.dataset}"
    emb_dir = f"{output_dir}/embeddings"

    # Load fact annotations
    facts_path = f"outputs/subsumption/{config.dataset}/query_facts.jsonl"
    if not os.path.exists(facts_path):
        print(f"  Fact annotations not found: {facts_path}")
        print(f"  Run first: python experiment/annotate_subsumption.py --dataset {config.dataset} --mode queries")
        return

    query_facts = {}
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("annotation") is not None:
                query_facts[entry["id"]] = entry["annotation"]["facts"]
    print(f"  Loaded fact annotations for {len(query_facts)} queries")

    # Load original queries for fallback
    from util.dataloader import DataLoader as DL
    loader = DL(
        f"{config.data_path}/corpus.jsonl",
        f"{config.data_path}/queries.jsonl",
        f"{config.data_path}/qrels_train.tsv",
    ).load()

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(config.bge_model_name, use_fp16=True)

    # Encode fact type label strings
    print("  Encoding fact type label embeddings...")
    ft_output = model.encode(FACT_TYPES, batch_size=32, max_length=64)
    ft_tensor = torch.tensor(ft_output["dense_vecs"], dtype=torch.float32)
    torch.save(ft_tensor, f"{emb_dir}/EMBD_FACT_TYPES.pt")
    print(f"  Saved: {emb_dir}/EMBD_FACT_TYPES.pt, shape={ft_tensor.shape}")

    # Encode each query's facts as separate paragraphs
    os.makedirs(f"{emb_dir}/queries_facts", exist_ok=True)
    query_paras_facts = {}
    para_counts = []
    n_fallback = 0

    print(f"  Encoding query fact embeddings ({len(loader.queries)} queries)...")
    for qid, query in tqdm(loader.queries.items(), desc="Fact embeddings"):
        emb_path = f"{emb_dir}/queries_facts/{qid}.pt"

        if qid in query_facts and query_facts[qid]:
            facts = query_facts[qid]
            sentences = [f["text"] for f in facts]
            roles = [f.get("fact_type", "GENERAL") for f in facts]
        else:
            sentences = [query["text"]]
            roles = ["GENERAL"]
            n_fallback += 1

        if os.path.exists(emb_path):
            existing = torch.load(emb_path)
            if existing.shape[0] == len(sentences):
                para_counts.append(len(sentences))
                query_paras_facts[qid] = [
                    {"sentence": s, "role": r} for s, r in zip(sentences, roles)
                ]
                continue

        embeddings = model.encode(sentences, batch_size=32, max_length=config.encode_max_length)
        emb_tensor = torch.tensor(embeddings["dense_vecs"], dtype=torch.float32)
        torch.save(emb_tensor, emb_path)
        para_counts.append(len(sentences))

        query_paras_facts[qid] = [
            {"sentence": s, "role": r} for s, r in zip(sentences, roles)
        ]

    with open(f"{output_dir}/query_paragraphs_facts.json", "w", encoding="utf-8") as f:
        json.dump(query_paras_facts, f, ensure_ascii=False, indent=2)

    print(f"  Avg facts per query: {np.mean(para_counts):.1f}")
    print(f"  Queries with fact annotations: {len(query_facts)}, fallback to original: {n_fallback}")
    print(f"  Saved: {emb_dir}/queries_facts/ and {output_dir}/query_paragraphs_facts.json")

    del model


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Para-GNN inputs")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--method", default="adapted", choices=["full", "adapted"])
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--encode_fact_types", action="store_true",
                        help="Re-encode queries using fact annotations from annotate_subsumption.py")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Pre-computing: {name}")
        print(f"{'='*60}")

        config = ParaGNNConfig(
            dataset=name,
            data_path=cfg["path"],
            lang=cfg["lang"],
            method=args.method,
            max_relevant=args.max_relevant,
        )

        if not args.skip_bm25:
            precompute_bm25_scores(config)

        if not args.skip_embeddings:
            precompute_paragraph_embeddings(config)

        if args.encode_fact_types:
            precompute_fact_type_embeddings(config)


if __name__ == "__main__":
    main()
