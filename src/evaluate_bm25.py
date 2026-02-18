import jieba
import argparse
import numpy as np

from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking

DATASETS = {
    "kuhperdata": {"path": "data/kuhperdata", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}


def tokenize_chinese(texts: list[str]) -> list[str]:
    jieba.setLogLevel(20)  # suppress jieba loading logs
    return [" ".join(jieba.cut(t)) for t in texts]


def run_bm25(loader: DataLoader, top_k: int, lang: str = "en", b: float = 0.75, k1: float = 1.5, n_gram: int = 1):
    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()

    if lang == "zh":
        print("  Tokenizing Chinese text with jieba...")
        doc_texts = tokenize_chinese(doc_texts)
        query_texts = tokenize_chinese(query_texts)

    bm25 = BM25(b=b, k1=k1, n_gram=n_gram)
    bm25.fit(doc_texts)

    rankings = {}
    for qid, query in zip(query_ids, query_texts):
        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        rankings[qid] = [doc_ids[idx] for idx in ranked_indices]

    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}

    return evaluate_ranking(rankings, ground_truth, top_k)


def main():
    parser = argparse.ArgumentParser(description="BM25 evaluation on statute retrieval datasets")
    parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--n_gram", type=int, default=1)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, cfg in datasets.items():
        data_dir, lang = cfg["path"], cfg["lang"]
        corpus_path = f"{data_dir}/corpus.jsonl"
        queries_path = f"{data_dir}/queries.jsonl"
        qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

        print(f"\n{'=' * 60}")
        print(f"  {name.upper()} ({args.split} split)")
        print(f"{'=' * 60}")

        loader = DataLoader(corpus_path, queries_path, qrels_path).load()
        print(f"  Corpus: {len(loader.corpus)} docs, Queries: {len(loader.qrels)} queries")

        results = run_bm25(loader, args.top_k, lang, args.bm25_b, args.bm25_k1, args.n_gram)

        print(f"\n  MRR@{args.top_k}:       {results[f'mrr@{args.top_k}']:.4f}")
        print(f"  Recall@{args.top_k}:    {results[f'recall@{args.top_k}']:.4f}")
        print(f"  Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
        print(f"  Hit rate:       {results['hit_rate']:.4f}")
        print(f"  N queries:      {results['n_queries']}")


if __name__ == "__main__":
    main()
