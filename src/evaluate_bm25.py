import jieba
import argparse
import numpy as np

from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking, save_predictions

DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "kuhperdata-exp": {"path": "data/kuhperdata-exp", "lang": "id"},
    "kuhperdata-summ-exp": {"path": "data/kuhperdata-summ-exp", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}

DATASET_DEFAULTS = {
    "kuhperdata-humanized": {"use_stemmer": False, "remove_stopwords": False},
    "kuhperdata-summarized": {"use_stemmer": False, "remove_stopwords": False},
}


def tokenize_chinese(texts: list[str]) -> list[str]:
    jieba.setLogLevel(20)  # suppress jieba loading logs
    return [" ".join(jieba.cut(t)) for t in texts]


def run_bm25(loader: DataLoader, top_k: int, lang: str = "en",
             b: float = 0.75, k1: float = 1.5, n_gram: int = 1,
             use_stemmer: bool = False, remove_stopwords: bool = False,
             save_top_k: int = 100):
    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()

    if lang == "zh":
        print("  Tokenizing Chinese text with jieba...")
        doc_texts = tokenize_chinese(doc_texts)
        query_texts = tokenize_chinese(query_texts)

    bm25 = BM25(b=b, k1=k1, n_gram=n_gram, lang=lang,
                use_stemmer=use_stemmer, use_stopwords=remove_stopwords)
    bm25.fit(doc_texts)

    rankings = {}
    all_scores = {}
    k = max(top_k, save_top_k)
    for qid, query in zip(query_ids, query_texts):
        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1][:k]
        rankings[qid] = [doc_ids[idx] for idx in ranked_indices]
        all_scores[qid] = {doc_ids[idx]: float(scores[idx]) for idx in ranked_indices}

    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}

    results = evaluate_ranking(rankings, ground_truth, top_k)
    results["_rankings"] = rankings
    results["_scores"] = all_scores
    results["_ground_truth"] = ground_truth
    return results


def main():
    parser = argparse.ArgumentParser(description="BM25 evaluation on statute retrieval datasets")
    parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--n_gram", type=int, default=1)
    parser.add_argument("--use_stemmer", action=argparse.BooleanOptionalAction, default=None,
                        help="Indonesian stemmer via PySastrawi (default: on for kuhperdata)")
    parser.add_argument("--remove_stopwords", action=argparse.BooleanOptionalAction, default=None,
                        help="Remove Indonesian stopwords via PySastrawi (default: on for kuhperdata)")
    parser.add_argument("--max_relevant", type=int, default=5,
                        help="Max ground-truth docs per query (queries with more are excluded)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Path to save per-query top-100 predictions as JSONL")
    args = parser.parse_args()

    datasets = DATASET_DEFAULTS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, cfg in (DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}).items():
        data_dir, lang = cfg["path"], cfg["lang"]
        corpus_path = f"{data_dir}/corpus.jsonl"
        queries_path = f"{data_dir}/queries.jsonl"
        qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

        # Apply dataset defaults for unset flags
        dset_defaults = DATASET_DEFAULTS.get(name, {})
        use_stemmer = args.use_stemmer if args.use_stemmer is not None else dset_defaults.get("use_stemmer", False)
        remove_stopwords = args.remove_stopwords if args.remove_stopwords is not None else dset_defaults.get("remove_stopwords", False)

        print(f"\n{'=' * 60}")
        print(f"  {name.upper()} ({args.split} split)")
        print(f"  BM25 | stemmer={use_stemmer} stopwords={remove_stopwords}")
        print(f"{'=' * 60}")

        loader = DataLoader(corpus_path, queries_path, qrels_path).load()
        if args.max_relevant:
            before = len(loader.qrels)
            loader.filter_max_relevant(args.max_relevant)
            print(f"  Corpus: {len(loader.corpus)} docs, Queries: {len(loader.qrels)} (filtered from {before}, max_relevant={args.max_relevant})")
        else:
            print(f"  Corpus: {len(loader.corpus)} docs, Queries: {len(loader.qrels)} queries")

        results = run_bm25(loader, args.top_k, lang, args.bm25_b, args.bm25_k1, args.n_gram,
                           use_stemmer=use_stemmer, remove_stopwords=remove_stopwords)

        print(f"\n  MRR@{args.top_k}:       {results[f'mrr@{args.top_k}']:.4f}")
        print(f"  Recall@{args.top_k}:    {results[f'recall@{args.top_k}']:.4f}")
        print(f"  Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
        print(f"  Hit rate:       {results['hit_rate']:.4f}")
        print(f"  N queries:      {results['n_queries']}")

        save_predictions(
            results["_rankings"], results["_ground_truth"],
            args.save_predictions.format(dataset=name) if args.save_predictions else None,
            method="bm25", dataset=name, scores=results["_scores"],
        )


if __name__ == "__main__":
    main()
