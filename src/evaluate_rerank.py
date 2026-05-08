"""
Evaluate BM25 + Reranker baseline (no graph expansion).

This is the critical ablation for GAR/QUAM: isolate the reranker's contribution
from the graph expansion. Same BM25 pool, same scorer, no corpus graph.

Pipeline: BM25 top-N → reranker scores all N docs → re-sort by reranker score

Comparison:
  BM25 alone          (evaluate_bm25.py)
  BM25 + Reranker     (this script)       ← isolates reranker contribution
  BM25 + GAR          (evaluate_gar.py)   ← isolates graph contribution
  BM25 + QUAM         (evaluate_quam.py)

Scorer variants:
  - monot5:   castorini/monot5-base-msmarco       (English-only)
  - mt5:      unicamp-dl/mt5-base-mmarco-v2       (14 languages incl. id/fr/en/zh)
  - bge:      BAAI/bge-reranker-v2-m3             (100+ languages)

Usage:
  python src/evaluate_rerank.py --dataset kuhperdata
  python src/evaluate_rerank.py --dataset kuhperdata --scorer bge --pool_size 200
  python src/evaluate_rerank.py --dataset all
"""

import argparse
import time
import jieba
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

SCORERS = {
    "monot5": ("seq2seq", "castorini/monot5-base-msmarco"),
    "mt5":    ("seq2seq", "unicamp-dl/mt5-base-mmarco-v2"),
    "bge":    ("cross_encoder", "BAAI/bge-reranker-v2-m3"),
}

DATASET_DEFAULTS = {
    "kuhperdata-humanized": {
        "scorer": "bge",
        "pool_size": 200,
        "use_stemmer": False,
        "remove_stopwords": False,
    },
    "kuhperdata-summarized": {
        "scorer": "bge",
        "pool_size": 200,
        "use_stemmer": False,
        "remove_stopwords": False,
    },
}

_GLOBAL_DEFAULTS = {
    "scorer": "bge",
    "pool_size": 200,
    "use_stemmer": False,
    "remove_stopwords": False,
}


def tokenize_chinese(texts: list[str]) -> list[str]:
    jieba.setLogLevel(20)
    return [" ".join(jieba.cut(t)) for t in texts]


# ---------------------------------------------------------------------------
# Scorers (same as evaluate_gar.py)
# ---------------------------------------------------------------------------

class MonoT5Scorer:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.true_id = self.tokenizer.convert_tokens_to_ids("▁true")
        self.false_id = self.tokenizer.convert_tokens_to_ids("▁false")
        if self.true_id == self.tokenizer.unk_token_id:
            self.true_id = self.tokenizer.convert_tokens_to_ids("true")
        if self.false_id == self.tokenizer.unk_token_id:
            self.false_id = self.tokenizer.convert_tokens_to_ids("false")

    def score_pairs(self, queries: List[str], documents: List[str],
                    batch_size: Optional[int] = None) -> List[float]:
        prompts = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(queries, documents)]
        bs = batch_size or self.batch_size
        all_scores = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                decoder_input_ids = torch.full(
                    (len(batch), 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long, device=self.device,
                )
                outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                tf_logits = logits[:, [self.true_id, self.false_id]]
                probs = torch.softmax(tf_logits, dim=-1)
                all_scores.extend(probs[:, 0].cpu().tolist())
        return all_scores

    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class CrossEncoderScorer:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.batch_size = batch_size
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading {model_name}...")
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, queries: List[str], documents: List[str],
                    batch_size: Optional[int] = None) -> List[float]:
        pairs = [[q, d] for q, d in zip(queries, documents)]
        bs = batch_size or self.batch_size
        scores = self.model.predict(pairs, batch_size=bs, show_progress_bar=True)
        return scores.tolist()

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


def build_scorer(scorer_name: str, batch_size: int):
    kind, model_name = SCORERS[scorer_name]
    if kind == "seq2seq":
        return MonoT5Scorer(model_name, batch_size=batch_size)
    elif kind == "cross_encoder":
        return CrossEncoderScorer(model_name, batch_size=batch_size)
    raise ValueError(f"Unknown scorer kind: {kind}")


# ---------------------------------------------------------------------------
# BM25 pool
# ---------------------------------------------------------------------------

def get_bm25_pools(loader: DataLoader, top_n: int, lang: str,
                   b: float = 0.75, k1: float = 1.5, n_gram: int = 1,
                   use_stemmer: bool = False, use_stopwords: bool = False,
                   ) -> Dict[str, List[Tuple[str, float]]]:
    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()
    if lang == "zh":
        doc_texts = tokenize_chinese(doc_texts)
        query_texts = tokenize_chinese(query_texts)
    bm25 = BM25(b=b, k1=k1, n_gram=n_gram, lang=lang,
                use_stemmer=use_stemmer, use_stopwords=use_stopwords)
    bm25.fit(doc_texts)
    eval_qids = set(loader.qrels.keys())
    pools = {}
    for qid, query in zip(query_ids, query_texts):
        if qid not in eval_qids:
            continue
        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        pools[qid] = [(doc_ids[idx], float(scores[idx])) for idx in ranked_indices]
    return pools


# ---------------------------------------------------------------------------
# Rerank (no graph)
# ---------------------------------------------------------------------------

def rerank_pools(
    pools: Dict[str, List[Tuple[str, float]]],
    loader: DataLoader,
    scorer,
    batch_size: int,
    top_k: int,
) -> Dict[str, List[str]]:
    """Score all BM25 pool docs with the reranker, return top-k per query."""
    # Flatten all (query, doc) pairs
    qids, query_texts, doc_ids_flat, doc_texts_flat = [], [], [], []
    for qid, pool in pools.items():
        query_text = loader.queries[qid]["text"]
        for doc_id, _ in pool:
            qids.append(qid)
            query_texts.append(query_text)
            doc_ids_flat.append(doc_id)
            doc_texts_flat.append(loader.corpus[doc_id]["text"])

    total = len(qids)
    n_queries = len(pools)
    print(f"  Scoring {total:,} pairs ({n_queries} queries x ~{total // n_queries} docs) "
          f"in batches of {batch_size}...")

    t0 = time.time()
    all_scores = scorer.score_pairs(query_texts, doc_texts_flat, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Scoring done in {elapsed:.1f}s ({total / elapsed:.0f} pairs/s)")

    # Rebuild per-query rankings sorted by reranker score
    per_query: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in pools}
    for qid, doc_id, score in zip(qids, doc_ids_flat, all_scores):
        per_query[qid].append((doc_id, score))

    rankings = {}
    all_doc_scores = {}
    save_k = max(top_k, 100)
    for qid, scored_docs in per_query.items():
        scored_docs.sort(key=lambda x: -x[1])
        rankings[qid] = [doc_id for doc_id, _ in scored_docs[:save_k]]
        all_doc_scores[qid] = {doc_id: float(score) for doc_id, score in scored_docs[:save_k]}

    return rankings, all_doc_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_args(args):
    if args.dataset != "all":
        dset = DATASET_DEFAULTS.get(args.dataset, {})
    else:
        dset = {}

    def pick(attr, key=None):
        key = key or attr
        val = getattr(args, attr)
        if val is None:
            val = dset.get(key, _GLOBAL_DEFAULTS[key])
        setattr(args, attr, val)

    pick("scorer")
    pick("pool_size")
    pick("use_stemmer")
    pick("remove_stopwords")


def main():
    parser = argparse.ArgumentParser(description="BM25 + Reranker baseline (no graph)")
    parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=None,
                        help="BM25 pool size to rerank (default: 200)")
    parser.add_argument("--scorer", type=str, default=None, choices=[*SCORERS],
                        help="Scorer model (default: bge)")
    parser.add_argument("--scorer_batch_size", type=int, default=256,
                        help="Batch size for scoring (no iterative loop, so use large batches)")
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_ngram", type=int, default=1)
    parser.add_argument("--use_stemmer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--remove_stopwords", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max_relevant", type=int, default=5,
                        help="Max ground-truth docs per query (queries with more are excluded)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Path to save per-query top-100 predictions as JSONL")
    args = parser.parse_args()

    _resolve_args(args)

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    scorer = build_scorer(args.scorer, batch_size=args.scorer_batch_size)

    for name, cfg in datasets.items():
        if args.dataset == "all":
            dset_defaults = DATASET_DEFAULTS.get(name, {})
            scorer_name = dset_defaults.get("scorer", _GLOBAL_DEFAULTS["scorer"])
            pool_size = dset_defaults.get("pool_size", _GLOBAL_DEFAULTS["pool_size"])
            use_stemmer = dset_defaults.get("use_stemmer", _GLOBAL_DEFAULTS["use_stemmer"])
            remove_stopwords = dset_defaults.get("remove_stopwords", _GLOBAL_DEFAULTS["remove_stopwords"])
        else:
            scorer_name = args.scorer
            pool_size = args.pool_size
            use_stemmer = args.use_stemmer
            remove_stopwords = args.remove_stopwords

        data_dir, lang = cfg["path"], cfg["lang"]
        corpus_path = f"{data_dir}/corpus.jsonl"
        queries_path = f"{data_dir}/queries.jsonl"
        qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

        print(f"\n{'=' * 60}")
        print(f"  BM25 + Reranker (no graph): {name.upper()} ({args.split} split)")
        print(f"  Scorer: {scorer_name} ({SCORERS[scorer_name][1]})")
        print(f"  Pool: {pool_size}, stemmer={use_stemmer}, stopwords={remove_stopwords}")
        print(f"{'=' * 60}")

        loader = DataLoader(corpus_path, queries_path, qrels_path).load()
        if args.max_relevant:
            before = len(loader.qrels)
            loader.filter_max_relevant(args.max_relevant)
            print(f"  Corpus: {len(loader.corpus)} docs, Queries: {len(loader.qrels)} (filtered from {before}, max_relevant={args.max_relevant})")
        else:
            print(f"  Corpus: {len(loader.corpus)} docs, Queries: {len(loader.qrels)} queries")

        # Step 1: BM25 pool
        print(f"\n  Step 1: BM25 initial pool (top-{pool_size})")
        bm25_pools = get_bm25_pools(
            loader, pool_size, lang,
            b=args.bm25_b, k1=args.bm25_k1, n_gram=args.bm25_ngram,
            use_stemmer=use_stemmer, use_stopwords=remove_stopwords,
        )

        # Step 2: Rerank with scorer (no graph)
        print(f"\n  Step 2: Rerank top-{pool_size} with {scorer_name}")
        t0 = time.time()
        rankings, all_scores = rerank_pools(
            bm25_pools, loader, scorer,
            batch_size=args.scorer_batch_size, top_k=args.top_k,
        )
        elapsed = time.time() - t0

        ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}
        results = evaluate_ranking(rankings, ground_truth, args.top_k)

        print(f"\n  MRR@{args.top_k}:       {results[f'mrr@{args.top_k}']:.4f}")
        print(f"  Recall@{args.top_k}:    {results[f'recall@{args.top_k}']:.4f}")
        print(f"  Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
        print(f"  Hit rate:       {results['hit_rate']:.4f}")
        print(f"  N queries:      {results['n_queries']}")
        print(f"  Time (rerank):  {elapsed:.1f}s")

        if args.save_predictions:
            pred_path = args.save_predictions.format(dataset=name)
            save_predictions(
                rankings, ground_truth, pred_path,
                method="rerank", dataset=name, scores=all_scores,
            )

    scorer.cleanup()


if __name__ == "__main__":
    main()
