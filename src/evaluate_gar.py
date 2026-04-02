"""
Evaluate GAR (Graph-based Adaptive Re-ranking) on statute retrieval datasets.

Methodology: MacAvaney et al., "Adaptive Re-Ranking with a Corpus Graph" (CIKM 2022)

Faithful to paper:
  - Corpus graph built from BM25 (each doc as query against corpus)
  - Initial pool from BM25
  - Scorer: monoT5 cross-encoder re-ranker

Scorer variants:
  - monot5:   castorini/monot5-base-msmarco       (English-only, original paper)
  - mt5:      unicamp-dl/mt5-base-mmarco-v2       (14 languages incl. id/fr/en/zh)
  - bge:      BAAI/bge-reranker-v2-m3             (100+ languages, best for Indonesian)

Usage:
  python src/evaluate_gar.py --dataset kuhperdata
  python src/evaluate_gar.py --dataset all --budget 100 --graph_k 16
  python src/evaluate_gar.py --dataset kuhperdata --scorer bge
  python src/evaluate_gar.py --dataset kuhperdata --budget 200 --initial_pool_size 300
"""

import argparse
import time
import jieba
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking
from gar.corpus_graph import CorpusGraph
from gar.adaptive_reranker import GAR

DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}

# (scorer_type, model_name)
SCORERS = {
    "monot5": ("seq2seq", "castorini/monot5-base-msmarco"),
    "mt5":    ("seq2seq", "unicamp-dl/mt5-base-mmarco-v2"),
    "bge":    ("cross_encoder", "BAAI/bge-reranker-v2-m3"),
}

# Per-dataset recommended defaults (override global defaults when dataset is specified)
DATASET_DEFAULTS = {
    "kuhperdata": {
        "scorer": "bge",
        "initial_pool_size": 100,
        "budget": 50,
        # PySastrawi stemmer is too aggressive for formal legal Indonesian
        # (e.g. ketentuan→tentu, peraturan→atur loses discriminative specificity).
        # Disable by default; enable with --use_stemmer / --remove_stopwords if needed.
        "use_stemmer": False,
        "remove_stopwords": False,
    },
}

_GLOBAL_DEFAULTS = {
    "scorer": "mt5",
    "initial_pool_size": 100,
    "budget": 100,
    "use_stemmer": False,
    "remove_stopwords": False,
}


def tokenize_chinese(texts: list[str]) -> list[str]:
    jieba.setLogLevel(20)
    return [" ".join(jieba.cut(t)) for t in texts]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

class MonoT5Scorer:
    """
    monoT5 / mT5 cross-encoder scorer.

    Input format:  "Query: {query} Document: {document} Relevant:"
    Output:        softmax over "true" / "false" token logits → P(relevant)
    """

    def __init__(self, model_name: str, device: str = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

        # Get token ids for "true" and "false"
        self.true_id = self.tokenizer.convert_tokens_to_ids("▁true")
        self.false_id = self.tokenizer.convert_tokens_to_ids("▁false")
        # Fallback for tokenizers that don't use sentencepiece prefix
        if self.true_id == self.tokenizer.unk_token_id:
            self.true_id = self.tokenizer.convert_tokens_to_ids("true")
        if self.false_id == self.tokenizer.unk_token_id:
            self.false_id = self.tokenizer.convert_tokens_to_ids("false")

    def _score_prompts(self, prompts: List[str], batch_size: int) -> List[float]:
        all_scores = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
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

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Score one query against a list of documents."""
        prompts = [f"Query: {query} Document: {doc} Relevant:" for doc in documents]
        return self._score_prompts(prompts, self.batch_size)

    def score_pairs(self, queries: List[str], documents: List[str],
                    batch_size: Optional[int] = None) -> List[float]:
        """Score parallel lists of queries and documents (cross-query bulk scoring)."""
        prompts = [
            f"Query: {q} Document: {d} Relevant:"
            for q, d in zip(queries, documents)
        ]
        return self._score_prompts(prompts, batch_size or self.batch_size)

    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class CrossEncoderScorer:
    """
    BERT-style cross-encoder scorer via sentence-transformers CrossEncoder.

    Returns raw logit scores (higher = more relevant).
    Works with any sentence-transformers-compatible cross-encoder model,
    including BAAI/bge-reranker-v2-m3 for multilingual / Indonesian use.
    """

    def __init__(self, model_name: str, device: str = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.batch_size = batch_size
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading {model_name}...")
        self.model = CrossEncoder(model_name, device=device)

    def _predict(self, pairs: List[List[str]], batch_size: int,
                 show_progress_bar: bool = False) -> List[float]:
        scores = self.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=show_progress_bar)
        return scores.tolist()

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Score one query against a list of documents."""
        pairs = [[query, doc] for doc in documents]
        return self._predict(pairs, self.batch_size)

    def score_pairs(self, queries: List[str], documents: List[str],
                    batch_size: Optional[int] = None) -> List[float]:
        """Score parallel lists of queries and documents (cross-query bulk scoring)."""
        pairs = [[q, d] for q, d in zip(queries, documents)]
        return self._predict(pairs, batch_size or self.batch_size, show_progress_bar=True)

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


def build_scorer(scorer_name: str, batch_size: int):
    """Instantiate the appropriate scorer by name."""
    kind, model_name = SCORERS[scorer_name]
    if kind == "seq2seq":
        return MonoT5Scorer(model_name, batch_size=batch_size)
    elif kind == "cross_encoder":
        return CrossEncoderScorer(model_name, batch_size=batch_size)
    raise ValueError(f"Unknown scorer kind: {kind}")


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def get_or_build_graph(dataset_name: str, doc_ids: List[str], doc_texts: List[str],
                       graph_k: int, lang: str, bm25_b: float, bm25_k1: float,
                       bm25_ngram: int, cache_dir: str,
                       use_stemmer: bool = False, use_stopwords: bool = False) -> CorpusGraph:
    # Include preprocessing config in cache key so different configs don't collide
    tags = f"k{graph_k}"
    if use_stemmer:
        tags += "_stemmed"
    if use_stopwords:
        tags += "_nostop"
    graph_path = Path(cache_dir) / dataset_name / f"bm25_graph_{tags}"

    if graph_path.exists():
        graph = CorpusGraph.load(graph_path)
        if len(graph.doc_ids) == len(doc_ids):
            print(f"  Loaded cached BM25 corpus graph ({tags})")
            return graph

    graph = CorpusGraph.build_from_bm25(
        doc_ids, doc_texts, k=graph_k,
        b=bm25_b, k1=bm25_k1, n_gram=bm25_ngram, lang=lang,
        use_stemmer=use_stemmer, use_stopwords=use_stopwords,
    )
    graph.save(graph_path)
    print(f"  Saved to {graph_path}")
    return graph


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
    # Only compute pools for queries that have relevance judgments —
    # queries.jsonl may contain train/dev splits not present in the qrels file.
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
# Pre-scoring
# ---------------------------------------------------------------------------

def pre_score_pools(
    pools: Dict[str, List[Tuple[str, float]]],
    loader: DataLoader,
    scorer,
    prescore_batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """
    Score all BM25 pool (query, doc) pairs in one bulk forward pass.

    Cross-encoder models treat each (query, doc) pair independently, so
    we can freely batch across queries — saturating the GPU much better
    than the per-query-per-batch GAR loop.

    Returns: {qid: {doc_id: score}}
    """
    # Flatten into parallel lists
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
    print(f"  Pre-scoring {total:,} pairs ({n_queries} queries × ~{total // n_queries} docs) "
          f"in batches of {prescore_batch_size}...")

    t0 = time.time()
    all_scores = scorer.score_pairs(query_texts, doc_texts_flat, batch_size=prescore_batch_size)
    elapsed = time.time() - t0
    print(f"  Pre-scoring done in {elapsed:.1f}s  "
          f"({total / elapsed:.0f} pairs/s)")

    # Rebuild per-query dict
    pre_scores: Dict[str, Dict[str, float]] = {qid: {} for qid in pools}
    for qid, doc_id, score in zip(qids, doc_ids_flat, all_scores):
        pre_scores[qid][doc_id] = score

    return pre_scores


# ---------------------------------------------------------------------------
# GAR runner
# ---------------------------------------------------------------------------

def make_scorer_fn(
    query_text: str,
    corpus: Dict[str, Dict],
    scorer,
    cache: Optional[Dict[str, float]] = None,
):
    """
    Create a GAR scorer closure for a single query.

    Docs present in `cache` (pre-scored BM25 pool) are returned instantly.
    Only graph-frontier docs that fall outside the pre-scored pool incur
    a live model call.
    """
    _cache = cache or {}

    def fn(batch_doc_ids: List[str]) -> Dict[str, float]:
        if not batch_doc_ids:
            return {}
        result = {did: _cache[did] for did in batch_doc_ids if did in _cache}
        uncached = [did for did in batch_doc_ids if did not in _cache]
        if uncached:
            doc_texts = [corpus[did]["text"] for did in uncached]
            live_scores = scorer.score(query_text, doc_texts)
            result.update(dict(zip(uncached, live_scores)))
        return result

    return fn


def run_gar(
    loader: DataLoader,
    graph: CorpusGraph,
    bm25_pools: Dict[str, List[Tuple[str, float]]],
    scorer,
    top_k: int,
    budget: int,
    batch_size: int,
    graph_k_limit: Optional[int] = None,
    pre_scores: Optional[Dict[str, Dict[str, float]]] = None,
):
    query_ids = list(loader.qrels.keys())
    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}

    gar = GAR(corpus_graph=graph, budget=budget, batch_size=batch_size, backfill=True)

    rankings = {}
    n_expanded = []

    for i, qid in enumerate(query_ids):
        initial_pool = bm25_pools.get(qid, [])
        initial_doc_ids = {did for did, _ in initial_pool}

        query_text = loader.queries[qid]["text"]
        qid_cache = pre_scores.get(qid) if pre_scores else None
        scorer_fn = make_scorer_fn(query_text, loader.corpus, scorer, cache=qid_cache)
        reranked = gar.rerank(initial_pool, scorer_fn, graph_k=graph_k_limit)

        rankings[qid] = [doc_id for doc_id, _ in reranked[:top_k]]

        final_ids = {did for did, _ in reranked[:top_k]}
        n_expanded.append(len(final_ids - initial_doc_ids))

        if (i + 1) % 50 == 0 or i + 1 == len(query_ids):
            print(f"    {i + 1}/{len(query_ids)} queries processed")

    results = evaluate_ranking(rankings, ground_truth, top_k)
    results["avg_expanded_docs"] = float(np.mean(n_expanded))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_args(args):
    """Fill in None argparse values from dataset defaults then global defaults."""
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
    pick("initial_pool_size")
    pick("budget")
    pick("use_stemmer")
    pick("remove_stopwords")


def main():
    parser = argparse.ArgumentParser(description="GAR evaluation on statute retrieval datasets")
    parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--initial_pool_size", type=int, default=None,
                        help="BM25 pool size fed to GAR (default: 200 for kuhperdata, 100 otherwise)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Max docs scored by the reranker (default: 150 for kuhperdata, 100 otherwise)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Docs scored per GAR iteration (frontier exploration granularity)")
    parser.add_argument("--graph_k", type=int, default=16)
    parser.add_argument("--graph_k_limit", type=int, default=None)
    parser.add_argument("--scorer", type=str, default=None, choices=[*SCORERS],
                        help="Scorer model (default: bge for kuhperdata, mt5 otherwise). "
                             "bge=BAAI/bge-reranker-v2-m3 (multilingual/Indonesian), "
                             "mt5=unicamp-dl/mt5-base-mmarco-v2, "
                             "monot5=castorini/monot5-base-msmarco (English)")
    parser.add_argument("--scorer_batch_size", type=int, default=32,
                        help="Batch size for live (frontier) scorer calls")
    parser.add_argument("--prescore_batch_size", type=int, default=256,
                        help="Batch size for bulk pre-scoring the initial pool "
                             "(larger = faster, uses more GPU memory). "
                             "Set to 0 to disable pre-scoring.")
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_ngram", type=int, default=1)
    parser.add_argument("--use_stemmer", action=argparse.BooleanOptionalAction, default=None,
                        help="Indonesian stemmer via PySastrawi (default: off; "
                             "note: hurts BM25 on legal text due to over-stemming)")
    parser.add_argument("--remove_stopwords", action=argparse.BooleanOptionalAction, default=None,
                        help="Remove Indonesian stopwords via PySastrawi (default: off)")
    parser.add_argument("--cache_dir", type=str, default="outputs/gar")
    args = parser.parse_args()

    _resolve_args(args)

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    # Load scorer once (shared across datasets when --dataset all)
    scorer = build_scorer(args.scorer, batch_size=args.scorer_batch_size)

    for name, cfg in datasets.items():
        # For --dataset all, each dataset gets its own resolved settings
        if args.dataset == "all":
            dset_defaults = DATASET_DEFAULTS.get(name, {})
            scorer_name = dset_defaults.get("scorer", _GLOBAL_DEFAULTS["scorer"])
            initial_pool_size = dset_defaults.get("initial_pool_size", _GLOBAL_DEFAULTS["initial_pool_size"])
            budget = dset_defaults.get("budget", _GLOBAL_DEFAULTS["budget"])
            use_stemmer = dset_defaults.get("use_stemmer", _GLOBAL_DEFAULTS["use_stemmer"])
            remove_stopwords = dset_defaults.get("remove_stopwords", _GLOBAL_DEFAULTS["remove_stopwords"])
        else:
            scorer_name = args.scorer
            initial_pool_size = args.initial_pool_size
            budget = args.budget
            use_stemmer = args.use_stemmer
            remove_stopwords = args.remove_stopwords

        data_dir, lang = cfg["path"], cfg["lang"]
        corpus_path = f"{data_dir}/corpus.jsonl"
        queries_path = f"{data_dir}/queries.jsonl"
        qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

        print(f"\n{'=' * 60}")
        print(f"  GAR: {name.upper()} ({args.split} split)")
        print(f"  Scorer: {scorer_name} ({SCORERS[scorer_name][1]})")
        print(f"  Pool: {initial_pool_size}, Budget: {budget}, "
              f"stemmer={use_stemmer}, stopwords={remove_stopwords}")
        print(f"{'=' * 60}")

        loader = DataLoader(corpus_path, queries_path, qrels_path).load()
        doc_ids, doc_texts = loader.get_corpus_texts()
        print(f"  Corpus: {len(doc_ids)} docs, Queries: {len(loader.qrels)} queries")

        # Step 1: BM25 corpus graph
        print(f"\n  Step 1: BM25 corpus graph")
        graph = get_or_build_graph(
            name, doc_ids, doc_texts, args.graph_k, lang,
            args.bm25_b, args.bm25_k1, args.bm25_ngram, args.cache_dir,
            use_stemmer=use_stemmer, use_stopwords=remove_stopwords,
        )

        # Step 2: BM25 initial pool
        print(f"\n  Step 2: BM25 initial pool (top-{initial_pool_size})")
        bm25_pools = get_bm25_pools(
            loader, initial_pool_size, lang,
            b=args.bm25_b, k1=args.bm25_k1, n_gram=args.bm25_ngram,
            use_stemmer=use_stemmer, use_stopwords=remove_stopwords,
        )

        # Step 3: Pre-score initial pool in bulk
        # Pre-scoring batches (query, doc) pairs across all queries for GPU efficiency.
        # On CPU there is no batch parallelism benefit, so skip it to avoid
        # scoring more pairs than the budget actually requires.
        use_prescore = args.prescore_batch_size > 0 and torch.cuda.is_available()
        if args.prescore_batch_size > 0 and not torch.cuda.is_available():
            print(f"\n  Step 3a: Skipping pre-scoring (no CUDA — would be slower on CPU)")
        pre_scores = None
        if use_prescore:
            print(f"\n  Step 3a: Bulk pre-scoring initial pool")
            pre_scores = pre_score_pools(
                bm25_pools, loader, scorer, args.prescore_batch_size,
            )

        # Step 4: GAR (frontier docs only need live scoring)
        print(f"\n  Step {'3b' if pre_scores else '3'}: GAR "
              f"(pool={initial_pool_size}, budget={budget}, "
              f"batch={args.batch_size}, graph_k={args.graph_k})")
        t0 = time.time()
        results = run_gar(
            loader, graph, bm25_pools, scorer,
            top_k=args.top_k, budget=budget,
            batch_size=args.batch_size, graph_k_limit=args.graph_k_limit,
            pre_scores=pre_scores,
        )
        elapsed = time.time() - t0

        print(f"\n  MRR@{args.top_k}:       {results[f'mrr@{args.top_k}']:.4f}")
        print(f"  Recall@{args.top_k}:    {results[f'recall@{args.top_k}']:.4f}")
        print(f"  Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
        print(f"  Hit rate:       {results['hit_rate']:.4f}")
        print(f"  N queries:      {results['n_queries']}")
        print(f"  Avg expanded:   {results['avg_expanded_docs']:.1f} docs/query from graph")
        print(f"  Time (GAR):     {elapsed:.1f}s")

    scorer.cleanup()


if __name__ == "__main__":
    main()
