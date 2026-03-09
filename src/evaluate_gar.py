"""
Evaluate GAR (Graph-based Adaptive Re-ranking) on statute retrieval datasets.

Methodology: MacAvaney et al., "Adaptive Re-Ranking with a Corpus Graph" (CIKM 2022)

Faithful to paper:
  - Corpus graph built from BM25 (each doc as query against corpus)
  - Initial pool from BM25
  - Scorer: monoT5 cross-encoder re-ranker

monoT5 variants (multilingual via mMARCO):
  - monot5:   castorini/monot5-base-msmarco       (English-only, original paper)
  - mt5:      unicamp-dl/mt5-base-mmarco-v2       (14 languages incl. id/fr/en/zh)

Usage:
  python src/evaluate_gar.py --dataset kuhperdata
  python src/evaluate_gar.py --dataset all --budget 100 --graph_k 16
  python src/evaluate_gar.py --dataset kuhperdata --scorer monot5
"""

import argparse
import time
import jieba
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking
from gar.corpus_graph import CorpusGraph
from gar.adaptive_reranker import GAR

DATASETS = {
    "kuhperdata": {"path": "data/kuhperdata", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}

SCORERS = {
    "monot5": "castorini/monot5-base-msmarco",
    "mt5": "unicamp-dl/mt5-base-mmarco-v2",
}


def tokenize_chinese(texts: list[str]) -> list[str]:
    jieba.setLogLevel(20)
    return [" ".join(jieba.cut(t)) for t in texts]


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

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Score query-document pairs, return P(relevant) for each."""
        prompts = [
            f"Query: {query} Document: {doc} Relevant:" for doc in documents
        ]
        all_scores = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # Decoder input: start token
                decoder_input_ids = torch.full(
                    (len(batch_prompts), 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(
                    **inputs,
                    decoder_input_ids=decoder_input_ids,
                )
                # logits shape: (batch, 1, vocab_size)
                logits = outputs.logits[:, 0, :]
                # Extract true/false logits and softmax
                tf_logits = logits[:, [self.true_id, self.false_id]]
                probs = torch.softmax(tf_logits, dim=-1)
                # P(true) = probability of relevance
                batch_scores = probs[:, 0].cpu().tolist()

            all_scores.extend(batch_scores)

        return all_scores

    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def get_or_build_graph(dataset_name: str, doc_ids: List[str], doc_texts: List[str],
                       graph_k: int, lang: str, bm25_b: float, bm25_k1: float,
                       bm25_ngram: int, cache_dir: str) -> CorpusGraph:
    graph_path = Path(cache_dir) / dataset_name / f"bm25_graph_k{graph_k}"
    if graph_path.exists():
        graph = CorpusGraph.load(graph_path)
        if len(graph.doc_ids) == len(doc_ids):
            print(f"  Loaded cached BM25 corpus graph (k={graph_k})")
            return graph
    graph = CorpusGraph.build_from_bm25(
        doc_ids, doc_texts, k=graph_k,
        b=bm25_b, k1=bm25_k1, n_gram=bm25_ngram, lang=lang,
    )
    graph.save(graph_path)
    print(f"  Saved to {graph_path}")
    return graph


def get_bm25_pools(loader: DataLoader, top_n: int, lang: str,
                   b: float = 0.75, k1: float = 1.5, n_gram: int = 1
                   ) -> Dict[str, List[Tuple[str, float]]]:
    doc_ids, doc_texts = loader.get_corpus_texts()
    query_ids, query_texts = loader.get_query_texts()
    if lang == "zh":
        doc_texts = tokenize_chinese(doc_texts)
        query_texts = tokenize_chinese(query_texts)
    bm25 = BM25(b=b, k1=k1, n_gram=n_gram)
    bm25.fit(doc_texts)
    pools = {}
    for qid, query in zip(query_ids, query_texts):
        scores = bm25.transform(query)
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        pools[qid] = [(doc_ids[idx], float(scores[idx])) for idx in ranked_indices]
    return pools


def make_monot5_scorer(query_text: str, corpus: Dict[str, Dict], monot5: MonoT5Scorer):
    """Create a scorer function using monoT5 for a single query."""
    def scorer(batch_doc_ids: List[str]) -> Dict[str, float]:
        if not batch_doc_ids:
            return {}
        doc_texts = [corpus[did]["text"] for did in batch_doc_ids]
        scores = monot5.score(query_text, doc_texts)
        return dict(zip(batch_doc_ids, scores))
    return scorer


def run_gar(loader: DataLoader, graph: CorpusGraph,
            bm25_pools: Dict[str, List[Tuple[str, float]]],
            monot5: MonoT5Scorer, top_k: int, budget: int, batch_size: int,
            graph_k_limit: int = None):
    query_ids = list(loader.qrels.keys())
    ground_truth = {qid: list(docs.keys()) for qid, docs in loader.qrels.items()}

    gar = GAR(corpus_graph=graph, budget=budget, batch_size=batch_size, backfill=True)

    rankings = {}
    n_expanded = []

    for i, qid in enumerate(query_ids):
        initial_pool = bm25_pools.get(qid, [])
        initial_doc_ids = {did for did, _ in initial_pool}

        query_text = loader.queries[qid]["text"]
        scorer = make_monot5_scorer(query_text, loader.corpus, monot5)
        reranked = gar.rerank(initial_pool, scorer, graph_k=graph_k_limit)

        rankings[qid] = [doc_id for doc_id, _ in reranked[:top_k]]

        final_ids = {did for did, _ in reranked[:top_k]}
        n_expanded.append(len(final_ids - initial_doc_ids))

        if (i + 1) % 50 == 0 or i + 1 == len(query_ids):
            print(f"    {i + 1}/{len(query_ids)} queries processed")

    results = evaluate_ranking(rankings, ground_truth, top_k)
    results["avg_expanded_docs"] = float(np.mean(n_expanded))
    return results


def main():
    parser = argparse.ArgumentParser(description="GAR evaluation on statute retrieval datasets")
    parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--initial_pool_size", type=int, default=100)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--graph_k", type=int, default=16)
    parser.add_argument("--graph_k_limit", type=int, default=None)
    parser.add_argument("--scorer", type=str, default="mt5", choices=[*SCORERS],
                        help="monot5=English-only (original paper), mt5=multilingual (14 langs)")
    parser.add_argument("--scorer_batch_size", type=int, default=32)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_ngram", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="outputs/gar")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    # Load monoT5 scorer once
    scorer_model_name = SCORERS[args.scorer]
    monot5 = MonoT5Scorer(scorer_model_name, batch_size=args.scorer_batch_size)

    for name, cfg in datasets.items():
        data_dir, lang = cfg["path"], cfg["lang"]
        corpus_path = f"{data_dir}/corpus.jsonl"
        queries_path = f"{data_dir}/queries.jsonl"
        qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

        print(f"\n{'=' * 60}")
        print(f"  GAR: {name.upper()} ({args.split} split)")
        print(f"  Scorer: {args.scorer} ({scorer_model_name})")
        print(f"{'=' * 60}")

        loader = DataLoader(corpus_path, queries_path, qrels_path).load()
        doc_ids, doc_texts = loader.get_corpus_texts()
        print(f"  Corpus: {len(doc_ids)} docs, Queries: {len(loader.qrels)} queries")

        # Step 1: BM25 corpus graph
        print(f"\n  Step 1: BM25 corpus graph")
        graph = get_or_build_graph(
            name, doc_ids, doc_texts, args.graph_k, lang,
            args.bm25_b, args.bm25_k1, args.bm25_ngram, args.cache_dir,
        )

        # Step 2: BM25 initial pool
        print(f"\n  Step 2: BM25 initial pool (top-{args.initial_pool_size})")
        bm25_pools = get_bm25_pools(
            loader, args.initial_pool_size, lang,
            b=args.bm25_b, k1=args.bm25_k1, n_gram=args.bm25_ngram,
        )

        # Step 3: GAR with monoT5
        print(f"\n  Step 3: GAR (pool={args.initial_pool_size}, budget={args.budget}, "
              f"batch={args.batch_size}, graph_k={args.graph_k})")
        t0 = time.time()
        results = run_gar(
            loader, graph, bm25_pools, monot5,
            top_k=args.top_k, budget=args.budget,
            batch_size=args.batch_size, graph_k_limit=args.graph_k_limit,
        )
        elapsed = time.time() - t0

        print(f"\n  MRR@{args.top_k}:       {results[f'mrr@{args.top_k}']:.4f}")
        print(f"  Recall@{args.top_k}:    {results[f'recall@{args.top_k}']:.4f}")
        print(f"  Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
        print(f"  Hit rate:       {results['hit_rate']:.4f}")
        print(f"  N queries:      {results['n_queries']}")
        print(f"  Avg expanded:   {results['avg_expanded_docs']:.1f} docs/query from graph")
        print(f"  Time:           {elapsed:.1f}s")

    monot5.cleanup()


if __name__ == "__main__":
    main()
