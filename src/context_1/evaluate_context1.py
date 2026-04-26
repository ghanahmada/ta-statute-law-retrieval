"""Evaluate Context-1 agentic retrieval on statute retrieval datasets.

Usage:
  1. Start vLLM server with Context-1:
     vllm serve chromadb/context-1 \
       --tool-call-parser openai \
       --enable-auto-tool-choice \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.95

  2. Run evaluation:
     python src/context_1/evaluate_context1.py \
       --dataset kuhperdata-humanized \
       --model chromadb/context-1 \
       --concurrency 4

  Output:
    outputs/context_1/{dataset}/agent_log.jsonl  (per-query logs, resumable)
    Console: MRR@10, Recall@10, Precision@10, Hit Rate
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.bm25 import BM25
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking

from context_1.hybrid_search import HybridSearcher
from context_1.tools import ToolExecutor
from context_1.agent import AgenticRetriever


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "stard": {"path": "data/stard", "lang": "zh"},
}


def load_done(log_path: str) -> tuple[set[str], dict[str, list[str]]]:
    done = set()
    rankings = {}
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec["qid"]
                    done.add(qid)
                    rankings[qid] = rec["ranked_doc_ids"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return done, rankings


def build_bm25(doc_texts: list[str], lang: str) -> BM25:
    if lang == "zh":
        import jieba
        jieba.setLogLevel(20)
        doc_texts = [" ".join(jieba.cut(t)) for t in doc_texts]

    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang=lang)
    bm25.fit(doc_texts)
    return bm25


def load_corpus_embeddings(embeddings_dir: str) -> np.ndarray | None:
    emb_path = Path(embeddings_dir) / "bge_m3_corpus.npy"
    if emb_path.exists():
        embs = np.load(emb_path)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embs / norms
    return None


def load_query_encoder(device: str = "cpu"):
    from FlagEmbedding import BGEM3FlagModel
    use_fp16 = device != "cpu"
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)


def load_reranker(device: str = "cpu"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)


class SimpleReranker:
    def __init__(self, model):
        self.model = model

    def score_pairs(self, queries, documents, batch_size=32):
        pairs = [[q, d] for q, d in zip(queries, documents)]
        scores = self.model.predict(pairs, batch_size=batch_size)
        return scores.tolist()


async def run_one_query(
    retriever: AgenticRetriever,
    sem: asyncio.Semaphore,
    qid: str,
    query_text: str,
) -> dict:
    async with sem:
        t0 = time.time()
        try:
            state = await retriever.run(query_text)
            ranked = list(state.selected_doc_ids.keys())
            return {
                "qid": qid,
                "ranked_doc_ids": ranked,
                "n_selected": len(ranked),
                "n_seen": len(state.seen_doc_ids),
                "turns": state.turn_count,
                "error": state.error,
                "elapsed_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            return {
                "qid": qid,
                "ranked_doc_ids": [],
                "n_selected": 0,
                "n_seen": 0,
                "turns": 0,
                "error": str(e),
                "elapsed_s": round(time.time() - t0, 2),
            }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Context-1 agentic retrieval",
    )
    parser.add_argument("--dataset", default="kuhperdata-humanized",
                        choices=DATASETS.keys())
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default="chromadb/context-1")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--use_reranker", action="store_true",
                        help="Use cross-encoder reranker on search results")
    parser.add_argument("--encoder_device", default="cuda",
                        help="Device for BGE-M3 encoder and reranker "
                        "(use cpu to leave GPU fully for vLLM)")
    parser.add_argument("--embeddings_dir", default="outputs/embeddings")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pad_to_k", type=int, default=0,
                        help="Pad agent rankings to k docs using seen_doc_ids "
                        "(0=no padding, 10=pad to 10 for fair comparison)")
    parser.add_argument("--debug_qid", default=None,
                        help="Run a single query and dump full conversation")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    data_dir = ds["path"]
    lang = ds["lang"]

    if args.output_dir is None:
        args.output_dir = f"outputs/context_1/{args.dataset}"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = f"{args.output_dir}/agent_log.jsonl"

    print("=" * 60)
    print(f"Context-1 Agentic Retrieval — {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Max turns: {args.max_turns}, Concurrency: {args.concurrency}")
    print("=" * 60)

    # --- Load data ---
    loader = DataLoader(
        f"{data_dir}/corpus.jsonl",
        f"{data_dir}/queries.jsonl",
        f"{data_dir}/qrels_{args.split}.tsv",
    ).load()

    if args.max_relevant > 0:
        before = len(loader.qrels)
        loader.filter_max_relevant(args.max_relevant)
        print(f"Queries: {len(loader.qrels)} (from {before}, "
              f"max_relevant={args.max_relevant})")

    doc_ids, doc_texts = loader.get_corpus_texts()
    corpus = loader.corpus
    test_qids = sorted(loader.qrels.keys())
    print(f"Corpus: {len(doc_ids)} documents")
    print(f"Test queries: {len(test_qids)}")

    # --- BM25 ---
    print("\nFitting BM25...")
    bm25 = build_bm25(doc_texts, lang)

    # --- Dense embeddings ---
    print("Loading corpus embeddings...")
    corpus_embeddings = load_corpus_embeddings(args.embeddings_dir)
    if corpus_embeddings is None:
        print("  No cached embeddings found. Encoding corpus with BGE-M3...")
        encoder = load_query_encoder(device=args.encoder_device)
        from evaluate_dense_retrieval import encode_with_bge
        corpus_embeddings = encode_with_bge(doc_texts, encoder)
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        corpus_embeddings = corpus_embeddings / norms
        os.makedirs(args.embeddings_dir, exist_ok=True)
        np.save(f"{args.embeddings_dir}/bge_m3_corpus.npy", corpus_embeddings)
        print(f"  Saved to {args.embeddings_dir}/bge_m3_corpus.npy")
    else:
        encoder = load_query_encoder(device=args.encoder_device)
        print(f"  Loaded cached embeddings: {corpus_embeddings.shape}")
    print(f"  Query encoder on: {args.encoder_device}")

    # --- Reranker (optional) ---
    reranker = None
    if args.use_reranker:
        print("Loading reranker...")
        reranker = SimpleReranker(load_reranker(device=args.encoder_device))

    # --- Build search pipeline ---
    searcher = HybridSearcher(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        corpus_embeddings=corpus_embeddings,
        bm25=bm25,
        query_encoder=encoder,
        reranker=reranker,
    )

    from context_1.token_budget import TokenBudgetTracker
    tool_executor = ToolExecutor(
        corpus=corpus,
        hybrid_searcher=searcher,
        token_counter=TokenBudgetTracker().count_tokens,
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    retriever = AgenticRetriever(
        client=client,
        model=args.model,
        tool_executor=tool_executor,
        max_turns=args.max_turns,
        pad_to_k=args.pad_to_k,
    )

    # --- Resume ---
    done_qids, prev_rankings = load_done(log_path)
    if done_qids:
        print(f"\nResuming: {len(done_qids)} queries already processed")

    remaining_qids = [qid for qid in test_qids if qid not in done_qids]
    print(f"Queries to process: {len(remaining_qids)}")

    # --- Debug mode: single query ---
    if args.debug_qid:
        qid = args.debug_qid
        query_text = loader.queries[qid]["text"]
        print(f"\n{'='*60}")
        print(f"  DEBUG: Running single query {qid}")
        print(f"  Query: {query_text[:200]}")
        print(f"{'='*60}\n")
        state = await retriever.run(query_text)
        print(f"\n{'='*60}")
        print(f"  CONVERSATION DUMP ({len(state.messages)} messages)")
        print(f"{'='*60}")
        for i, msg in enumerate(state.messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            print(f"\n--- [{i}] {role} ---")
            if content:
                print(content[:1000])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    print(f"  TOOL CALL: {fn.get('name')}({fn.get('arguments', '')[:200]})")
        print(f"\n{'='*60}")
        print(f"  Selected: {len(state.selected_doc_ids)} docs")
        print(f"  Seen: {len(state.seen_doc_ids)} docs")
        print(f"  Turns: {state.turn_count}")
        print(f"  Ground truth: {list(loader.qrels.get(qid, {}).keys())}")
        print(f"{'='*60}")
        return

    # --- Run ---
    if remaining_qids:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [
            run_one_query(
                retriever, sem, qid, loader.queries[qid]["text"],
            )
            for qid in remaining_qids
        ]

        results = await tqdm_asyncio.gather(
            *tasks, desc="Agentic retrieval",
        )

        with open(log_path, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                prev_rankings[result["qid"]] = result["ranked_doc_ids"]

        errors = [r for r in results if r["error"]]
        if errors:
            print(f"\n{len(errors)} queries had errors")
            for r in errors[:5]:
                print(f"  {r['qid']}: {r['error']}")

        avg_turns = np.mean([r["turns"] for r in results])
        avg_selected = np.mean([r["n_selected"] for r in results])
        avg_seen = np.mean([r["n_seen"] for r in results])
        avg_time = np.mean([r["elapsed_s"] for r in results])
        print(f"\nAgent stats:")
        print(f"  Avg turns: {avg_turns:.1f}")
        print(f"  Avg selected docs: {avg_selected:.1f}")
        print(f"  Avg seen docs: {avg_seen:.1f}")
        print(f"  Avg time/query: {avg_time:.1f}s")

    # --- Evaluate ---
    print(f"\n{'='*60}")
    print(f"  Results — {args.dataset} ({args.split})")
    print(f"{'='*60}")

    rankings = {}
    for qid in test_qids:
        if qid in prev_rankings:
            rankings[qid] = prev_rankings[qid]

    ground_truth = {
        qid: list(docs.keys())
        for qid, docs in loader.qrels.items()
        if qid in rankings
    }

    if not rankings:
        print("No rankings to evaluate.")
        return

    k = args.top_k
    metrics = evaluate_ranking(rankings, ground_truth, k)
    print(f"  MRR@{k}:       {metrics[f'mrr@{k}']:.4f}")
    print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
    print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f"  Hit Rate:      {metrics['hit_rate']:.4f}")
    print(f"  Queries:       {metrics['n_queries']}")


if __name__ == "__main__":
    asyncio.run(main())
