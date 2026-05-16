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
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    "kuhperdata-exp": {"path": "data/kuhperdata-exp", "lang": "id"},
    "kuhperdata-summ-exp": {"path": "data/kuhperdata-summ-exp", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "coliee": {"path": "data/coliee", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}


def load_done(log_path: str) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    done = set()
    rankings = {}
    seen_rankings = {}
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
                    seen_rankings[qid] = rec.get("ranked_seen_100", rec["ranked_doc_ids"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done, rankings, seen_rankings


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


def _extract_conversation(messages: list[dict]) -> list[dict]:
    """Extract a compact conversation log with reasoning from agent messages."""
    conv = []
    for msg in messages:
        entry = {"role": msg.get("role", "?")}
        if msg.get("reasoning"):
            entry["reasoning"] = msg["reasoning"]
        content = msg.get("content", "")
        if content:
            entry["content"] = content[:2000]
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc["function"]["name"],
                 "args": tc["function"]["arguments"][:500]}
                for tc in tool_calls
            ]
        conv.append(entry)
    return conv


async def run_one_query(
    retriever: AgenticRetriever,
    sem: asyncio.Semaphore,
    qid: str,
    query_text: str,
    gt_docs: list[str] | None = None,
) -> dict:
    async with sem:
        t0 = time.time()
        gt_set = set(gt_docs or [])
        try:
            state = await retriever.run(query_text)
            ranked = list(state.selected_doc_ids.keys())
            ranked_seen_100 = sorted(
                state.seen_doc_ids,
                key=lambda d: state.doc_scores.get(d, 0),
                reverse=True,
            )[:100]
            for call in state.tool_call_log:
                call["hit_relevant"] = bool(set(call["doc_ids_returned"]) & gt_set)
            return {
                "qid": qid,
                "ranked_doc_ids": ranked,
                "ranked_seen_100": ranked_seen_100,
                "doc_scores": {d: round(s, 6) for d, s in state.doc_scores.items()},
                "n_selected": len(ranked),
                "n_seen": len(state.seen_doc_ids),
                "n_read": len(state.read_doc_ids),
                "turns": state.turn_count,
                "n_frames_declared": len(state.frames),
                "n_frames_covered": len([f for f, docs in state.frames.items() if docs]),
                "n_gate_triggers": state.n_gate_triggers,
                "n_similarity_rejections": state.n_similarity_rejections,
                "error": state.error,
                "elapsed_s": round(time.time() - t0, 2),
                "tool_call_log": state.tool_call_log,
                "conversation": _extract_conversation(state.messages),
            }
        except Exception as e:
            return {
                "qid": qid,
                "ranked_doc_ids": [],
                "n_selected": 0,
                "n_seen": 0,
                "n_read": 0,
                "turns": 0,
                "n_frames_declared": 0,
                "n_frames_covered": 0,
                "n_gate_triggers": 0,
                "n_similarity_rejections": 0,
                "error": str(e),
                "elapsed_s": round(time.time() - t0, 2),
                "tool_call_log": [],
                "conversation": [],
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
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--use_reranker", action="store_true",
                        help="Use cross-encoder reranker on search results")
    parser.add_argument("--encoder_device", default="cuda",
                        help="Device for BGE-M3 encoder and reranker "
                        "(use cpu to leave GPU fully for vLLM)")
    parser.add_argument("--embeddings_dir", default=None,
                        help="Directory for cached BGE-M3 corpus embeddings "
                        "(default: outputs/embeddings/<dataset>)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pad_to_k", type=int, default=0,
                        help="Pad agent rankings to k docs using seen_doc_ids "
                        "(0=no padding, 10=pad to 10 for fair comparison)")
    parser.add_argument("--dense_source", default="bge",
                        choices=["bge", "structgnn"],
                        help="Dense embedding source: bge (BGE-M3) or structgnn "
                        "(GNN corpus+query embeddings)")
    parser.add_argument("--gnn_model_dir", default=None,
                        help="Path to StructGNN model dir with best_model.pt "
                        "and gnn_corpus_embeddings.npy (required if --dense_source structgnn)")
    parser.add_argument("--gnn_alpha", type=float, default=0.8,
                        help="Alpha for StructGNN scoring: alpha*gnn + (1-alpha)*bm25")
    parser.add_argument("--debug_qid", default=None,
                        help="Run a single query and dump full conversation")
    parser.add_argument("--no_hierarchy", action="store_true",
                        help="Use flat prompt instead of L1-L4 hierarchy scaffold")
    parser.add_argument("--no_coverage_gate", action="store_true",
                        help="Disable coverage-gate enforcement")
    parser.add_argument("--no_similarity_guard", action="store_true",
                        help="Disable query similarity guard")
    parser.add_argument("--similarity_threshold", type=float, default=0.92,
                        help="Cosine similarity threshold for query rejection (default: 0.92)")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    data_dir = ds["path"]
    lang = ds["lang"]

    if args.embeddings_dir is None:
        args.embeddings_dir = f"outputs/embeddings/{args.dataset}"

    if args.output_dir is None:
        dense_tag = f"_{args.dense_source}" if args.dense_source != "bge" else ""
        args.output_dir = f"outputs/context_1/{args.dataset}{dense_tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = f"{args.output_dir}/agent_log.jsonl"

    print("=" * 60)
    print(f"Context-1 Agentic Retrieval — {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Dense source: {args.dense_source}")
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

    # --- Reranker (optional) ---
    reranker = None
    if args.use_reranker:
        print("Loading reranker...")
        reranker = SimpleReranker(load_reranker(device=args.encoder_device))

    # --- Build search pipeline ---
    if args.dense_source == "structgnn":
        gnn_model_dir = (args.gnn_model_dir
                         or f"outputs/paragnn/{args.dataset}/adapted_struct")
        gnn_emb_path = Path(gnn_model_dir) / "gnn_corpus_embeddings.npy"
        gnn_model_path = Path(gnn_model_dir) / "best_model.pt"
        rr_emb_path = (Path(gnn_model_dir).parent
                       / "embeddings" / "EMBD_CONST.pt")

        if not gnn_emb_path.exists() or not gnn_model_path.exists():
            print(f"ERROR: StructGNN files not found in {gnn_model_dir}")
            print(f"  Run first: python src/paragnn/inference.py "
                  f"--dataset {args.dataset} --export_embeddings")
            return

        print(f"\nLoading StructGNN searcher...")
        print(f"  Corpus embeddings: {gnn_emb_path}")
        print(f"  Model: {gnn_model_path}")
        gnn_corpus_emb = np.load(gnn_emb_path)
        print(f"  Loaded: {gnn_corpus_emb.shape}")

        import torch
        rr_const_emb = torch.load(rr_emb_path, map_location="cpu")

        print("  Loading BGE-M3 (for GNN query encoding)...")
        encoder = load_query_encoder(device=args.encoder_device)

        from paragnn.gnn_searcher import StructGNNSearcher

        searcher = StructGNNSearcher(
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            corpus_embeddings=gnn_corpus_emb,
            bm25=bm25,
            bge_encoder=encoder,
            model_path=str(gnn_model_path),
            rr_const_emb=rr_const_emb,
            alpha=args.gnn_alpha,
            device=args.encoder_device,
            reranker=reranker,
        )
        print(f"  StructGNN searcher ready (alpha={searcher.alpha})")
    else:
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
        use_hierarchy=not args.no_hierarchy,
        use_coverage_gate=not args.no_coverage_gate,
        use_similarity_guard=not args.no_similarity_guard,
        similarity_threshold=args.similarity_threshold,
    )

    # --- Resume ---
    done_qids, prev_rankings, prev_seen_rankings = load_done(log_path)
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
        if state.error:
            print(f"  ERROR: {state.error}")
        print(f"  Ground truth: {list(loader.qrels.get(qid, {}).keys())}")
        print(f"{'='*60}")
        return

    # --- Run ---
    if remaining_qids:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [
            run_one_query(
                retriever, sem, qid, loader.queries[qid]["text"],
                gt_docs=list(loader.qrels.get(qid, {}).keys()),
            )
            for qid in remaining_qids
        ]

        conv_log_path = f"{args.output_dir}/agent_conversations.jsonl"
        results = []
        log_file = open(log_path, "a", encoding="utf-8")
        conv_file = open(conv_log_path, "a", encoding="utf-8")
        pbar = tqdm_asyncio(total=len(tasks), desc="Agentic retrieval")
        interrupted = False
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                prev_rankings[result["qid"]] = result["ranked_doc_ids"]
                prev_seen_rankings[result["qid"]] = result.get("ranked_seen_100", result["ranked_doc_ids"])

                conversation = result.pop("conversation", [])
                qid = result["qid"]
                gt_docs = list(loader.qrels.get(qid, {}).keys())
                result["ground_truth"] = gt_docs
                log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                log_file.flush()

                conv_rec = {
                    "qid": result["qid"],
                    "conversation": conversation,
                }
                conv_file.write(json.dumps(conv_rec, ensure_ascii=False) + "\n")
                conv_file.flush()

                pbar.update(1)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n\nInterrupted! Evaluating {len(results)} completed queries...")
        finally:
            pbar.close()
            log_file.close()
            conv_file.close()

        errors = [r for r in results if r["error"]]
        if errors:
            print(f"\n{len(errors)} queries had errors")
            for r in errors[:5]:
                print(f"  {r['qid']}: {r['error']}")

        if results:
            avg_turns = np.mean([r["turns"] for r in results])
            avg_selected = np.mean([r["n_selected"] for r in results])
            avg_seen = np.mean([r["n_seen"] for r in results])
            avg_read = np.mean([r["n_read"] for r in results])
            avg_time = np.mean([r["elapsed_s"] for r in results])
            avg_frames_decl = np.mean([r.get("n_frames_declared", 0) for r in results])
            avg_frames_cov = np.mean([r.get("n_frames_covered", 0) for r in results])
            total_gate = sum(r.get("n_gate_triggers", 0) for r in results)
            total_sim_rej = sum(r.get("n_similarity_rejections", 0) for r in results)
            print(f"\nAgent stats ({len(results)} queries):")
            print(f"  Avg turns: {avg_turns:.1f}")
            print(f"  Avg selected docs: {avg_selected:.1f}")
            print(f"  Avg seen docs: {avg_seen:.1f}")
            print(f"  Avg read docs: {avg_read:.1f}")
            print(f"  Avg time/query: {avg_time:.1f}s")
            print(f"  Avg frames declared: {avg_frames_decl:.1f}")
            print(f"  Avg frames covered: {avg_frames_cov:.1f}")
            print(f"  Total gate triggers: {total_gate}")
            print(f"  Total similarity rejections: {total_sim_rej}")

        if interrupted:
            print(f"\nConversation log: {conv_log_path}")

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
    print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR@{k}:       {metrics[f'mrr@{k}']:.4f}")
    print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f"  Hit Rate:      {metrics['hit_rate']:.4f}")
    print(f"  Queries:       {metrics['n_queries']}")

    # --- Save predictions ---
    os.makedirs("outputs/predictions", exist_ok=True)
    if args.dense_source != "bge":
        tag = f"context1_{args.dense_source}"
    elif args.no_hierarchy and args.no_coverage_gate and args.no_similarity_guard:
        tag = "context1_flat"
    else:
        tag = "context1"
    pred_path = f"outputs/predictions/{tag}_{args.dataset}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for qid in test_qids:
            if qid in prev_rankings:
                gt = list(loader.qrels.get(qid, {}).keys())
                seen_100 = prev_seen_rankings.get(qid, [])
                f.write(json.dumps({
                    "qid": qid,
                    "ranked_doc_ids": prev_rankings[qid],
                    "ranked_seen_100": seen_100,
                    "ground_truth": gt,
                }, ensure_ascii=False) + "\n")
    print(f"  Predictions:   {pred_path}  ({len(prev_rankings)} queries)")


if __name__ == "__main__":
    asyncio.run(main())
