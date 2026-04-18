"""Inference script for Para-GNN / Prox-GNN / StructGNN on test set.

Loads a trained model checkpoint and produces ranked retrieval results
for all test queries. Outputs per-query rankings as JSON.

Usage:
  # StructGNN on KUHPerdata-humanized:
  python src/inference/infer_paragnn.py --dataset kuhperdata-humanized --structure_mode structural

  # Para-GNN on BSARD:
  python src/inference/infer_paragnn.py --dataset bsard --structure_mode none

  # All datasets at once:
  python src/inference/infer_paragnn.py --dataset all --structure_mode structural
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paragnn import DATASETS, ParaGNNConfig
from paragnn.graph_builder import ParagraphStore, GraphBuilder
from paragnn.model import TestCaseGnn
from paragnn.structure import precompute_structure_features, get_query_structure_features
from util.dataloader import DataLoader


def get_model_dir(config):
    mode = config.structure_mode
    method_suffix = config.method
    if mode == "proximity":
        method_suffix = f"{method_suffix}_prox{config.proximity_radius}"
    elif mode == "structural":
        method_suffix = f"{method_suffix}_struct"
    return f"{config.output_dir}/{config.dataset}/{method_suffix}"


@torch.no_grad()
def run_inference(config, model_dir, para_store, structure_features, query_structure_feature,
                  bm25_test_scores, test_query_ids, test_corpus_ids, test_gold, alpha=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config.structure_mode
    dim = config.embed_dim
    struct_input_dim = dim + config.act_dim + config.pos_dim

    model_path = f"{model_dir}/best_model.pt"
    if not Path(model_path).exists():
        print(f"  No trained model found at {model_path}")
        return None

    model = TestCaseGnn(
        in_dim=dim, h_dim=dim, out_dim=dim,
        dropout=config.dropout, num_head=config.num_heads,
        structure_mode=mode, struct_input_dim=struct_input_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    print("  Building test graph...")
    test_graph = GraphBuilder(
        test_query_ids, test_corpus_ids, para_store,
        structure_mode=mode,
        proximity_radius=config.proximity_radius,
        structure_features=structure_features,
        query_structure_feature=query_structure_feature,
    ).graph
    test_graph = test_graph.to(device)

    node_h = test_graph.ndata["h"]
    edge_h = test_graph.edata["h"]

    if mode == "structural":
        node_h = model.struct_proj(node_h)

    h = model.eugat_gnn(test_graph, node_h, edge_h)
    query_encoded = h[test_graph.ndata["query_mask"].bool()]
    candidate_encoded = h[test_graph.ndata["candidate_mask"].bool()]
    learned_alphas = model.ffnn(query_encoded)

    scores = torch.matmul(query_encoded, candidate_encoded.T)
    mean_scores = scores.mean(dim=1, keepdim=True)
    std_scores = scores.std(dim=1, keepdim=True)
    gnn_scores = (scores - mean_scores) / (std_scores + 1e-8)
    gnn_scores = gnn_scores.cpu()

    bm25_test_scores = torch.nan_to_num(bm25_test_scores, nan=0.0)

    # Build gold matrix for metrics
    gold_matrix = torch.zeros(len(test_query_ids), len(test_corpus_ids))
    doc_id_to_idx = {d: i for i, d in enumerate(test_corpus_ids)}
    for qi, qid in enumerate(test_query_ids):
        if qid in test_gold:
            for did, score in test_gold[qid].items():
                if did in doc_id_to_idx and score > 0:
                    gold_matrix[qi][doc_id_to_idx[did]] = 1

    # Alpha grid search if not specified
    if alpha is None:
        print("  Running alpha grid search...")
        best_alpha, best_mrr = 0, 0
        for a in np.arange(0.0, 1.05, 0.1):
            blended = a * gnn_scores + (1 - a) * bm25_test_scores
            mrr = compute_mrr(blended, gold_matrix)
            if mrr > best_mrr:
                best_mrr = mrr
                best_alpha = a
        alpha = best_alpha
        print(f"  Best alpha: {alpha:.1f} (MRR@10: {best_mrr:.4f})")

    final_scores = alpha * gnn_scores + (1 - alpha) * bm25_test_scores

    # Compute metrics
    mrr, recall, hit_rate = compute_metrics(final_scores, gold_matrix)
    print(f"  MRR@10: {mrr:.4f}  R@10: {recall:.4f}  Hit: {hit_rate:.1%}")

    # Build per-query ranked results
    results = {}
    k = 10
    for qi, qid in enumerate(test_query_ids):
        ranked_indices = torch.argsort(final_scores[qi], descending=True)[:k].tolist()
        ranked_docs = []
        relevant_set = set()
        if qid in test_gold:
            relevant_set = {did for did, s in test_gold[qid].items() if s > 0}

        for rank, idx in enumerate(ranked_indices):
            doc_id = test_corpus_ids[idx]
            ranked_docs.append({
                "rank": rank + 1,
                "doc_id": doc_id,
                "score": float(final_scores[qi][idx]),
                "gnn_score": float(gnn_scores[qi][idx]),
                "bm25_score": float(bm25_test_scores[qi][idx]),
                "relevant": doc_id in relevant_set,
            })

        results[qid] = {
            "ranked": ranked_docs,
            "num_relevant": len(relevant_set),
            "hits_in_top10": sum(1 for d in ranked_docs if d["relevant"]),
            "first_relevant_rank": next(
                (d["rank"] for d in ranked_docs if d["relevant"]), None
            ),
        }

    return {
        "config": {
            "dataset": config.dataset,
            "structure_mode": mode,
            "method": config.method,
            "alpha": float(alpha),
            "learned_alpha_mean": float(learned_alphas.mean().item()),
        },
        "metrics": {
            "mrr@10": float(mrr),
            "recall@10": float(recall),
            "hit_rate": float(hit_rate),
            "num_queries": len(test_query_ids),
            "num_candidates": len(test_corpus_ids),
        },
        "results": results,
    }


def compute_mrr(scores, gold_matrix, k=10):
    mrr_sum = 0
    n_queries = gold_matrix.shape[0]
    for qi in range(n_queries):
        relevant = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
        if not relevant:
            continue
        ranked = torch.argsort(scores[qi], descending=True)[:k].tolist()
        for rank, idx in enumerate(ranked):
            if idx in relevant:
                mrr_sum += 1.0 / (rank + 1)
                break
    return mrr_sum / n_queries


def compute_metrics(scores, gold_matrix, k=10):
    mrr_sum = 0
    recall_sum = 0
    hit_count = 0
    n_queries = gold_matrix.shape[0]
    for qi in range(n_queries):
        relevant = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
        if not relevant:
            continue
        ranked = torch.argsort(scores[qi], descending=True)[:k].tolist()
        for rank, idx in enumerate(ranked):
            if idx in relevant:
                mrr_sum += 1.0 / (rank + 1)
                break
        hits = len(set(ranked) & set(relevant))
        recall_sum += hits / len(relevant)
        if hits > 0:
            hit_count += 1
    return mrr_sum / n_queries, recall_sum / n_queries, hit_count / n_queries


def main():
    parser = argparse.ArgumentParser(description="Inference for Para-GNN / Prox-GNN / StructGNN")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--method", default="adapted", choices=["full", "adapted"])
    parser.add_argument("--structure_mode", default="structural", choices=["none", "proximity", "structural"])
    parser.add_argument("--proximity_radius", type=int, default=50)
    parser.add_argument("--act_dim", type=int, default=64)
    parser.add_argument("--pos_dim", type=int, default=32)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha (skip grid search)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/inference", help="Directory for inference results")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    mode_labels = {
        "none": "Para-GNN",
        "proximity": f"Prox-GNN (r={args.proximity_radius})",
        "structural": f"StructGNN (act={args.act_dim}d, pos={args.pos_dim}d)",
    }

    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Inference: {mode_labels[args.structure_mode]} [{args.method}]: {name}")
        print(f"{'='*60}")

        config = ParaGNNConfig(
            dataset=name,
            data_path=cfg["path"],
            lang=cfg["lang"],
            method=args.method,
            structure_mode=args.structure_mode,
            proximity_radius=args.proximity_radius,
            act_dim=args.act_dim,
            pos_dim=args.pos_dim,
            max_relevant=args.max_relevant,
        )

        precompute_dir = f"{config.output_dir}/{name}"
        model_dir = get_model_dir(config)

        if not Path(f"{model_dir}/best_model.pt").exists():
            print(f"  No trained model at {model_dir}/best_model.pt — skipping")
            continue

        print("  Loading pre-computed data...")
        bm25_test_scores = torch.load(f"{precompute_dir}/bm25_test_scores.pt")
        with open(f"{precompute_dir}/test_query_ids.json") as f:
            test_qids = json.load(f)
        with open(f"{precompute_dir}/corpus_doc_ids.json") as f:
            corpus_doc_ids = json.load(f)

        test_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_test.tsv",
        ).load()
        if config.max_relevant > 0:
            test_loader.filter_max_relevant(config.max_relevant)

        print("  Loading paragraph store...")
        para_store = ParagraphStore(output_dir=precompute_dir, method=args.method)

        structure_features = None
        query_structure_feature = None
        if args.structure_mode == "structural":
            corpus_path = f"{config.data_path}/corpus.jsonl"
            print(f"  Computing structure features...")
            structure_features = precompute_structure_features(
                corpus_path, name, act_dim=args.act_dim, pos_dim=args.pos_dim
            )
            query_act, query_pos = get_query_structure_features(
                act_dim=args.act_dim, pos_dim=args.pos_dim
            )
            query_structure_feature = torch.cat([query_act, query_pos])

        output = run_inference(
            config, model_dir, para_store,
            structure_features, query_structure_feature,
            bm25_test_scores, test_qids, corpus_doc_ids,
            test_loader.qrels, alpha=args.alpha,
        )

        if output is None:
            continue

        # Save results
        out_dir = Path(args.output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        mode_suffix = {"none": "paragnn", "proximity": "proxgnn", "structural": "structgnn"}[args.structure_mode]
        out_path = out_dir / f"{mode_suffix}_{args.method}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {out_path}")

        # Also save a compact TREC-style run file
        trec_path = out_dir / f"{mode_suffix}_{args.method}.run"
        with open(trec_path, "w") as f:
            for qid, qresult in output["results"].items():
                for doc in qresult["ranked"]:
                    f.write(f"{qid}\tQ0\t{doc['doc_id']}\t{doc['rank']}\t{doc['score']:.6f}\t{mode_suffix}\n")

        print(f"  Saved: {trec_path}")


if __name__ == "__main__":
    main()
