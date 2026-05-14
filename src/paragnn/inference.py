"""Export StructGNN / Para-GNN inference rankings from a trained checkpoint.

Loads best_model.pt, computes blended GNN+BM25 scores with optimal alpha,
and writes per-query top-K rankings with ground truth rank positions.

Usage:
  python src/paragnn/inference.py --dataset kuhperdata-exp --structure_mode structural --top_k 100
  python src/paragnn/inference.py --dataset kuhperdata-summ-exp --structure_mode structural --top_k 100
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
from util.metrics import save_predictions


def load_precomputed(output_dir: str):
    bm25_test_scores = torch.load(f"{output_dir}/bm25_test_scores.pt")
    bm25_val_scores = torch.load(f"{output_dir}/bm25_val_scores.pt")
    with open(f"{output_dir}/test_query_ids.json") as f:
        test_qids = json.load(f)
    with open(f"{output_dir}/val_query_ids.json") as f:
        val_qids = json.load(f)
    with open(f"{output_dir}/corpus_doc_ids.json") as f:
        corpus_doc_ids = json.load(f)
    return bm25_test_scores, bm25_val_scores, test_qids, val_qids, corpus_doc_ids


def build_gold_matrix(test_qids, corpus_doc_ids, qrels):
    doc_id_to_idx = {d: i for i, d in enumerate(corpus_doc_ids)}
    gold = torch.zeros(len(test_qids), len(corpus_doc_ids))
    for qi, qid in enumerate(test_qids):
        if qid in qrels:
            for did, score in qrels[qid].items():
                if did in doc_id_to_idx and score > 0:
                    gold[qi][doc_id_to_idx[did]] = 1
    return gold


@torch.no_grad()
def get_gnn_scores(model, test_graph, device):
    model.eval()
    node_h = test_graph.ndata["h"]
    edge_h = test_graph.edata["h"]

    if model.structure_mode == "structural":
        node_h = model.struct_proj(node_h)

    h = model.eugat_gnn(test_graph, node_h, edge_h)
    query_encoded = h[test_graph.ndata["query_mask"].bool()]
    candidate_encoded = h[test_graph.ndata["candidate_mask"].bool()]

    scores = torch.matmul(query_encoded, candidate_encoded.T)
    mean_scores = scores.mean(dim=1, keepdim=True)
    std_scores = scores.std(dim=1, keepdim=True)
    gnn_scores = (scores - mean_scores) / (std_scores + 1e-8)

    return gnn_scores.cpu(), candidate_encoded.cpu()


def grid_search_alpha(gnn_scores, bm25_scores, gold_matrix, k=10):
    best_alpha, best_mrr = 0, 0
    for alpha in np.arange(0.0, 1.05, 0.1):
        scores = alpha * gnn_scores + (1 - alpha) * bm25_scores
        mrr = compute_mrr(scores, gold_matrix, k)
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = alpha
    return best_alpha, best_mrr


def compute_mrr(scores, gold_matrix, k=10):
    mrr_sum = 0
    n = gold_matrix.shape[0]
    for qi in range(n):
        relevant = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
        if not relevant:
            continue
        ranked = torch.argsort(scores[qi], descending=True)[:k].tolist()
        for rank, idx in enumerate(ranked):
            if idx in relevant:
                mrr_sum += 1.0 / (rank + 1)
                break
    return mrr_sum / n


def export_rankings(
    scores, test_qids, corpus_doc_ids, gold_matrix, alpha, debiased, out_path, top_k,
):
    gt_ranks_all = []

    with open(out_path, "w", encoding="utf-8") as f:
        for qi, qid in enumerate(test_qids):
            row = scores[qi]
            sorted_indices = torch.argsort(row, descending=True)

            rankings = []
            for rank, idx in enumerate(sorted_indices[:top_k].tolist()):
                rankings.append({
                    "rank": rank + 1,
                    "doc_id": corpus_doc_ids[idx],
                    "score": round(float(row[idx]), 6),
                })

            full_rank_lookup = {idx: r for r, idx in enumerate(sorted_indices.tolist())}
            gt_docs = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
            ground_truth = []
            for idx in gt_docs:
                r = full_rank_lookup[idx] + 1
                ground_truth.append({
                    "doc_id": corpus_doc_ids[idx],
                    "rank": r,
                    "score": round(float(row[idx]), 6),
                    "in_top_k": r <= top_k,
                })
                gt_ranks_all.append(r)
            ground_truth.sort(key=lambda x: x["rank"])

            rec = {
                "qid": qid,
                "rankings": rankings,
                "ground_truth": ground_truth,
                "n_corpus": len(corpus_doc_ids),
                "alpha": round(float(alpha), 2),
                "debiased": debiased,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return gt_ranks_all


def main():
    parser = argparse.ArgumentParser(description="StructGNN / Para-GNN inference")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--method", default="adapted", choices=["full", "adapted"])
    parser.add_argument("--structure_mode", default="structural",
                        choices=["none", "proximity", "structural"])
    parser.add_argument("--act_dim", type=int, default=64)
    parser.add_argument("--pos_dim", type=int, default=32)
    parser.add_argument("--max_relevant", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--proximity_radius", type=int, default=50)
    parser.add_argument("--export_embeddings", action="store_true",
                        help="Export GNN corpus embeddings as .npy for hybrid search")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    config = ParaGNNConfig(
        dataset=args.dataset,
        data_path=cfg["path"],
        lang=cfg["lang"],
        method=args.method,
        structure_mode=args.structure_mode,
        act_dim=args.act_dim,
        pos_dim=args.pos_dim,
        max_relevant=args.max_relevant,
    )

    mode = args.structure_mode
    method_suffix = args.method
    if mode == "proximity":
        method_suffix = f"{method_suffix}_prox{args.proximity_radius}"
    elif mode == "structural":
        method_suffix = f"{method_suffix}_struct"

    base_dir = f"{config.output_dir}/{args.dataset}"
    model_dir = f"{base_dir}/{method_suffix}"
    model_path = f"{model_dir}/best_model.pt"

    if not Path(model_path).exists():
        print(f"No trained model at {model_path}")
        print(f"Run training first: python src/evaluate_paragnn.py --dataset {args.dataset} --structure_mode {mode}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_name = {"none": "Para-GNN", "proximity": "Prox-GNN", "structural": "StructGNN"}[mode]

    print(f"{'='*60}")
    print(f"  {mode_name} Inference: {args.dataset}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load precomputed data
    print("\nLoading precomputed data...")
    bm25_test_scores, bm25_val_scores, test_qids, val_qids, corpus_doc_ids = load_precomputed(base_dir)
    bm25_test_scores = torch.nan_to_num(bm25_test_scores, nan=0.0)
    bm25_val_scores = torch.nan_to_num(bm25_val_scores, nan=0.0)

    # Load val + test qrels
    val_loader = DataLoader(
        f"{cfg['path']}/corpus.jsonl",
        f"{cfg['path']}/queries.jsonl",
        f"{cfg['path']}/qrels_val.tsv",
    ).load()
    test_loader = DataLoader(
        f"{cfg['path']}/corpus.jsonl",
        f"{cfg['path']}/queries.jsonl",
        f"{cfg['path']}/qrels_test.tsv",
    ).load()
    if args.max_relevant > 0:
        val_loader.filter_max_relevant(args.max_relevant)
        test_loader.filter_max_relevant(args.max_relevant)

    val_gold_matrix = build_gold_matrix(val_qids, corpus_doc_ids, val_loader.qrels)
    test_gold_matrix = build_gold_matrix(test_qids, corpus_doc_ids, test_loader.qrels)
    print(f"  Val queries: {len(val_qids)}, Test queries: {len(test_qids)}, Corpus: {len(corpus_doc_ids)}")

    # Load paragraph store
    print("Loading paragraph store...")
    para_store = ParagraphStore(output_dir=base_dir, method=args.method)

    # Structure features
    structure_features = None
    query_structure_feature = None
    if mode == "structural":
        corpus_path = f"{cfg['path']}/corpus.jsonl"
        print(f"Computing structure features (act_dim={args.act_dim}, pos_dim={args.pos_dim})...")
        structure_features = precompute_structure_features(
            corpus_path, args.dataset, act_dim=args.act_dim, pos_dim=args.pos_dim,
        )
        query_act, query_pos = get_query_structure_features(
            act_dim=args.act_dim, pos_dim=args.pos_dim,
        )
        query_structure_feature = torch.cat([query_act, query_pos])

    # Build val + test graphs
    print("Building val graph...")
    val_graph = GraphBuilder(
        val_qids, corpus_doc_ids, para_store,
        structure_mode=mode,
        proximity_radius=args.proximity_radius,
        structure_features=structure_features,
        query_structure_feature=query_structure_feature,
    ).graph
    val_graph = val_graph.to(device)

    print("Building test graph...")
    test_graph = GraphBuilder(
        test_qids, corpus_doc_ids, para_store,
        structure_mode=mode,
        proximity_radius=args.proximity_radius,
        structure_features=structure_features,
        query_structure_feature=query_structure_feature,
    ).graph
    test_graph = test_graph.to(device)

    # Load model
    print("Loading model checkpoint...")
    dim = config.embed_dim
    struct_input_dim = dim + args.act_dim + args.pos_dim
    model = TestCaseGnn(
        in_dim=dim, h_dim=dim, out_dim=dim,
        dropout=config.dropout, num_head=config.num_heads,
        structure_mode=mode, struct_input_dim=struct_input_dim,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Compute GNN scores on val + test
    print("Computing GNN scores...")
    gnn_val, _ = get_gnn_scores(model, val_graph, device)
    gnn_test, candidate_embeddings = get_gnn_scores(model, test_graph, device)
    gnn_val_debiased = gnn_val - gnn_val.mean(dim=0, keepdim=True)
    gnn_test_debiased = gnn_test - gnn_test.mean(dim=0, keepdim=True)

    # Export GNN corpus embeddings for hybrid search (raw, unnormalized to match scoring)
    if args.export_embeddings:
        emb_np = candidate_embeddings.numpy()
        emb_path = f"{model_dir}/gnn_corpus_embeddings.npy"
        np.save(emb_path, emb_np)
        print(f"  Exported GNN corpus embeddings: {emb_path} {emb_np.shape}")

    bm25_val_cpu = bm25_val_scores.cpu()
    bm25_test_cpu = bm25_test_scores.cpu()

    # Sweep alpha on VAL (not test — no leakage)
    print("\nAlpha grid search on VAL (original):")
    best_alpha, best_mrr = grid_search_alpha(gnn_val, bm25_val_cpu, val_gold_matrix)
    print(f"  Best: alpha={best_alpha:.1f}, MRR@10={best_mrr:.4f}")

    print("Alpha grid search on VAL (debiased):")
    best_alpha_d, best_mrr_d = grid_search_alpha(gnn_val_debiased, bm25_val_cpu, val_gold_matrix)
    print(f"  Best: alpha={best_alpha_d:.1f}, MRR@10={best_mrr_d:.4f}")

    # Pick best variant based on val
    use_debiased = best_mrr_d > best_mrr
    if use_debiased:
        final_gnn = gnn_test_debiased
        final_alpha = best_alpha_d
        final_val_mrr = best_mrr_d
    else:
        final_gnn = gnn_test
        final_alpha = best_alpha
        final_val_mrr = best_mrr

    final_scores = final_alpha * final_gnn + (1 - final_alpha) * bm25_test_cpu
    final_mrr = compute_mrr(final_scores, test_gold_matrix)

    print(f"\nVal-selected: alpha={final_alpha:.1f} ({'debiased' if use_debiased else 'original'}), val MRR={final_val_mrr:.4f}")
    print(f"Test MRR@10 (val-frozen alpha): {final_mrr:.4f}")

    # Export rankings
    out_path = f"{model_dir}/rankings_top{args.top_k}.jsonl"
    print(f"\nExporting top-{args.top_k} rankings...")
    gt_ranks = export_rankings(
        final_scores, test_qids, corpus_doc_ids, test_gold_matrix,
        final_alpha, use_debiased, out_path, args.top_k,
    )

    print(f"\n  Saved: {out_path}")
    print(f"  Queries: {len(test_qids)}")
    if gt_ranks:
        print(f"  GT rank stats: median={np.median(gt_ranks):.0f}, "
              f"mean={np.mean(gt_ranks):.0f}, "
              f"p90={np.percentile(gt_ranks, 90):.0f}, "
              f"in-top-{args.top_k}={sum(1 for r in gt_ranks if r <= args.top_k)}/{len(gt_ranks)}")

    # Save to outputs/predictions/ in standard format
    method_name = {"none": "paragnn", "proximity": "proxgnn", "structural": "structgnn"}[mode]
    top_k = min(args.top_k, final_scores.shape[1])
    std_rankings: dict = {}
    std_scores: dict = {}
    for qi, qid in enumerate(test_qids):
        idx = torch.argsort(final_scores[qi], descending=True)[:top_k].tolist()
        std_rankings[qid] = [corpus_doc_ids[i] for i in idx]
        std_scores[qid] = {corpus_doc_ids[i]: float(final_scores[qi, i]) for i in idx}
    std_gt = {qid: list(test_loader.qrels[qid].keys()) for qid in test_qids if qid in test_loader.qrels}
    save_predictions(std_rankings, std_gt, method=method_name, dataset=args.dataset, scores=std_scores)

    # Also export both original and debiased for analysis
    out_orig = f"{model_dir}/rankings_top{args.top_k}_original.jsonl"
    scores_orig = best_alpha * gnn_test + (1 - best_alpha) * bm25_test_cpu
    gt_orig = export_rankings(
        scores_orig, test_qids, corpus_doc_ids, test_gold_matrix,
        best_alpha, False, out_orig, args.top_k,
    )
    print(f"  Also saved (original): {out_orig}")

    out_deb = f"{model_dir}/rankings_top{args.top_k}_debiased.jsonl"
    scores_deb = best_alpha_d * gnn_test_debiased + (1 - best_alpha_d) * bm25_test_cpu
    gt_deb = export_rankings(
        scores_deb, test_qids, corpus_doc_ids, test_gold_matrix,
        best_alpha_d, True, out_deb, args.top_k,
    )
    print(f"  Also saved (debiased): {out_deb}")

    print(f"\n{'='*60}")
    print(f"  Done. Files in {model_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
