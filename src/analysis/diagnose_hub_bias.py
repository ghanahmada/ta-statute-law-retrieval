"""Diagnose hub bias in StructGNN: norm vs directional.

Loads the EXISTING trained model (trained without L2 norm) and checks:
1. Are hub articles' high scores due to embedding NORM or DIRECTION?
2. Does L2 normalization at inference-only fix the ranking?
3. Do non-hub GT docs have higher cosine similarity than random docs?

Run on GPU VM before committing to retraining:
  python src/analysis/diagnose_hub_bias.py --dataset kuhperdata-humanized

If cosine_nonhub_gt > cosine_random: directional signal exists → L2 norm may help
If cosine_nonhub_gt ≈ cosine_random: signal is weak → need stronger learning signal
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paragnn import DATASETS, ParaGNNConfig
from paragnn.graph_builder import ParagraphStore, GraphBuilder
from paragnn.model import TestCaseGnn
from paragnn.structure import precompute_structure_features, get_query_structure_features
from util.dataloader import DataLoader

HUB_ARTICLES = {"1365", "1865", "1320", "1337", "1234", "188"}


def get_model_dir(config, use_fact_types=False):
    mode = config.structure_mode
    method_suffix = config.method
    if mode == "proximity":
        method_suffix = f"{method_suffix}_prox{config.proximity_radius}"
    elif mode == "structural":
        method_suffix = f"{method_suffix}_struct"
    if use_fact_types:
        method_suffix = f"{method_suffix}_facts"
    return f"{config.output_dir}/{config.dataset}/{method_suffix}"


@torch.no_grad()
def run_diagnostic(config, model_dir, para_store, structure_features,
                   query_structure_feature, bm25_test_scores,
                   test_query_ids, test_corpus_ids, test_gold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config.structure_mode
    dim = config.embed_dim
    struct_input_dim = dim + config.act_dim + config.pos_dim

    model = TestCaseGnn(
        in_dim=dim, h_dim=dim, out_dim=dim,
        dropout=config.dropout, num_head=config.num_heads,
        structure_mode=mode, struct_input_dim=struct_input_dim,
    )
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu"))
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

    # ── Stage 1: Extract embeddings BEFORE EUGAT ──
    node_h = test_graph.ndata["h"]
    if mode == "structural":
        node_h = model.struct_proj(node_h)

    pre_eugat_query = node_h[test_graph.ndata["query_mask"].bool()].cpu()
    pre_eugat_cand = node_h[test_graph.ndata["candidate_mask"].bool()].cpu()

    # ── Stage 2: Extract embeddings AFTER EUGAT ──
    h = model.eugat_gnn(test_graph, node_h, test_graph.edata["h"])
    post_eugat_query = h[test_graph.ndata["query_mask"].bool()].cpu()
    post_eugat_cand = h[test_graph.ndata["candidate_mask"].bool()].cpu()

    # ── Build gold matrix and hub/non-hub split ──
    doc_id_to_idx = {d: i for i, d in enumerate(test_corpus_ids)}
    gold_per_query = {}
    for qi, qid in enumerate(test_query_ids):
        if qid in test_gold:
            gt_ids = {did for did, s in test_gold[qid].items() if s > 0 and did in doc_id_to_idx}
            gold_per_query[qi] = gt_ids

    hub_indices = {doc_id_to_idx[d] for d in HUB_ARTICLES if d in doc_id_to_idx}

    # ── Analysis 1: Norm distribution ──
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: L2 Norm Distribution")
    print("=" * 60)

    for stage_name, cand_emb, query_emb in [
        ("Pre-EUGAT", pre_eugat_cand, pre_eugat_query),
        ("Post-EUGAT", post_eugat_cand, post_eugat_query),
    ]:
        cand_norms = torch.norm(cand_emb, dim=1)
        query_norms = torch.norm(query_emb, dim=1)

        hub_norms = cand_norms[list(hub_indices)]
        non_hub_mask = torch.ones(len(test_corpus_ids), dtype=torch.bool)
        for idx in hub_indices:
            non_hub_mask[idx] = False
        nonhub_norms = cand_norms[non_hub_mask]

        print(f"\n  {stage_name}:")
        print(f"    Query norms:     mean={query_norms.mean():.4f}  std={query_norms.std():.4f}")
        print(f"    All cand norms:  mean={cand_norms.mean():.4f}  std={cand_norms.std():.4f}")
        print(f"    Hub cand norms:  mean={hub_norms.mean():.4f}  std={hub_norms.std():.4f}")
        print(f"    NonHub cand:     mean={nonhub_norms.mean():.4f}  std={nonhub_norms.std():.4f}")
        print(f"    Hub/NonHub ratio: {hub_norms.mean() / nonhub_norms.mean():.4f}")

        # Top-20 by norm
        top_norm_indices = torch.argsort(cand_norms, descending=True)[:20].tolist()
        hub_in_top20 = sum(1 for i in top_norm_indices if i in hub_indices)
        print(f"    Hub articles in top-20 by norm: {hub_in_top20}/{min(len(hub_indices), 20)}")

    # ── Analysis 2: Dot product vs cosine similarity scoring ──
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Dot Product vs Cosine Scoring")
    print("=" * 60)

    cand_emb = post_eugat_cand
    query_emb = post_eugat_query

    dot_scores = torch.matmul(query_emb, cand_emb.T)
    cosine_scores = torch.matmul(
        F.normalize(query_emb, dim=-1),
        F.normalize(cand_emb, dim=-1).T,
    )
    asym_scores = torch.matmul(
        query_emb,
        F.normalize(cand_emb, dim=-1).T,
    )

    gold_matrix = torch.zeros(len(test_query_ids), len(test_corpus_ids))
    for qi, gt_ids in gold_per_query.items():
        for did in gt_ids:
            gold_matrix[qi][doc_id_to_idx[did]] = 1

    for name, raw_scores in [
        ("Dot product (original)", dot_scores),
        ("Cosine (symmetric L2)", cosine_scores),
        ("Asymmetric (cand-only L2)", asym_scores),
    ]:
        # z-normalize
        mean_s = raw_scores.mean(dim=1, keepdim=True)
        std_s = raw_scores.std(dim=1, keepdim=True)
        norm_scores = (raw_scores - mean_s) / (std_s + 1e-8)

        recall = compute_recall(norm_scores, gold_matrix)
        recall_raw = compute_recall(raw_scores, gold_matrix)
        mrr = compute_mrr(norm_scores, gold_matrix)
        mrr_raw = compute_mrr(raw_scores, gold_matrix)

        print(f"\n  {name}:")
        print(f"    Recall@10 (raw):        {recall_raw:.4f}  MRR@10: {mrr_raw:.4f}")
        print(f"    Recall@10 (z-normed):   {recall:.4f}  MRR@10: {mrr:.4f}")

    # ── Analysis 3: Cosine similarity for hub-GT vs non-hub-GT vs random ──
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: Directional Signal (Cosine Similarity)")
    print("=" * 60)

    query_normed = F.normalize(query_emb, dim=-1)
    cand_normed = F.normalize(cand_emb, dim=-1)
    cos_matrix = torch.matmul(query_normed, cand_normed.T)

    hub_gt_cosines = []
    nonhub_gt_cosines = []
    random_cosines = []

    rng = np.random.RandomState(42)

    for qi, gt_ids in gold_per_query.items():
        gt_indices = [doc_id_to_idx[d] for d in gt_ids if d in doc_id_to_idx]
        if not gt_indices:
            continue

        for idx in gt_indices:
            cos_val = cos_matrix[qi, idx].item()
            if idx in hub_indices:
                hub_gt_cosines.append(cos_val)
            else:
                nonhub_gt_cosines.append(cos_val)

        # Sample same number of random non-GT candidates
        non_gt = [j for j in range(len(test_corpus_ids)) if j not in set(gt_indices)]
        sample_size = min(len(gt_indices) * 5, len(non_gt))
        random_indices = rng.choice(non_gt, size=sample_size, replace=False)
        for idx in random_indices:
            random_cosines.append(cos_matrix[qi, idx].item())

    hub_gt_cosines = np.array(hub_gt_cosines) if hub_gt_cosines else np.array([0.0])
    nonhub_gt_cosines = np.array(nonhub_gt_cosines) if nonhub_gt_cosines else np.array([0.0])
    random_cosines = np.array(random_cosines)

    print(f"\n  Hub GT cosine:      mean={hub_gt_cosines.mean():.4f}  std={hub_gt_cosines.std():.4f}  n={len(hub_gt_cosines)}")
    print(f"  Non-hub GT cosine:  mean={nonhub_gt_cosines.mean():.4f}  std={nonhub_gt_cosines.std():.4f}  n={len(nonhub_gt_cosines)}")
    print(f"  Random doc cosine:  mean={random_cosines.mean():.4f}  std={random_cosines.std():.4f}  n={len(random_cosines)}")
    print(f"\n  Gap (non-hub GT - random): {nonhub_gt_cosines.mean() - random_cosines.mean():.4f}")
    print(f"  Gap (hub GT - random):     {hub_gt_cosines.mean() - random_cosines.mean():.4f}")

    if nonhub_gt_cosines.mean() - random_cosines.mean() > 0.02:
        print("\n  → Directional signal EXISTS for non-hub GT. L2 norm + retrain may help.")
    elif nonhub_gt_cosines.mean() - random_cosines.mean() > 0.005:
        print("\n  → Weak directional signal. L2 norm alone is unlikely sufficient.")
    else:
        print("\n  → No directional signal for non-hub GT. Need different approach.")

    # ── Analysis 4: Per-hub-article frequency in top-k ──
    print("\n" + "=" * 60)
    print("  ANALYSIS 4: Hub Article Frequency (Dot vs Cosine)")
    print("=" * 60)

    for name, scores in [("Dot+z-norm", dot_scores), ("Cosine", cos_matrix)]:
        if name == "Dot+z-norm":
            mean_s = scores.mean(dim=1, keepdim=True)
            std_s = scores.std(dim=1, keepdim=True)
            scores = (scores - mean_s) / (std_s + 1e-8)

        freq = defaultdict(int)
        for qi in range(len(test_query_ids)):
            top10 = torch.argsort(scores[qi], descending=True)[:10].tolist()
            for idx in top10:
                if idx in hub_indices:
                    doc_id = test_corpus_ids[idx]
                    freq[doc_id] += 1

        n_queries = len(test_query_ids)
        print(f"\n  {name} — hub articles in top-10:")
        for doc_id, count in sorted(freq.items(), key=lambda x: -x[1]):
            print(f"    {doc_id}: {count}/{n_queries} queries ({count/n_queries:.1%})")

    # ── Analysis 5: MRR split by hub/non-hub GT ──
    print("\n" + "=" * 60)
    print("  ANALYSIS 5: MRR@10 Split (Hub-GT vs Non-Hub-GT Queries)")
    print("=" * 60)

    for name, scores in [
        ("Dot+z-norm (original)", dot_scores),
        ("Cosine (symmetric)", cos_matrix),
    ]:
        if "Dot" in name:
            mean_s = scores.mean(dim=1, keepdim=True)
            std_s = scores.std(dim=1, keepdim=True)
            scores = (scores - mean_s) / (std_s + 1e-8)

        hub_recall_sum, hub_mrr_sum, hub_count = 0, 0, 0
        nonhub_recall_sum, nonhub_mrr_sum, nonhub_count = 0, 0, 0

        for qi, gt_ids in gold_per_query.items():
            gt_indices = set(doc_id_to_idx[d] for d in gt_ids if d in doc_id_to_idx)
            if not gt_indices:
                continue

            has_hub_gt = any(idx in hub_indices for idx in gt_indices)

            ranked_list = torch.argsort(scores[qi], descending=True)[:10].tolist()
            ranked_set = set(ranked_list)
            recall_q = len(ranked_set & gt_indices) / len(gt_indices)
            rr = 0.0
            for rank, idx in enumerate(ranked_list):
                if idx in gt_indices:
                    rr = 1.0 / (rank + 1)
                    break

            if has_hub_gt:
                hub_recall_sum += recall_q
                hub_mrr_sum += rr
                hub_count += 1
            else:
                nonhub_recall_sum += recall_q
                nonhub_mrr_sum += rr
                nonhub_count += 1

        hub_recall = hub_recall_sum / hub_count if hub_count else 0
        hub_mrr = hub_mrr_sum / hub_count if hub_count else 0
        nonhub_recall = nonhub_recall_sum / nonhub_count if nonhub_count else 0
        nonhub_mrr = nonhub_mrr_sum / nonhub_count if nonhub_count else 0

        print(f"\n  {name}:")
        print(f"    Hub-GT queries:     Recall@10={hub_recall:.4f}  MRR@10={hub_mrr:.4f}  (n={hub_count})")
        print(f"    Non-hub-GT queries: Recall@10={nonhub_recall:.4f}  MRR@10={nonhub_mrr:.4f}  (n={nonhub_count})")


def compute_recall(scores, gold_matrix, k=10):
    recall_sum = 0
    n_queries = gold_matrix.shape[0]
    for qi in range(n_queries):
        relevant = set(gold_matrix[qi].nonzero(as_tuple=True)[0].tolist())
        if not relevant:
            continue
        ranked = set(torch.argsort(scores[qi], descending=True)[:k].tolist())
        recall_sum += len(ranked & relevant) / len(relevant)
    return recall_sum / n_queries


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


def main():
    parser = argparse.ArgumentParser(description="Diagnose hub bias: norm vs direction")
    parser.add_argument("--dataset", default="kuhperdata-humanized")
    parser.add_argument("--method", default="adapted")
    parser.add_argument("--structure_mode", default="structural")
    parser.add_argument("--proximity_radius", type=int, default=50)
    parser.add_argument("--act_dim", type=int, default=64)
    parser.add_argument("--pos_dim", type=int, default=32)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--use_fact_types", action="store_true")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    config = ParaGNNConfig(
        dataset=args.dataset,
        data_path=cfg["path"],
        lang=cfg["lang"],
        method=args.method,
        structure_mode=args.structure_mode,
        proximity_radius=args.proximity_radius,
        act_dim=args.act_dim,
        pos_dim=args.pos_dim,
        max_relevant=args.max_relevant,
    )

    precompute_dir = f"{config.output_dir}/{args.dataset}"
    model_dir = get_model_dir(config, use_fact_types=args.use_fact_types)

    print("Loading data...")
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

    para_store = ParagraphStore(output_dir=precompute_dir, method=args.method,
                                use_fact_types=args.use_fact_types)

    structure_features = None
    query_structure_feature = None
    if args.structure_mode == "structural":
        corpus_path = f"{config.data_path}/corpus.jsonl"
        structure_features = precompute_structure_features(
            corpus_path, args.dataset, act_dim=args.act_dim, pos_dim=args.pos_dim
        )
        query_act, query_pos = get_query_structure_features(
            act_dim=args.act_dim, pos_dim=args.pos_dim
        )
        query_structure_feature = torch.cat([query_act, query_pos])

    run_diagnostic(
        config, model_dir, para_store, structure_features,
        query_structure_feature, bm25_test_scores,
        test_qids, corpus_doc_ids, test_loader.qrels,
    )


if __name__ == "__main__":
    main()
