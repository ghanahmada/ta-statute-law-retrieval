"""Analysis 03 Part 2: Embedding Analysis for StructGNN.

Five analyses showing how StructGNN transforms embedding space:
  A. similarity    — co-relevant vs hard-negative cosine sim in GNN vs BGE-M3 space
  B. before_after  — Para-GNN vs StructGNN embedding distances (needs both embeddings)
  C. neighborhood  — co-relevant articles appear in each other's NN in GNN vs BGE-M3 space
  D. separation    — Cohen's d and AUC for co-relevant vs hard-negative distributions
  E. collapse      — SVD singular value decay, effective rank, isotropy check

Usage:
  python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis similarity
  python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis neighborhood
  python src/analysis/embedding_analysis.py --dataset coliee --analysis before_after
  python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis separation
  python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis collapse
  python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis all
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path("data")
PARAGNN_DIR = Path("outputs/paragnn")
PRED_DIR = Path("outputs/predictions")
OUTPUT_DIR = Path("outputs/analysis/embedding_analysis")


def load_qrels_test(dataset: str) -> dict[str, list[str]]:
    path = DATA_DIR / dataset / "qrels_test.tsv"
    qrels = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                qrels[parts[0]].append(parts[1])
    return dict(qrels)


def load_doc_ids(dataset: str) -> list[str]:
    path = PARAGNN_DIR / dataset / "corpus_doc_ids.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_gnn_embeddings(dataset: str, model_type: str = "adapted_struct") -> np.ndarray:
    path = PARAGNN_DIR / dataset / model_type / "gnn_corpus_embeddings.npy"
    if not path.exists():
        return None
    return np.load(str(path))


def load_bge_embeddings(dataset: str, doc_ids: list[str]) -> np.ndarray:
    import torch
    emb_dir = PARAGNN_DIR / dataset / "embeddings" / "corpus"
    embeddings = []
    for doc_id in doc_ids:
        pt_path = emb_dir / f"{doc_id}.pt"
        if pt_path.exists():
            t = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            if t.dim() == 2:
                t = t.mean(dim=0)
            embeddings.append(t.numpy())
        else:
            embeddings.append(np.zeros(1024))
    return np.stack(embeddings)


def load_bm25_predictions(dataset: str) -> dict[str, list[str]]:
    path = PRED_DIR / f"bm25_{dataset}.jsonl"
    if not path.exists():
        return {}
    rankings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rankings[rec["qid"]] = rec["ranked_doc_ids"]
    return rankings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    return normalized @ normalized.T


def get_hard_negatives(qrels: dict, bm25_rankings: dict, doc_id_to_idx: dict, top_k: int = 100) -> dict[str, list[str]]:
    """For each query, get BM25 top-K docs that are NOT relevant (hard negatives)."""
    hard_negs = {}
    for qid, rel_docs in qrels.items():
        rel_set = set(rel_docs)
        bm25_ranked = bm25_rankings.get(qid, [])[:top_k]
        negs = [d for d in bm25_ranked if d not in rel_set and d in doc_id_to_idx]
        if negs:
            hard_negs[qid] = negs
    return hard_negs


def analysis_similarity(dataset: str, doc_ids: list[str], gnn_emb: np.ndarray,
                         bge_emb: np.ndarray, qrels: dict, n_random: int = 1000):
    """A. Compare cosine similarity of co-relevant vs hard-negative pairs."""
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS A: EMBEDDING SIMILARITY — {dataset}")
    print(f"{'=' * 80}")

    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    bm25_rankings = load_bm25_predictions(dataset)
    hard_negs = get_hard_negatives(qrels, bm25_rankings, doc_id_to_idx)

    # Co-relevant pairs
    corel_gnn_sims = []
    corel_bge_sims = []
    for qid, rel_docs in qrels.items():
        valid = [d for d in rel_docs if d in doc_id_to_idx]
        if len(valid) < 2:
            continue
        for d_i, d_j in combinations(valid, 2):
            i, j = doc_id_to_idx[d_i], doc_id_to_idx[d_j]
            corel_gnn_sims.append(cosine_sim(gnn_emb[i], gnn_emb[j]))
            corel_bge_sims.append(cosine_sim(bge_emb[i], bge_emb[j]))

    # Hard-negative pairs (relevant doc vs BM25-retrieved non-relevant doc)
    hardneg_gnn_sims = []
    hardneg_bge_sims = []
    rng = np.random.default_rng(42)
    for qid, rel_docs in qrels.items():
        valid_rel = [d for d in rel_docs if d in doc_id_to_idx]
        negs = hard_negs.get(qid, [])
        if not valid_rel or not negs:
            continue
        sampled_negs = negs[:min(10, len(negs))]
        for rel_doc in valid_rel:
            rel_idx = doc_id_to_idx[rel_doc]
            for neg_doc in sampled_negs:
                neg_idx = doc_id_to_idx[neg_doc]
                hardneg_gnn_sims.append(cosine_sim(gnn_emb[rel_idx], gnn_emb[neg_idx]))
                hardneg_bge_sims.append(cosine_sim(bge_emb[rel_idx], bge_emb[neg_idx]))

    # Random pairs (for reference)
    random_gnn_sims = []
    random_bge_sims = []
    n_docs = len(doc_ids)
    for _ in range(n_random):
        i, j = rng.choice(n_docs, size=2, replace=False)
        random_gnn_sims.append(cosine_sim(gnn_emb[i], gnn_emb[j]))
        random_bge_sims.append(cosine_sim(bge_emb[i], bge_emb[j]))

    print(f"\n  Co-relevant pairs: {len(corel_gnn_sims)}")
    print(f"  Hard-negative pairs: {len(hardneg_gnn_sims)}")
    print(f"  Random pairs: {n_random}")
    print(f"\n  {'':20} {'Co-relevant':>12} {'Hard-neg':>12} {'Random':>12}")
    print(f"  {'-'*60}")
    print(f"  {'GNN (StructGNN)':<20} {np.mean(corel_gnn_sims):>12.4f} {np.mean(hardneg_gnn_sims):>12.4f} {np.mean(random_gnn_sims):>12.4f}")
    print(f"  {'BGE-M3 (raw)':<20} {np.mean(corel_bge_sims):>12.4f} {np.mean(hardneg_bge_sims):>12.4f} {np.mean(random_bge_sims):>12.4f}")
    print(f"\n  Separation (co-relevant - hard-negative):")
    gnn_sep = np.mean(corel_gnn_sims) - np.mean(hardneg_gnn_sims)
    bge_sep = np.mean(corel_bge_sims) - np.mean(hardneg_bge_sims)
    print(f"    GNN:   {gnn_sep:+.4f}")
    print(f"    BGE:   {bge_sep:+.4f}")
    print(f"    Ratio: {gnn_sep / max(abs(bge_sep), 1e-6):.1f}x")

    return {
        "corel_gnn_mean": float(np.mean(corel_gnn_sims)),
        "corel_bge_mean": float(np.mean(corel_bge_sims)),
        "hardneg_gnn_mean": float(np.mean(hardneg_gnn_sims)),
        "hardneg_bge_mean": float(np.mean(hardneg_bge_sims)),
        "random_gnn_mean": float(np.mean(random_gnn_sims)),
        "random_bge_mean": float(np.mean(random_bge_sims)),
        "n_corel_pairs": len(corel_gnn_sims),
        "n_hardneg_pairs": len(hardneg_gnn_sims),
        "corel_gnn_sims": corel_gnn_sims,
        "corel_bge_sims": corel_bge_sims,
        "hardneg_gnn_sims": hardneg_gnn_sims,
        "hardneg_bge_sims": hardneg_bge_sims,
    }


def analysis_before_after(dataset: str, doc_ids: list[str], structgnn_emb: np.ndarray,
                           paragnn_emb: np.ndarray, qrels: dict, n_random: int = 1000):
    """B. Compare Para-GNN vs StructGNN embeddings for co-relevant pairs."""
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS B: BEFORE/AFTER STRUCTURAL FEATURES — {dataset}")
    print(f"{'=' * 80}")

    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    corel_struct_sims = []
    corel_para_sims = []
    for qid, rel_docs in qrels.items():
        valid = [d for d in rel_docs if d in doc_id_to_idx]
        if len(valid) < 2:
            continue
        for d_i, d_j in combinations(valid, 2):
            i, j = doc_id_to_idx[d_i], doc_id_to_idx[d_j]
            corel_struct_sims.append(cosine_sim(structgnn_emb[i], structgnn_emb[j]))
            corel_para_sims.append(cosine_sim(paragnn_emb[i], paragnn_emb[j]))

    rng = np.random.default_rng(42)
    random_struct_sims = []
    random_para_sims = []
    n_docs = len(doc_ids)
    for _ in range(n_random):
        i, j = rng.choice(n_docs, size=2, replace=False)
        random_struct_sims.append(cosine_sim(structgnn_emb[i], structgnn_emb[j]))
        random_para_sims.append(cosine_sim(paragnn_emb[i], paragnn_emb[j]))

    print(f"\n  Co-relevant pairs: {len(corel_struct_sims)}")
    print(f"\n  {'':20} {'Co-relevant':>12} {'Random':>12} {'Δ(corel-rand)':>14}")
    print(f"  {'-'*60}")
    struct_delta = np.mean(corel_struct_sims) - np.mean(random_struct_sims)
    para_delta = np.mean(corel_para_sims) - np.mean(random_para_sims)
    print(f"  {'StructGNN':<20} {np.mean(corel_struct_sims):>12.4f} {np.mean(random_struct_sims):>12.4f} {struct_delta:>+13.4f}")
    print(f"  {'Para-GNN':<20} {np.mean(corel_para_sims):>12.4f} {np.mean(random_para_sims):>12.4f} {para_delta:>+13.4f}")
    print(f"\n  Structural features increase co-relevant separation by: {struct_delta - para_delta:+.4f}")

    return {
        "corel_struct_mean": float(np.mean(corel_struct_sims)),
        "corel_para_mean": float(np.mean(corel_para_sims)),
        "random_struct_mean": float(np.mean(random_struct_sims)),
        "random_para_mean": float(np.mean(random_para_sims)),
    }


def analysis_neighborhood(dataset: str, doc_ids: list[str], gnn_emb: np.ndarray,
                            bge_emb: np.ndarray, qrels: dict, k_values: list[int] = None):
    """C. Neighborhood coverage: do co-relevant articles appear in each other's NN?"""
    if k_values is None:
        k_values = [5, 10, 20, 50]

    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS C: NEIGHBORHOOD COVERAGE — {dataset}")
    print(f"{'=' * 80}")

    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    print(f"  Computing similarity matrices ({len(doc_ids)} docs)...", flush=True)
    gnn_sim_matrix = cosine_sim_matrix(gnn_emb)
    bge_sim_matrix = cosine_sim_matrix(bge_emb)

    gnn_coverage = {k: [] for k in k_values}
    bge_coverage = {k: [] for k in k_values}

    for qid, rel_docs in qrels.items():
        valid = [d for d in rel_docs if d in doc_id_to_idx]
        if len(valid) < 2:
            continue
        valid_idxs = [doc_id_to_idx[d] for d in valid]

        for anchor_idx in valid_idxs:
            other_idxs = set(valid_idxs) - {anchor_idx}

            gnn_scores = gnn_sim_matrix[anchor_idx]
            gnn_ranked = np.argsort(-gnn_scores)

            bge_scores = bge_sim_matrix[anchor_idx]
            bge_ranked = np.argsort(-bge_scores)

            for k in k_values:
                gnn_topk = set(gnn_ranked[1:k+1])
                bge_topk = set(bge_ranked[1:k+1])
                gnn_hits = len(other_idxs & gnn_topk) / len(other_idxs)
                bge_hits = len(other_idxs & bge_topk) / len(other_idxs)
                gnn_coverage[k].append(gnn_hits)
                bge_coverage[k].append(bge_hits)

    print(f"\n  Queries with >= 2 relevant docs evaluated")
    print(f"\n  {'K':>5} {'GNN Coverage':>14} {'BGE Coverage':>14} {'Delta':>8} {'Ratio':>7}")
    print(f"  {'-'*50}")
    results = {}
    for k in k_values:
        gnn_mean = float(np.mean(gnn_coverage[k]))
        bge_mean = float(np.mean(bge_coverage[k]))
        delta = gnn_mean - bge_mean
        ratio = gnn_mean / max(bge_mean, 1e-6)
        print(f"  {k:>5} {gnn_mean:>13.1%} {bge_mean:>13.1%} {delta:>+7.1%} {ratio:>6.1f}x")
        results[k] = {"gnn": gnn_mean, "bge": bge_mean}

    return results


def analysis_separation(dataset: str, doc_ids: list[str], gnn_emb: np.ndarray,
                         bge_emb: np.ndarray, qrels: dict, sim_results: dict = None):
    """D. Cohen's d and AUC for co-relevant vs hard-negative distributions."""
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS D: SEPARATION METRICS — {dataset}")
    print(f"{'=' * 80}")

    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    if sim_results and "corel_gnn_sims" in sim_results:
        corel_gnn = np.array(sim_results["corel_gnn_sims"])
        corel_bge = np.array(sim_results["corel_bge_sims"])
        hardneg_gnn = np.array(sim_results["hardneg_gnn_sims"])
        hardneg_bge = np.array(sim_results["hardneg_bge_sims"])
    else:
        bm25_rankings = load_bm25_predictions(dataset)
        hard_negs = get_hard_negatives(qrels, bm25_rankings, doc_id_to_idx)

        corel_gnn, corel_bge = [], []
        for qid, rel_docs in qrels.items():
            valid = [d for d in rel_docs if d in doc_id_to_idx]
            if len(valid) < 2:
                continue
            for d_i, d_j in combinations(valid, 2):
                i, j = doc_id_to_idx[d_i], doc_id_to_idx[d_j]
                corel_gnn.append(cosine_sim(gnn_emb[i], gnn_emb[j]))
                corel_bge.append(cosine_sim(bge_emb[i], bge_emb[j]))

        hardneg_gnn, hardneg_bge = [], []
        for qid, rel_docs in qrels.items():
            valid_rel = [d for d in rel_docs if d in doc_id_to_idx]
            negs = hard_negs.get(qid, [])
            if not valid_rel or not negs:
                continue
            for rel_doc in valid_rel:
                rel_idx = doc_id_to_idx[rel_doc]
                for neg_doc in negs[:10]:
                    neg_idx = doc_id_to_idx[neg_doc]
                    hardneg_gnn.append(cosine_sim(gnn_emb[rel_idx], gnn_emb[neg_idx]))
                    hardneg_bge.append(cosine_sim(bge_emb[rel_idx], bge_emb[neg_idx]))

        corel_gnn = np.array(corel_gnn)
        corel_bge = np.array(corel_bge)
        hardneg_gnn = np.array(hardneg_gnn)
        hardneg_bge = np.array(hardneg_bge)

    if len(corel_gnn) == 0 or len(hardneg_gnn) == 0:
        print("  SKIP: insufficient co-relevant or hard-negative pairs")
        return {}

    def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
        n1, n2 = len(pos), len(neg)
        var1, var2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(pos) - np.mean(neg)) / pooled_std)

    def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        scores = np.concatenate([pos, neg])
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        n_pos = len(pos)
        n_neg = len(neg)
        tp = 0
        auc = 0.0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                auc += tp
        return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.0

    gnn_d = cohens_d(corel_gnn, hardneg_gnn)
    bge_d = cohens_d(corel_bge, hardneg_bge)
    gnn_auc = auc_score(corel_gnn, hardneg_gnn)
    bge_auc = auc_score(corel_bge, hardneg_bge)

    print(f"\n  Co-relevant pairs: {len(corel_gnn)}, Hard-negative pairs: {len(hardneg_gnn)}")
    print(f"\n  {'Metric':<12} {'GNN (StructGNN)':>16} {'BGE-M3 (raw)':>16} {'Interpretation'}")
    print(f"  {'-'*70}")
    print(f"  {'Cohen d':<12} {gnn_d:>16.3f} {bge_d:>16.3f}   {'(>0.8 = large effect)' if max(gnn_d, bge_d) > 0.8 else '(>0.5 = medium)'}")
    print(f"  {'AUC':<12} {gnn_auc:>16.3f} {bge_auc:>16.3f}   {'(1.0 = perfect separation)'}")

    print(f"\n  GNN vs BGE-M3:")
    print(f"    Cohen's d: {gnn_d:.3f} vs {bge_d:.3f} ({gnn_d/max(bge_d, 1e-6):.1f}x)")
    print(f"    AUC:       {gnn_auc:.3f} vs {bge_auc:.3f}")

    return {
        "gnn_cohens_d": gnn_d,
        "bge_cohens_d": bge_d,
        "gnn_auc": gnn_auc,
        "bge_auc": bge_auc,
        "n_positive": len(corel_gnn),
        "n_negative": len(hardneg_gnn),
    }


def analysis_collapse(dataset: str, doc_ids: list[str], gnn_emb: np.ndarray,
                       bge_emb: np.ndarray):
    """E. SVD singular value decay + effective rank (collapse check)."""
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS E: EMBEDDING COLLAPSE CHECK — {dataset}")
    print(f"{'=' * 80}")

    def compute_svd_stats(emb: np.ndarray, name: str) -> dict:
        centered = emb - emb.mean(axis=0)
        # Use truncated SVD for efficiency (top-200 singular values are enough)
        n_components = min(200, min(emb.shape) - 1)
        try:
            from scipy.sparse.linalg import svds
            _, s, _ = svds(centered, k=n_components)
            s = np.sort(s)[::-1]
        except Exception:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            s = s[:n_components]

        variance = s ** 2
        cumvar = np.cumsum(variance) / np.sum(variance)

        # Effective rank: dimensions for 90% variance
        eff_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        eff_rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1

        # Spectral entropy (normalized, 1.0 = perfectly isotropic)
        p = variance / variance.sum()
        p = p[p > 0]
        spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(p))

        # Top-1 dominance: fraction of variance in first singular value
        top1_ratio = float(variance[0] / variance.sum())
        top10_ratio = float(variance[:10].sum() / variance.sum())

        # Average pairwise cosine (isotropy proxy on a sample)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(emb), size=min(500, len(emb)), replace=False)
        sample = emb[sample_idx]
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = sample / norms
        sim_matrix = normalized @ normalized.T
        n = len(sample)
        triu_idx = np.triu_indices(n, k=1)
        avg_cos = float(sim_matrix[triu_idx].mean())

        print(f"\n  {name} ({emb.shape[0]} docs x {emb.shape[1]} dims):")
        print(f"    Effective rank @90% variance: {eff_rank_90}")
        print(f"    Effective rank @95% variance: {eff_rank_95}")
        print(f"    Effective rank @99% variance: {eff_rank_99}")
        print(f"    Top-1 SV variance ratio:      {top1_ratio:.4f}")
        print(f"    Top-10 SV variance ratio:     {top10_ratio:.4f}")
        print(f"    Spectral entropy (norm):      {spectral_entropy:.4f} (1.0 = isotropic)")
        print(f"    Avg pairwise cosine:          {avg_cos:.4f} (0.0 = isotropic)")

        if avg_cos > 0.8:
            print(f"    WARNING: high avg cosine ({avg_cos:.3f}) suggests dimensional collapse")
        elif avg_cos > 0.5:
            print(f"    NOTE: moderate avg cosine ({avg_cos:.3f}) — some anisotropy present")

        return {
            "shape": list(emb.shape),
            "eff_rank_90": eff_rank_90,
            "eff_rank_95": eff_rank_95,
            "eff_rank_99": eff_rank_99,
            "top1_variance_ratio": top1_ratio,
            "top10_variance_ratio": top10_ratio,
            "spectral_entropy": spectral_entropy,
            "avg_pairwise_cosine": avg_cos,
            "singular_values_top20": s[:20].tolist(),
        }

    gnn_stats = compute_svd_stats(gnn_emb, "StructGNN")
    bge_stats = compute_svd_stats(bge_emb, "BGE-M3")

    print(f"\n  COMPARISON:")
    print(f"    {'Metric':<28} {'StructGNN':>12} {'BGE-M3':>12}")
    print(f"    {'-'*55}")
    print(f"    {'Eff rank @90%':<28} {gnn_stats['eff_rank_90']:>12} {bge_stats['eff_rank_90']:>12}")
    print(f"    {'Eff rank @95%':<28} {gnn_stats['eff_rank_95']:>12} {bge_stats['eff_rank_95']:>12}")
    print(f"    {'Spectral entropy':<28} {gnn_stats['spectral_entropy']:>12.4f} {bge_stats['spectral_entropy']:>12.4f}")
    print(f"    {'Avg pairwise cosine':<28} {gnn_stats['avg_pairwise_cosine']:>12.4f} {bge_stats['avg_pairwise_cosine']:>12.4f}")

    return {"gnn": gnn_stats, "bge": bge_stats}


def generate_plots(dataset: str, doc_ids: list[str], gnn_emb: np.ndarray,
                   bge_emb: np.ndarray, qrels: dict, output_dir: Path,
                   sim_results: dict = None, collapse_results: dict = None):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Similarity histogram (co-relevant vs hard-negative)
    if sim_results and "corel_gnn_sims" in sim_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(sim_results["corel_bge_sims"], bins=30, alpha=0.7, label="Co-relevant", color="steelblue")
        axes[0].hist(sim_results["hardneg_bge_sims"], bins=30, alpha=0.5, label="Hard-negative", color="salmon")
        axes[0].set_title("BGE-M3 (Raw)")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].legend()

        axes[1].hist(sim_results["corel_gnn_sims"], bins=30, alpha=0.7, label="Co-relevant", color="darkorange")
        axes[1].hist(sim_results["hardneg_gnn_sims"], bins=30, alpha=0.5, label="Hard-negative", color="salmon")
        axes[1].set_title("StructGNN")
        axes[1].set_xlabel("Cosine Similarity")
        axes[1].legend()

        plt.suptitle(f"Cosine Similarity: Co-relevant vs Hard-negative — {dataset}")
        plt.tight_layout()
        plt.savefig(output_dir / f"similarity_hist_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {output_dir / f'similarity_hist_{dataset}.png'}")

    # Plot 2: Singular value decay
    if collapse_results and "gnn" in collapse_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        gnn_sv = collapse_results["gnn"]["singular_values_top20"]
        bge_sv = collapse_results["bge"]["singular_values_top20"]

        # Normalize to make comparable
        gnn_sv_norm = np.array(gnn_sv) / gnn_sv[0]
        bge_sv_norm = np.array(bge_sv) / bge_sv[0]

        ax.plot(range(1, len(gnn_sv_norm) + 1), gnn_sv_norm, 'o-', label="StructGNN", color="darkorange")
        ax.plot(range(1, len(bge_sv_norm) + 1), bge_sv_norm, 's-', label="BGE-M3", color="steelblue")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized Singular Value (/ SV1)")
        ax.set_title(f"Singular Value Decay — {dataset}")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"svd_decay_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {output_dir / f'svd_decay_{dataset}.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--analysis", default="all",
                        choices=["similarity", "before_after", "neighborhood", "separation", "collapse", "all"])
    parser.add_argument("--model_type", default="adapted_struct",
                        help="Subfolder under outputs/paragnn/<dataset>/ containing StructGNN embeddings")
    parser.add_argument("--paragnn_type", default="adapted",
                        help="Subfolder under outputs/paragnn/<dataset>/ containing Para-GNN embeddings")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset

    print(f"  Loading data for {dataset}...", flush=True)
    doc_ids = load_doc_ids(dataset)
    qrels = load_qrels_test(dataset)

    gnn_emb = load_gnn_embeddings(dataset, args.model_type)
    if gnn_emb is None:
        sys.exit(f"  ERROR: No StructGNN embeddings for {dataset}")

    results = {"dataset": dataset}

    run_sim = args.analysis in ("similarity", "all")
    run_ba = args.analysis in ("before_after", "all")
    run_nn = args.analysis in ("neighborhood", "all")
    run_sep = args.analysis in ("separation", "all")
    run_col = args.analysis in ("collapse", "all")

    # Load BGE-M3 if needed
    bge_emb = None
    if run_sim or run_nn or run_sep or run_col:
        print(f"  Loading BGE-M3 embeddings ({len(doc_ids)} docs)...", flush=True)
        bge_emb = load_bge_embeddings(dataset, doc_ids)

    sim_results_full = None
    if run_sim and bge_emb is not None:
        sim_results_full = analysis_similarity(dataset, doc_ids, gnn_emb, bge_emb, qrels)
        results["similarity"] = {k: v for k, v in sim_results_full.items()
                                  if not isinstance(v, list)}

    if run_ba:
        paragnn_emb = load_gnn_embeddings(dataset, args.paragnn_type)
        if paragnn_emb is not None:
            results["before_after"] = analysis_before_after(dataset, doc_ids, gnn_emb, paragnn_emb, qrels)
        else:
            print(f"\n  SKIP before_after: no Para-GNN embeddings for {dataset}")

    if run_nn and bge_emb is not None:
        results["neighborhood"] = analysis_neighborhood(dataset, doc_ids, gnn_emb, bge_emb, qrels)

    if run_sep and bge_emb is not None:
        results["separation"] = analysis_separation(dataset, doc_ids, gnn_emb, bge_emb, qrels, sim_results_full)

    collapse_results = None
    if run_col and bge_emb is not None:
        collapse_results = analysis_collapse(dataset, doc_ids, gnn_emb, bge_emb)
        results["collapse"] = collapse_results

    # Plots
    if not args.no_plots and bge_emb is not None:
        print(f"\n  Generating plots...", flush=True)
        generate_plots(dataset, doc_ids, gnn_emb, bge_emb, qrels, output_dir,
                       sim_results=sim_results_full, collapse_results=collapse_results)

    # Save results (exclude raw sim lists from JSON)
    results_path = output_dir / f"embedding_results_{dataset}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
