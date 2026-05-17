"""
Explainability Analysis: Para-GNN vs StructGNN on Statute Retrieval
====================================================================

Research Questions
------------------
RQ1 [PROXIMITY LEARNING]
    Does StructGNN learn that structurally adjacent articles are semantically
    related? We measure whether articles physically near each other within an act
    (e.g., KUHPerdata Pasal 1365–1370) are also closer in embedding space under
    StructGNN than under Para-GNN.
    Hypothesis: StructGNN's act-hash + position encoding induces a geometry where
    embedding distance correlates with structural distance; Para-GNN does not.

RQ2 [HUB BIAS]
    Does hub bias persist in Para-GNN after IPS weighting, and does StructGNN
    exacerbate or alleviate it? Hub articles (cited by many queries) may dominate
    rankings, hurting retrieval of long-tail, niche provisions.
    Hypothesis: Both models rank hub articles disproportionately; StructGNN may
    partially alleviate hub bias by anchoring representations to act structure
    rather than citation frequency.

RQ3 [NON-HUB DIRECTIONAL SIGNAL]
    Do Para-GNN and StructGNN embeddings carry meaningful directional signal for
    non-hub (long-tail) articles, or does the model primarily learn to rank hub
    articles highly regardless of query?
    Hypothesis: StructGNN has stronger cosine signal for non-hub GT docs because
    structural features constrain the embedding space to act context.

RQ4 [RETRIEVAL SCORE DECOMPOSITION]
    How does each model's GNN score correlate with BM25 on hub vs non-hub
    articles? A strong GNN–BM25 correlation on hub articles suggests the GNN
    has learned spurious frequency associations rather than semantic matching.
    Hypothesis: GNN scores are more independent of BM25 in StructGNN due to the
    additional structural signal.

Engineering Implications
------------------------
These analyses directly inform production system design:

  1. HUB DETECTION MONITOR
     Track the entropy of top-K distributions at query time. If entropy is low
     (always the same articles), the model has hub-collapsed. Alert before users
     notice degraded long-tail coverage.

  2. RETRIEVAL AUGMENTATION VIA EMBEDDING NEIGHBORHOOD
     If article 1365 is retrieved, its k-nearest embedding neighbors (1366, 1367)
     are likely co-relevant. This enables automatic "related article" expansion
     without extra model calls — a cheap, interpretable fallback for edge cases.

  3. ALPHA AS CONFIDENCE SIGNAL
     The GNN blending weight alpha ∈ [0,1] tells you when to trust the neural
     retriever vs BM25. In production, use alpha < threshold as a routing signal
     to a safer lexical fallback or human review queue.

  4. STRUCTURAL PROXIMITY FOR COLD-START GENERALIZATION
     For new legal domains (zero-shot), structural position encoding provides
     signal even before any query-document pairs are seen — articles in the same
     act at similar positions are likely to share relevance patterns. This is
     especially useful for low-resource legal corpora.

Usage
-----
    # Run all analyses (needs inference output + exported embeddings)
    python src/analysis/gnn_structure_explainability.py \\
        --dataset kuhperdata-humanized \\
        --paragnn_dir outputs/inference/kuhperdata-humanized/none_adapted \\
        --structgnn_dir outputs/inference/kuhperdata-humanized/structural_adapted \\
        --paragnn_emb outputs/paragnn/kuhperdata-humanized/adapted/gnn_corpus_embeddings.npy \\
        --structgnn_emb outputs/paragnn/kuhperdata-humanized/adapted_struct/gnn_corpus_embeddings.npy \\
        --output_dir outputs/analysis/gnn_explainability/kuhperdata-humanized

    # Skip embedding analyses (if .npy files not available)
    python src/analysis/gnn_structure_explainability.py \\
        --dataset kuhperdata-humanized \\
        --paragnn_dir outputs/inference/kuhperdata-humanized/none_adapted \\
        --structgnn_dir outputs/inference/kuhperdata-humanized/structural_adapted
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.dataloader import DataLoader

# ── Seaborn global style ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"Para-GNN": "#4C72B0", "StructGNN": "#DD8452", "BM25": "#55A868"}

HUB_THRESHOLD_DEFAULT = 8   # docs cited by >= this many queries = "hub"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_inference_json(path: str) -> dict:
    """Load infer_paragnn.py output JSON. Returns full dict."""
    with open(path) as f:
        return json.load(f)


def load_corpus(corpus_path: str) -> dict:
    """Returns {doc_id: {"title": ..., "text": ...}}."""
    corpus = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[str(doc["_id"])] = doc
    return corpus


def load_qrels(qrels_path: str) -> dict:
    """Returns {qid: set of relevant doc_ids}."""
    qrels = {}
    with open(qrels_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def compute_doc_freq(qrels: dict) -> Counter:
    return Counter(d for docs in qrels.values() for d in docs)


def get_rankings_from_json(data: dict) -> dict:
    """Returns {qid: [doc_id, ...]} top-K ordered."""
    rankings = {}
    for qid, qdata in data["results"].items():
        rankings[qid] = [r["doc_id"] for r in sorted(qdata["ranked"], key=lambda x: x["rank"])]
    return rankings


def get_scores_from_json(data: dict) -> dict:
    """Returns {qid: {doc_id: {"gnn_score": float, "bm25_score": float}}}."""
    scores = {}
    for qid, qdata in data["results"].items():
        scores[qid] = {
            r["doc_id"]: {"gnn": r.get("gnn_score", 0), "bm25": r.get("bm25_score", 0)}
            for r in qdata["ranked"]
        }
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def mrr_at_k(rankings: dict, qrels: dict, k: int = 10) -> float:
    total = 0.0
    for qid, ranked in rankings.items():
        relevant = qrels.get(qid, set())
        for rank, doc_id in enumerate(ranked[:k], 1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / len(rankings) if rankings else 0.0


def mrr_split_hub_nonhub(
    rankings: dict,
    qrels: dict,
    doc_freq: Counter,
    hub_threshold: int,
    k: int = 10,
) -> tuple[dict, dict]:
    hub_mrrs, nonhub_mrrs = [], []
    for qid, ranked in rankings.items():
        relevant = qrels.get(qid, set())
        is_hub_query = any(doc_freq[d] >= hub_threshold for d in relevant)
        rr = 0.0
        for rank, doc_id in enumerate(ranked[:k], 1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        if is_hub_query:
            hub_mrrs.append(rr)
        else:
            nonhub_mrrs.append(rr)
    return (
        {"mrr": np.mean(hub_mrrs) if hub_mrrs else 0.0, "n": len(hub_mrrs)},
        {"mrr": np.mean(nonhub_mrrs) if nonhub_mrrs else 0.0, "n": len(nonhub_mrrs)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# RQ1 — Proximity learning: embedding distance vs structural distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_act_positions(corpus: dict, dataset: str) -> dict:
    """
    Returns {doc_id: {"act": str, "position": float, "rank_in_act": int}}.
    Position is normalised rank within act [0, 1].
    """
    import re
    act_groups = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "") or ""
        # Simple act extraction: everything before the article number
        if dataset.startswith("kuhperdata"):
            act = "KUHPerdata"
        elif "stard" in dataset:
            m = re.match(r"(.+?)第", title)
            act = m.group(1).strip() if m else "unknown"
        elif "bsard" in dataset:
            m = re.search(r",\s*(.+?)\s*\(", title)
            act = m.group(1).strip() if m else "unknown"
        elif "ilpcsr" in dataset:
            m = re.search(r"\bof\s+(.+)", title, re.IGNORECASE)
            act = m.group(1).strip() if m else "unknown"
        else:
            act = "unknown"
        act_groups.setdefault(act, []).append(doc_id)

    result = {}
    for act, doc_ids in act_groups.items():
        n = len(doc_ids)
        for rank, doc_id in enumerate(doc_ids):
            result[doc_id] = {
                "act": act,
                "position": rank / max(n - 1, 1),
                "rank_in_act": rank,
                "act_size": n,
            }
    return result


def analyze_proximity_learning(
    paragnn_emb: np.ndarray,
    structgnn_emb: np.ndarray,
    corpus_ids: list[str],
    act_positions: dict,
    sample_size: int = 2000,
) -> pd.DataFrame:
    """
    For sampled pairs of articles from the same act, compute:
    - structural distance (|pos_i - pos_j|)
    - Para-GNN cosine distance
    - StructGNN cosine distance
    """
    rng = np.random.RandomState(42)
    valid_ids = [d for d in corpus_ids if d in act_positions]
    idx_map = {d: i for i, d in enumerate(corpus_ids)}

    rows = []
    attempts = 0
    while len(rows) < sample_size and attempts < sample_size * 20:
        i, j = rng.choice(len(valid_ids), size=2, replace=False)
        d_i, d_j = valid_ids[i], valid_ids[j]
        pi, pj = act_positions[d_i], act_positions[d_j]
        if pi["act"] != pj["act"]:
            attempts += 1
            continue

        ei = paragnn_emb[idx_map[d_i]]
        ej = paragnn_emb[idx_map[d_j]]
        para_cos = float(np.dot(ei, ej))  # already L2-normalised

        si = structgnn_emb[idx_map[d_i]]
        sj = structgnn_emb[idx_map[d_j]]
        struct_cos = float(np.dot(si, sj))

        struct_dist = abs(pi["position"] - pj["position"])
        rows.append({
            "structural_dist": struct_dist,
            "paragnn_cosine": para_cos,
            "structgnn_cosine": struct_cos,
            "act": pi["act"],
        })
        attempts += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# RQ1b — Case study: embedding neighbourhood of specific article (e.g. 1365)
# ─────────────────────────────────────────────────────────────────────────────

def find_article_by_number(corpus: dict, article_numbers: list[str]) -> dict:
    """Returns {article_number: doc_id} by searching titles."""
    import re
    result = {}
    for doc_id, doc in corpus.items():
        title = doc.get("title", "") or ""
        for num in article_numbers:
            if num in result:
                continue
            if re.search(rf"\b{num}\b", title):
                result[num] = doc_id
                break
    return result


def compute_pairwise_cosine(emb: np.ndarray, indices: list[int]) -> np.ndarray:
    sub = emb[indices]  # (K, D), already normalised
    return sub @ sub.T  # (K, K)


# ─────────────────────────────────────────────────────────────────────────────
# RQ2 — Hub bias: MRR split bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_hub_nonhub_mrr(
    results: dict,
    hub_threshold: int,
    out_path: Path,
):
    """
    results = {
        "Para-GNN":  {"hub": {"mrr": float, "n": int}, "nonhub": {...}},
        "StructGNN": ...,
        "BM25":      ...,
    }
    """
    rows = []
    for model, split in results.items():
        rows.append({"Model": model, "Split": f"Hub (≥{hub_threshold} queries)", "MRR@10": split["hub"]["mrr"], "n": split["hub"]["n"]})
        rows.append({"Model": model, "Split": "Non-Hub", "MRR@10": split["nonhub"]["mrr"], "n": split["nonhub"]["n"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="Split", y="MRR@10", hue="Model",
                palette=PALETTE, ax=ax, edgecolor="white", linewidth=0.8)

    # Annotate n
    for bar, row in zip(ax.patches, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{row['MRR@10']:.3f}\n(n={row['n']})",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_title("RQ2 — Hub Bias: MRR@10 on Hub vs Non-Hub Queries", fontweight="bold")
    ax.set_ylabel("MRR@10")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path / "rq2_hub_nonhub_mrr.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] rq2_hub_nonhub_mrr.png")


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 — Cosine signal: GT vs random violin plot
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine_signal(
    emb: np.ndarray,
    corpus_ids: list[str],
    qrels: dict,
    doc_freq: Counter,
    hub_threshold: int,
    sample_random: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx_map = {d: i for i, d in enumerate(corpus_ids)}
    rows = []

    query_ids = list(qrels.keys())
    query_indices = [i for i, d in enumerate(corpus_ids)]  # use corpus as query proxy

    for qid in query_ids:
        relevant = [d for d in qrels[qid] if d in idx_map]
        if not relevant:
            continue

        # Use mean of relevant embeddings as query proxy
        q_emb = np.mean([emb[idx_map[d]] for d in relevant], axis=0)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        for d in relevant:
            cos = float(np.dot(q_emb, emb[idx_map[d]]))
            is_hub = doc_freq[d] >= hub_threshold
            rows.append({"cosine": cos, "category": "Hub GT" if is_hub else "Non-Hub GT"})

        non_gt = [d for d in corpus_ids if d not in set(relevant) and d in idx_map]
        sample = rng.choice(non_gt, size=min(sample_random * len(relevant), len(non_gt)), replace=False)
        for d in sample:
            cos = float(np.dot(q_emb, emb[idx_map[d]]))
            rows.append({"cosine": cos, "category": "Random (Non-GT)"})

    return pd.DataFrame(rows)


def plot_cosine_signal(
    para_df: pd.DataFrame,
    struct_df: pd.DataFrame,
    out_path: Path,
):
    para_df["Model"] = "Para-GNN"
    struct_df["Model"] = "StructGNN"
    df = pd.concat([para_df, struct_df], ignore_index=True)

    order = ["Hub GT", "Non-Hub GT", "Random (Non-GT)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, model in zip(axes, ["Para-GNN", "StructGNN"]):
        sub = df[df["Model"] == model]
        color = PALETTE[model]
        sns.violinplot(data=sub, x="category", y="cosine", order=order,
                       ax=ax, color=color, inner="box", cut=0, linewidth=0.8)
        ax.set_title(f"{model}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Cosine Similarity" if model == "Para-GNN" else "")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("RQ3 — Directional Signal: GT Cosine vs Random Cosine", fontweight="bold", y=1.01)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path / "rq3_cosine_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] rq3_cosine_signal.png")


# ─────────────────────────────────────────────────────────────────────────────
# RQ4 — GNN score vs BM25 score correlation (hub vs non-hub)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gnn_bm25_correlation(
    scores: dict,
    qrels: dict,
    doc_freq: Counter,
    hub_threshold: int,
    model_name: str,
    out_path: Path,
):
    gnn_vals, bm25_vals, categories = [], [], []

    for qid, doc_scores in scores.items():
        relevant = qrels.get(qid, set())
        for doc_id, s in doc_scores.items():
            gnn_vals.append(s["gnn"])
            bm25_vals.append(s["bm25"])
            if doc_id in relevant:
                cat = "Hub GT" if doc_freq[doc_id] >= hub_threshold else "Non-Hub GT"
            else:
                cat = "Non-Relevant"
            categories.append(cat)

    df = pd.DataFrame({"GNN Score": gnn_vals, "BM25 Score": bm25_vals, "Category": categories})
    df = df.sample(min(len(df), 8000), random_state=42)  # downsample for speed

    fig, ax = plt.subplots(figsize=(7, 5))
    cat_palette = {"Hub GT": "#e74c3c", "Non-Hub GT": "#2ecc71", "Non-Relevant": "#bdc3c7"}
    sns.scatterplot(data=df, x="BM25 Score", y="GNN Score", hue="Category",
                    palette=cat_palette, alpha=0.4, s=15, ax=ax, linewidth=0)
    ax.set_title(f"RQ4 — {model_name}: GNN vs BM25 Score (Hub/Non-Hub)", fontweight="bold")
    ax.legend(title="", markerscale=2)
    sns.despine()
    fig.tight_layout()
    fname = f"rq4_gnn_bm25_{'paragnn' if 'Para' in model_name else 'structgnn'}.png"
    fig.savefig(out_path / fname, dpi=150)
    plt.close(fig)
    print(f"  [saved] {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# RQ1 — Proximity learning: scatter + binned correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_proximity_learning(prox_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (col, model) in zip(axes, [("paragnn_cosine", "Para-GNN"), ("structgnn_cosine", "StructGNN")]):
        color = PALETTE[model]
        # Bin structural distance into 10 buckets
        prox_df["bin"] = pd.cut(prox_df["structural_dist"], bins=10)
        binned = prox_df.groupby("bin")[col].mean().reset_index()
        binned["mid"] = binned["bin"].apply(lambda x: x.mid)

        sns.scatterplot(
            data=prox_df.sample(min(len(prox_df), 3000), random_state=42),
            x="structural_dist", y=col, ax=ax,
            alpha=0.15, s=10, color=color, linewidth=0,
        )
        ax.plot(binned["mid"], binned[col], color="black", linewidth=2, label="Bin mean")
        ax.set_xlabel("Structural Distance (|pos_i − pos_j|)")
        ax.set_ylabel(f"Cosine Similarity")
        ax.set_title(f"{model}", fontweight="bold")
        ax.legend()

    fig.suptitle(
        "RQ1 — Proximity Learning: Embedding Similarity vs Structural Distance\n"
        "(same-act article pairs; lower structural dist = more adjacent)",
        fontweight="bold", y=1.02,
    )
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path / "rq1_proximity_learning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] rq1_proximity_learning.png")


# ─────────────────────────────────────────────────────────────────────────────
# RQ1b — Case study heatmap: articles 1360–1375 (or any range)
# ─────────────────────────────────────────────────────────────────────────────

def plot_article_neighborhood_heatmap(
    paragnn_emb: np.ndarray,
    structgnn_emb: np.ndarray,
    corpus_ids: list[str],
    corpus: dict,
    article_numbers: list[str],
    out_path: Path,
    label: str = "Article",
):
    idx_map = {d: i for i, d in enumerate(corpus_ids)}

    # Try matching as doc_ids first, then fall back to title search
    found = {}
    for num in article_numbers:
        if num in idx_map:
            found[num] = num
        else:
            match = find_article_by_number(corpus, [num])
            if num in match:
                found[num] = match[num]

    if len(found) < 2:
        print(f"  [skip] rq1b heatmap — found only {len(found)} articles")
        return

    titles = {did: corpus.get(did, {}).get("title", did) for did in found.values()}
    labels = [titles[found[num]] for num in found]
    indices = [idx_map[found[num]] for num in found]

    para_sim = compute_pairwise_cosine(paragnn_emb, indices)
    struct_sim = compute_pairwise_cosine(structgnn_emb, indices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sim, title in zip(axes, [para_sim, struct_sim], ["Para-GNN", "StructGNN"]):
        sns.heatmap(
            pd.DataFrame(sim, index=labels, columns=labels),
            ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0.0, vmax=1.0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"{title}", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "RQ1b — Embedding Neighbourhood: Pairwise Cosine Similarity\n"
        "Structurally adjacent articles should be more similar in StructGNN",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path / "rq1b_article_neighborhood_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] rq1b_article_neighborhood_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Hub frequency bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_hub_frequency_in_rankings(
    rankings_para: dict,
    rankings_struct: dict,
    doc_freq: Counter,
    hub_threshold: int,
    corpus: dict,
    out_path: Path,
    top_hubs: int = 15,
):
    hub_docs = {d for d, f in doc_freq.items() if f >= hub_threshold}
    n_queries = len(rankings_para)

    def count_appearances(rankings):
        freq = Counter()
        for ranked in rankings.values():
            for doc_id in ranked[:10]:
                if doc_id in hub_docs:
                    freq[doc_id] += 1
        return freq

    para_freq = count_appearances(rankings_para)
    struct_freq = count_appearances(rankings_struct)

    all_hubs = set(list(para_freq.keys()) + list(struct_freq.keys()))
    if not all_hubs:
        print("  [skip] hub frequency chart — no hub articles found in top-10")
        return

    rows = []
    for doc_id in all_hubs:
        title = corpus.get(doc_id, {}).get("title", doc_id)
        short = title[:30] + "…" if len(title) > 30 else title
        rows.append({"doc": short, "Model": "Para-GNN", "appearances": para_freq[doc_id] / n_queries})
        rows.append({"doc": short, "Model": "StructGNN", "appearances": struct_freq[doc_id] / n_queries})

    df = pd.DataFrame(rows)
    top_docs = df.groupby("doc")["appearances"].max().nlargest(top_hubs).index
    df = df[df["doc"].isin(top_docs)]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df, y="doc", x="appearances", hue="Model",
                palette={"Para-GNN": PALETTE["Para-GNN"], "StructGNN": PALETTE["StructGNN"]},
                ax=ax, orient="h", edgecolor="white")
    ax.set_xlabel("Fraction of Queries Where Article Appears in Top-10")
    ax.set_ylabel("")
    ax.set_title("RQ2b — Hub Article Frequency in Top-10 Predictions", fontweight="bold")
    ax.axvline(doc_freq[next(iter(hub_docs))] / n_queries if hub_docs else 0.5,
               color="gray", linestyle="--", alpha=0.4, label="GT frequency baseline")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path / "rq2b_hub_frequency.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] rq2b_hub_frequency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Training alpha curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_curve(training_log_path: str, out_path: Path):
    if not Path(training_log_path).exists():
        print(f"  [skip] alpha curve — {training_log_path} not found")
        return

    with open(training_log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    alphas = [e.get("learned_alpha", None) for e in log]
    val_mrr = [e.get("val_mrr@10", None) for e in log]

    if all(a is None for a in alphas):
        print("  [skip] alpha curve — no learned_alpha in training log")
        return

    fig, ax1 = plt.subplots(figsize=(9, 4))
    color1 = "#4C72B0"
    ax1.plot(epochs, alphas, color=color1, linewidth=2, label="Mean alpha")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Learned Alpha (GNN weight)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    if any(v is not None for v in val_mrr):
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_mrr, color="#DD8452", linewidth=2, linestyle="--", label="Val MRR@10")
        ax2.set_ylabel("Val MRR@10", color="#DD8452")
        ax2.tick_params(axis="y", labelcolor="#DD8452")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("RQ4 — Learned Alpha Over Training\n(alpha=1: trust GNN fully; alpha=0: trust BM25 fully)", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path / "rq4_alpha_curve.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] rq4_alpha_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNN structure explainability analysis")
    parser.add_argument("--dataset", default="kuhperdata-humanized")
    parser.add_argument("--paragnn_dir", default=None,
                        help="Dir with Para-GNN inference output JSON (none_adapted.json)")
    parser.add_argument("--structgnn_dir", default=None,
                        help="Dir with StructGNN inference output JSON (structural_adapted.json)")
    parser.add_argument("--paragnn_emb", default=None,
                        help="Para-GNN corpus embeddings .npy (N_docs × D)")
    parser.add_argument("--structgnn_emb", default=None,
                        help="StructGNN corpus embeddings .npy (N_docs × D)")
    parser.add_argument("--paragnn_training_log", default=None,
                        help="Para-GNN training_log.json path")
    parser.add_argument("--structgnn_training_log", default=None,
                        help="StructGNN training_log.json path")
    parser.add_argument("--data_dir", default=None,
                        help="Dataset dir (default: data/<dataset>)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save figures (default: outputs/analysis/gnn_explainability/<dataset>)")
    parser.add_argument("--hub_threshold", type=int, default=HUB_THRESHOLD_DEFAULT,
                        help="Min query count to classify a doc as hub")
    parser.add_argument("--article_numbers", nargs="+", default=None,
                        help="Article numbers for case study heatmap (auto-selected if omitted)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    # ── Resolve paths ──
    data_dir = args.data_dir or f"data/{args.dataset}"
    out_path = Path(args.output_dir or f"outputs/analysis/gnn_explainability/{args.dataset}")
    out_path.mkdir(parents=True, exist_ok=True)

    corpus_path = f"{data_dir}/corpus.jsonl"
    qrels_path = f"{data_dir}/qrels_{args.split}.tsv"

    print(f"\n{'='*60}")
    print(f"  GNN Explainability: {args.dataset}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    # ── Load base data ──
    print("Loading corpus and qrels...")
    corpus = load_corpus(corpus_path)
    qrels = load_qrels(qrels_path)
    doc_freq = compute_doc_freq(qrels)

    hub_docs = {d for d, f in doc_freq.items() if f >= args.hub_threshold}
    print(f"  Corpus: {len(corpus)} docs | Queries: {len(qrels)} | Hub docs (≥{args.hub_threshold}): {len(hub_docs)}")
    print(f"  Top 10 hub articles: {doc_freq.most_common(10)}\n")

    # ── Load inference results ──
    para_data = struct_data = None
    para_rankings = struct_rankings = {}
    para_scores = struct_scores = {}

    if args.paragnn_dir:
        json_files = list(Path(args.paragnn_dir).glob("*.json"))
        if json_files:
            print(f"Loading Para-GNN results: {json_files[0]}")
            para_data = load_inference_json(str(json_files[0]))
            para_rankings = get_rankings_from_json(para_data)
            para_scores = get_scores_from_json(para_data)

    if args.structgnn_dir:
        json_files = list(Path(args.structgnn_dir).glob("*.json"))
        if json_files:
            print(f"Loading StructGNN results: {json_files[0]}")
            struct_data = load_inference_json(str(json_files[0]))
            struct_rankings = get_rankings_from_json(struct_data)
            struct_scores = get_scores_from_json(struct_data)

    # ── Load embeddings ──
    para_emb = struct_emb = None
    corpus_ids = list(corpus.keys())

    if args.paragnn_emb and Path(args.paragnn_emb).exists():
        print(f"Loading Para-GNN embeddings: {args.paragnn_emb}")
        para_emb = np.load(args.paragnn_emb).astype(np.float32)
        norms = np.linalg.norm(para_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        para_emb = para_emb / norms
        print(f"  Shape: {para_emb.shape}")

    if args.structgnn_emb and Path(args.structgnn_emb).exists():
        print(f"Loading StructGNN embeddings: {args.structgnn_emb}")
        struct_emb = np.load(args.structgnn_emb).astype(np.float32)
        norms = np.linalg.norm(struct_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        struct_emb = struct_emb / norms
        print(f"  Shape: {struct_emb.shape}")

    print()

    # ── RQ2: Hub vs Non-Hub MRR ──
    if para_rankings or struct_rankings:
        print("Running RQ2: Hub vs Non-Hub MRR...")
        mrr_results = {}
        for model, rankings in [("Para-GNN", para_rankings), ("StructGNN", struct_rankings)]:
            if rankings:
                h, nh = mrr_split_hub_nonhub(rankings, qrels, doc_freq, args.hub_threshold)
                mrr_results[model] = {"hub": h, "nonhub": nh}
                print(f"  {model}: hub MRR={h['mrr']:.4f} (n={h['n']}), non-hub MRR={nh['mrr']:.4f} (n={nh['n']})")
        if mrr_results:
            plot_hub_nonhub_mrr(mrr_results, args.hub_threshold, out_path)

        # RQ2b: hub frequency in top-10
        if para_rankings and struct_rankings:
            print("Running RQ2b: Hub article frequency in top-10...")
            plot_hub_frequency_in_rankings(
                para_rankings, struct_rankings, doc_freq, args.hub_threshold, corpus, out_path
            )

    # ── RQ4: GNN vs BM25 scatter ──
    if para_scores:
        print("Running RQ4: Para-GNN score correlation...")
        plot_gnn_bm25_correlation(para_scores, qrels, doc_freq, args.hub_threshold, "Para-GNN", out_path)
    if struct_scores:
        print("Running RQ4: StructGNN score correlation...")
        plot_gnn_bm25_correlation(struct_scores, qrels, doc_freq, args.hub_threshold, "StructGNN", out_path)

    # ── RQ4: Alpha curve ──
    if args.paragnn_training_log:
        print("Running RQ4: Para-GNN alpha curve...")
        plot_alpha_curve(args.paragnn_training_log, out_path)
    if args.structgnn_training_log:
        print("Running RQ4: StructGNN alpha curve...")
        plot_alpha_curve(args.structgnn_training_log, out_path)

    # ── Embedding-based analyses (require .npy) ──
    if para_emb is not None and struct_emb is not None:
        # Check shape alignment
        if len(para_emb) != len(corpus_ids) or len(struct_emb) != len(corpus_ids):
            print(f"  [warn] Embedding size mismatch: para={len(para_emb)}, struct={len(struct_emb)}, corpus={len(corpus_ids)}")
            print("         Trying to align by order...")

        # RQ3: Cosine signal
        print("Running RQ3: Cosine signal (GT vs random)...")
        para_cos_df = compute_cosine_signal(para_emb, corpus_ids, qrels, doc_freq, args.hub_threshold)
        struct_cos_df = compute_cosine_signal(struct_emb, corpus_ids, qrels, doc_freq, args.hub_threshold)
        plot_cosine_signal(para_cos_df, struct_cos_df, out_path)

        # RQ1: Proximity learning
        print("Running RQ1: Proximity learning...")
        act_positions = compute_act_positions(corpus, args.dataset)
        prox_df = analyze_proximity_learning(para_emb, struct_emb, corpus_ids, act_positions)
        if not prox_df.empty:
            para_corr = prox_df["structural_dist"].corr(prox_df["paragnn_cosine"])
            struct_corr = prox_df["structural_dist"].corr(prox_df["structgnn_cosine"])
            print(f"  Structural dist vs cosine correlation:")
            print(f"    Para-GNN:  r={para_corr:.4f} (expected ≈0)")
            print(f"    StructGNN: r={struct_corr:.4f} (expected <0 if proximity learned)")
            plot_proximity_learning(prox_df, out_path)

        # RQ1b: Article neighbourhood heatmap
        article_numbers = args.article_numbers
        if article_numbers is None:
            # Auto-select: pick a consecutive range from the most-relevant hub article's neighborhood
            top_doc = doc_freq.most_common(1)[0][0] if doc_freq else corpus_ids[len(corpus_ids) // 2]
            top_idx = corpus_ids.index(top_doc)
            start = max(0, top_idx - 4)
            end = min(len(corpus_ids), start + 9)
            article_numbers = corpus_ids[start:end]
        print(f"Running RQ1b: Article neighbourhood heatmap {article_numbers}...")
        plot_article_neighborhood_heatmap(
            para_emb, struct_emb, corpus_ids, corpus,
            article_numbers, out_path,
            label="Article",
        )
    else:
        print("\n  [info] Skipping embedding analyses — provide --paragnn_emb and --structgnn_emb")
        print("         Run inference with --export_embeddings to generate .npy files")

    print(f"\n✓ All analyses complete. Figures saved to: {out_path}\n")


if __name__ == "__main__":
    main()
