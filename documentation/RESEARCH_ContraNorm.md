# RESEARCH: ContraNorm — A Contrastive Learning Perspective on Oversmoothing and Beyond

**Paper:** Guo, X., Wang, Y., Du, T., & Wang, Y. (ICLR 2023)  
**Repo:** https://github.com/PKU-ML/ContraNorm  
**Local clone:** `ContraNorm/`

## Problem It Solves

Dimensional collapse in GNNs/Transformers — representations lie in a narrow cone rather than using the full embedding space. Standard oversmoothing metrics (average cosine similarity) miss this; effective rank and singular value decay capture it.

**Our measured collapse (Analysis 03E, kuhperdata-exp):**
- StructGNN: avg pairwise cosine = 0.885, effective rank @90% = 64/1024
- BGE-M3: avg pairwise cosine = 0.563, effective rank @90% = 126/1024

## Core Idea

Transfer the uniformity loss from contrastive learning into a normalization layer. Instead of adding uniformity loss to training objective (which they show has limited effect), derive a **drop-in layer** from one gradient step of the uniformity loss on representations.

## The ContraNorm Operation

Given representation matrix `H` (N nodes × D dims):

```
H_out = (1 + s) * H - s * softmax(H @ H.T / τ) @ H
```

Where:
- `s` = scale factor (hyperparameter, typically 0.2-1.0 for GNNs)
- `τ` = temperature (controls sharpness of similarity weighting)
- `softmax(H @ H.T / τ)` = pairwise attention matrix

**Intuition:** For each node, subtract a weighted average of similar nodes. This pushes apart nodes that are too similar, spreading representations more uniformly.

### With adjacency masking (GNN variant):

```python
norm_x = F.normalize(x, dim=1)
sim = norm_x @ norm_x.T / tau
sim.masked_fill_(adj > 0, -inf)       # mask connected neighbors
sim = F.softmax(sim, dim=1)
x_neg = sim @ x
x_out = (1 + s) * x - s * x_neg
```

The masking prevents pushing apart nodes that *should* be similar (connected neighbors). Only non-neighbor similar nodes get repelled.

## Implementation in Repo (`ContraNorm/gnn/layers.py:166-175`)

```python
if self.mode == 'CN':
    norm_x = nn.functional.normalize(x, dim=1)
    sim = norm_x @ norm_x.T / tau
    if adj.size(1) == 2:
        sim[adj[0], adj[1]] = -np.inf
    else:
        sim.masked_fill_(adj.to_dense() > 1e-5, -np.inf)
    sim = nn.functional.softmax(sim, dim=1)
    x_neg = sim @ x    
    x = (1 + self.scale) * x - self.scale * x_neg   
```

## Placement in GNN Architecture

From their experiments (DeepGCN model, `models.py:148-175`):
- Applied **after each graph convolution layer, before activation**
- Pattern: `x = conv(x, adj)` → `x = contranorm(x)` → `x = relu(x)`

For their GAT model (`models.py:87-94`):
- Applied after attention layer, before ELU activation

## How to Apply to Our EUGAT (`src/paragnn/eugat.py:195-210`)

Current flow:
```
Layer 1: EUGATConv1 → dropout → relu + residual
Layer 2: EUGATConv2 → relu → residual from input
```

Insert ContraNorm after Layer 1 residual (before Layer 2):
```
Layer 1: EUGATConv1 → dropout → relu + residual → ContraNorm
Layer 2: EUGATConv2 → relu → residual from input
```

This matches their placement: after attention aggregation + residual, before next layer.

## Hyperparameters

| Param | Transformers (paper) | GNNs (paper) | Our Ablation Grid |
|-------|---------------------|--------------|-------------------|
| `s` (scale) | {0.005, 0.01, 0.05, 0.1, 0.2} | {0.2, 0.5, 0.8, 1.0} | {0.0, 0.1, 0.2, 0.5, 1.0} |
| `τ` (temperature) | 1.0 | 1.0 | 1.0 (fixed) |
| Adjacency masking | N/A | Yes | **No** (see below) |

**Key difference**: Transformers use much smaller scale (0.005-0.2) vs GNNs (0.2-1.0).
Our model is a 2-layer GNN, so start from the GNN range.

### Why we skip adjacency masking

Our graph structure is fundamentally different from citation/social networks:
- In ContraNorm's experiments, edges mean "these nodes ARE related" → mask them to avoid pushing apart related nodes
- In our EUGAT graph, edges connect query-to-candidates and candidates within a BM25 pool — they are **retrieval candidates, not semantic neighbors**
- Co-relevant articles that should stay close are NOT necessarily connected by edges
- Therefore: apply ContraNorm **without** adjacency masking — let it spread ALL similar nodes, then the training loss will pull co-relevant ones back together

## Theoretical Guarantees (from paper)

**Proposition 1:** Under the regularized update `H_t = ((1+s)I - sĀ)H_b`, if the symmetric matrix `P = (I-ee^T)(I-Ā) + (I-Ā)^T(I-ee^T)` is positive semi-definite, then `Var(H_t) ≥ Var(H_b)`.

**Proposition 2:** For update `H_t = (1+s)H_b - s(H_b H_b^T)H_b`, if `s` satisfies `1 + (1-σ²_max)s > 0`, then `erank(H_t) > erank(H_b)`.

Translation: ContraNorm provably increases effective rank when scale `s` is small enough relative to the largest singular value.

## Complexity

`O(N² × D)` — same as self-attention. For our graphs:
- Training batches: ~300 nodes per graph → negligible overhead
- Inference: full corpus graph (~2000 nodes for kuhperdata) → still fast

## Evaluation Plan

After retraining with ContraNorm:
1. Re-run Analysis E (collapse check): expect effective rank to increase, avg cosine to decrease
2. Re-run Analysis D (separation): expect Cohen's d and AUC to improve
3. Re-run Analysis C (neighborhood coverage): should maintain or improve 1.7x advantage
4. Compare Recall@10 / MRR@10 on test set against baseline StructGNN

## Key Difference from Post-hoc Whitening

- Whitening is applied after training → model never learns to use the full space
- ContraNorm is applied during training → model learns representations that are inherently more isotropic
- ContraNorm maintains discriminative power (training loss still pulls relevant pairs together)
- Whitening can destroy learned structure; ContraNorm preserves it while expanding the space

## References

- Guo, X., Wang, Y., Du, T., & Wang, Y. (2023). ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond. ICLR 2023.
- Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2022). Understanding Dimensional Collapse in Contrastive Self-supervised Learning. ICLR 2022.
- Roth, A. & Liebig, T. (2023). Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks. LoG 2023.
