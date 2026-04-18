# Research: Late Interaction & MoE for Retrieval Fusion

Date: 2026-04-18

---

## Late Interaction (ColBERT-style)

- MaxSim: per-token similarity between query and document tokens, max per query token, sum across query tokens. Preserves token-level granularity that dense encoders lose.
- Storage cost for 55K statute corpus with BGE-M3 (1024d): ~2.8GB for per-token embeddings.
- Most practical integration: additional scoring signal alongside BM25 and GNN, not replacing the GNN. The GNN handles structural reasoning that ColBERT cannot.
- QDER uses **late fusion** (score-level combination), not late interaction (representation-level). Different concept.

### Key Papers
- ColBERT (Khattab & Zaharia, 2020): Original late interaction
- ColBERTv2 (Santhanam et al., 2022): Residual compression, denoised supervision
- ColPali (2024): Late interaction for visual document retrieval
- QDER: Late fusion, not late interaction

---

## MoE for Retrieval Fusion

- MoDE (SIGIR 2024): Per-query adaptive fusion outperforms fixed alpha blending by 5-10%.
- Proposed architecture: lightweight 3-expert gating network
  - Expert 1: BM25 score
  - Expert 2: GNN score
  - Expert 3: Structural signal
  - Gate input: query embedding → softmax → mixing weights
- Replaces current alpha grid search with learned, query-conditioned alternative (different queries get different alpha values).
- Parameter cost: ~3K for a simple gate network (negligible).
- Risk: alpha grid search already works well and is interpretable. MoE adds training complexity for potentially modest gains.

### Key Papers
- MoDE (SIGIR 2024): Mixture of dense experts for retrieval
- Switch Transformer (Fedus et al., 2022): Sparse MoE scaling
- Mixture of Experts in retrieval literature

---

## Recommendation

MoE-style adaptive fusion is more practical:
1. Directly replaces alpha grid search with something learnable
2. Minimal parameter overhead (~3K)
3. Strengthens paper contribution (query-adaptive fusion)
4. Late interaction is a bigger architectural change with less clear payoff given GNN already does token-level reasoning through graph attention

## Status

Pending: waiting for BSARD, STARD, IL-PCSR StructGNN results before deciding on method extensions.
