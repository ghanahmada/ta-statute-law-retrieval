"""
QUAM: Query Affinity Modelling for Adaptive Retrieval

Adapted from: "Quam: Adaptive Retrieval through Query Affinity Modelling"
(Rathee, MacAvaney & Anand, WSDM 2025)

Builds on GAR by improving the frontier document selection:
  - GAR: frontier[neighbor] = max(parent_score)          — ignores edge weights
  - QUAM: frontier[neighbor] += aff_score * R(parent)    — weighted accumulation

Key innovation (SetAff):
  1. Maintain set S = top-s scored documents by re-ranker score
  2. Softmax-normalize scores within S
  3. For each newly scored doc in S, expand its graph neighbors
  4. Frontier score = sum of (edge_weight * softmax_parent_score) across all parents
"""
