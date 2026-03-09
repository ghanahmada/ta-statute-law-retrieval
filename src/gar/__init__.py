"""
GAR: Graph-based Adaptive Re-ranking

Adapted from: "Adaptive Re-Ranking with a Corpus Graph" (MacAvaney et al., CIKM 2022)

The key idea: standard re-ranking has a recall ceiling — if a relevant document
isn't in the initial pool, the re-ranker can never find it. GAR overcomes this
by iteratively expanding the candidate pool using a corpus graph (kNN neighbors).

Algorithm:
  1. Build corpus graph offline (each doc → top-K nearest neighbors by embedding similarity)
  2. Start with initial pool from first-stage retriever (e.g. BM25 top-N)
  3. Score a batch of candidates with a re-ranker
  4. For highest-scored docs, fetch their graph neighbors → add to pool
  5. Alternate between scoring pool docs and graph neighbors until budget exhausted
"""
