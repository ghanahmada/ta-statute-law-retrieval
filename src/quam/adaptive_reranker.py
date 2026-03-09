"""
QUAM: Query Affinity Modelling — SetAff adaptive re-ranking algorithm.

Implements the SetAff frontier update from Rathee et al. (WSDM 2025).
Key difference from GAR: frontier scores are accumulated using edge weights
and softmax-normalized parent relevance scores, rather than simple max replacement.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Callable, Optional

from gar.corpus_graph import CorpusGraph


class QUAM:
    """
    QUAM with SetAff frontier update.

    Args:
        corpus_graph: kNN corpus graph for neighbor lookup (with edge weights)
        budget: maximum number of documents to score (re-ranking budget)
        batch_size: documents scored per iteration
        top_s: size of set S (top scored docs used for frontier expansion)
        backfill: whether to include unscored initial results in output
    """

    def __init__(
        self,
        corpus_graph: CorpusGraph,
        budget: int = 100,
        batch_size: int = 16,
        top_s: int = 30,
        backfill: bool = True,
    ):
        self.corpus_graph = corpus_graph
        self.budget = budget
        self.batch_size = batch_size
        self.top_s = top_s
        self.backfill = backfill

    def rerank(
        self,
        initial_results: List[Tuple[str, float]],
        scorer: Callable[[List[str]], Dict[str, float]],
        graph_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Adaptively re-rank using SetAff frontier expansion.

        Args:
            initial_results: list of (doc_id, score) from first-stage retriever,
                             sorted by score descending
            scorer: function that takes a list of doc_ids and returns {doc_id: score}
            graph_k: limit neighbors per lookup (default: use full graph k)

        Returns:
            list of (doc_id, score) sorted by re-ranker score descending
        """
        scored: Dict[str, float] = {}

        # Two sources: pool (initial results) and frontier (graph neighbors)
        pool = Counter({doc_id: score for doc_id, score in initial_results})
        frontier = Counter()

        # R1: all re-ranked documents with their scores (for building set S)
        all_reranked: Dict[str, float] = {}

        iteration = 0

        while len(scored) < self.budget:
            remaining = self.budget - len(scored)
            size = min(self.batch_size, remaining)

            # Alternate: even -> pool, odd -> frontier
            if iteration % 2 == 0:
                primary, secondary = pool, frontier
            else:
                primary, secondary = frontier, pool

            # Pick top-scoring unscored docs from primary source
            batch = self._pick_batch(primary, scored, size)

            # Fall back to secondary source if primary is exhausted
            if not batch:
                batch = self._pick_batch(secondary, scored, size)

            if not batch:
                break

            # Score the batch with the expensive re-ranker
            new_scores = scorer(batch)
            scored.update(new_scores)
            all_reranked.update(new_scores)

            # Build set S: top-s documents from all re-ranked so far
            s_docs = self._build_set_s(all_reranked)

            # SetAff frontier expansion: only newly scored docs that are in S
            new_in_s = [did for did in batch if did in s_docs]
            if new_in_s:
                self._update_frontier_setaff(
                    new_in_s, s_docs, frontier, scored, graph_k
                )

            # Remove scored docs from both sources
            for doc_id in batch:
                pool.pop(doc_id, None)
                frontier.pop(doc_id, None)

            iteration += 1

        # Backfill: add unscored pool docs with decreasing scores
        if self.backfill and len(scored) < self.budget:
            min_score = min(scored.values()) if scored else 0.0
            for i, (doc_id, _) in enumerate(pool.most_common()):
                if doc_id not in scored:
                    scored[doc_id] = min_score - 1 - i
                    if len(scored) >= self.budget:
                        break

        return sorted(scored.items(), key=lambda x: -x[1])

    def _build_set_s(self, all_reranked: Dict[str, float]) -> Dict[str, float]:
        """Build set S: top-s documents by re-ranker score, with softmax-normalized scores."""
        # Sort by score descending, take top-s
        sorted_docs = sorted(all_reranked.items(), key=lambda x: -x[1])[:self.top_s]
        if not sorted_docs:
            return {}

        doc_ids, scores = zip(*sorted_docs)
        scores_arr = np.array(scores, dtype=np.float64)

        # Softmax normalization
        scores_arr = scores_arr - scores_arr.max()  # numerical stability
        exp_scores = np.exp(scores_arr)
        softmax_scores = exp_scores / exp_scores.sum()

        return dict(zip(doc_ids, softmax_scores.tolist()))

    def _update_frontier_setaff(
        self,
        new_docs_in_s: List[str],
        s_docs: Dict[str, float],
        frontier: Counter,
        scored: Dict[str, float],
        graph_k: Optional[int],
    ):
        """
        SetAff frontier update: accumulate edge_weight * softmax(parent_score).

        For each newly scored doc in S, look up its graph neighbors.
        Each neighbor's frontier score is incremented by:
            aff_score(parent, neighbor) * R(parent)
        where R(parent) is the softmax-normalized score within set S.
        """
        for doc_id in new_docs_in_s:
            s_score = s_docs[doc_id]  # softmax-normalized relevance
            for neighbor_id, aff_score in self.corpus_graph.neighbors(doc_id, graph_k):
                if neighbor_id not in scored:
                    frontier[neighbor_id] += aff_score * s_score

    def _pick_batch(
        self, source: Counter, scored: Dict[str, float], size: int
    ) -> List[str]:
        """Pick top-scoring unscored documents from a source."""
        batch = []
        for doc_id, _ in source.most_common():
            if doc_id not in scored:
                batch.append(doc_id)
                if len(batch) >= size:
                    break
        return batch
