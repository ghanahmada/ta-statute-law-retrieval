"""
GAR: Graph-based Adaptive Re-ranking algorithm.

Implements Algorithm 1 from MacAvaney et al. (CIKM 2022).
Alternates between scoring initial pool documents and corpus graph neighbors,
iteratively expanding the candidate pool to overcome recall limitations.
"""

from collections import Counter
from typing import Dict, List, Tuple, Callable, Optional

from gar.corpus_graph import CorpusGraph


class GAR:
    """
    Graph-based Adaptive Re-ranking.

    Args:
        corpus_graph: kNN corpus graph for neighbor lookup
        budget: maximum number of documents to score (re-ranking budget)
        batch_size: documents scored per iteration
        backfill: whether to include unscored initial results in output
    """

    def __init__(
        self,
        corpus_graph: CorpusGraph,
        budget: int = 100,
        batch_size: int = 10,
        backfill: bool = True,
    ):
        self.corpus_graph = corpus_graph
        self.budget = budget
        self.batch_size = batch_size
        self.backfill = backfill

    def rerank(
        self,
        initial_results: List[Tuple[str, float]],
        scorer: Callable[[List[str]], Dict[str, float]],
        graph_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Adaptively re-rank using the corpus graph.

        Args:
            initial_results: list of (doc_id, score) from first-stage retriever,
                             sorted by score descending
            scorer: function that takes a list of doc_ids and returns {doc_id: score}
                    This is the expensive re-ranker (e.g. dense cosine, cross-encoder)
            graph_k: limit neighbors per lookup (default: use full graph k)

        Returns:
            list of (doc_id, score) sorted by re-ranker score descending
        """
        scored: Dict[str, float] = {}

        # Two sources: pool (initial results) and frontier (graph neighbors)
        pool = Counter({doc_id: score for doc_id, score in initial_results})
        frontier = Counter()

        iteration = 0

        while len(scored) < self.budget:
            remaining = self.budget - len(scored)
            size = min(self.batch_size, remaining)

            # Alternate: even → pool, odd → frontier
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

            # Expand frontier: for each newly scored doc, add its graph neighbors
            for doc_id in batch:
                doc_score = new_scores.get(doc_id, 0.0)
                for neighbor_id, _ in self.corpus_graph.neighbors(doc_id, graph_k):
                    if neighbor_id not in scored:
                        # Frontier score = max score of any doc that pointed to it
                        frontier[neighbor_id] = max(
                            frontier.get(neighbor_id, 0.0), doc_score
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
