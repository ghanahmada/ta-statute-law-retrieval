"""
Corpus Graph: kNN document similarity graph for GAR.

Each document is a node. Edges connect each document to its K nearest neighbors.
Following the original GAR paper (MacAvaney et al., CIKM 2022), the graph is
built using BM25 similarity: each document is treated as a "query" against the
rest of the corpus, and the top-K BM25 results become its neighbors.

Storage format:
  - edges.npy: (n_docs, k) int32 — neighbor indices
  - weights.npy: (n_docs, k) float32 — BM25 similarity weights
  - doc_ids.json: ordered list mapping index → doc_id string
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union


class CorpusGraph:
    """kNN corpus graph for adaptive re-ranking."""

    def __init__(self, edges: np.ndarray, weights: np.ndarray, doc_ids: List[str]):
        self.edges = edges
        self.weights = weights
        self.doc_ids = doc_ids
        self.k = edges.shape[1]
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    def neighbors(self, doc_id: str, limit_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return neighbors of a document as (doc_id, weight) tuples."""
        idx = self._id_to_idx.get(doc_id)
        if idx is None:
            return []
        k = limit_k or self.k
        return [
            (self.doc_ids[int(self.edges[idx, j])], float(self.weights[idx, j]))
            for j in range(min(k, self.k))
            if self.edges[idx, j] != idx  # exclude self-loops
        ]

    def neighbor_ids(self, doc_id: str, limit_k: Optional[int] = None) -> List[str]:
        """Return just the neighbor doc_ids."""
        return [did for did, _ in self.neighbors(doc_id, limit_k)]

    def save(self, path: Union[str, Path]):
        """Save graph to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "edges.npy", self.edges)
        np.save(path / "weights.npy", self.weights)
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"n_docs": len(self.doc_ids), "k": self.k}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CorpusGraph":
        """Load graph from directory."""
        path = Path(path)
        edges = np.load(path / "edges.npy")
        weights = np.load(path / "weights.npy")
        with open(path / "doc_ids.json") as f:
            doc_ids = json.load(f)
        return cls(edges, weights, doc_ids)

    @classmethod
    def build_from_bm25(
        cls,
        doc_ids: List[str],
        doc_texts: List[str],
        k: int = 16,
        b: float = 0.75,
        k1: float = 1.5,
        n_gram: int = 1,
        lang: str = "en",
        verbose: bool = True,
    ) -> "CorpusGraph":
        """
        Build kNN graph using BM25 similarity (faithful to GAR paper).

        Each document is used as a "query" against the entire corpus.
        The top-K BM25 results (excluding itself) become its neighbors.

        Args:
            doc_ids: ordered doc_id strings
            doc_texts: corresponding document texts
            k: number of nearest neighbors per document
            b, k1, n_gram: BM25 parameters
            lang: language code ("zh" triggers jieba tokenization)
            verbose: print progress
        """
        from util.bm25 import BM25

        n_docs = len(doc_ids)
        edges = np.zeros((n_docs, k), dtype=np.int32)
        weights = np.zeros((n_docs, k), dtype=np.float32)

        texts = doc_texts
        if lang == "zh":
            import jieba
            jieba.setLogLevel(20)
            texts = [" ".join(jieba.cut(t)) for t in doc_texts]

        if verbose:
            print(f"  Building BM25 corpus graph: {n_docs} docs, k={k}")

        bm25 = BM25(b=b, k1=k1, n_gram=n_gram)
        bm25.fit(texts)

        for i in range(n_docs):
            scores = bm25.transform(texts[i])
            scores[i] = -1  # exclude self
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            edges[i] = top_k_idx
            weights[i] = scores[top_k_idx]

            if verbose and ((i + 1) % 500 == 0 or i + 1 == n_docs):
                print(f"    {i + 1}/{n_docs} docs processed")

        return cls(edges, weights, doc_ids)
