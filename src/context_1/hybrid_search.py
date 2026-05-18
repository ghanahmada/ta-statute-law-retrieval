"""Hybrid BM25 + dense retrieval with RRF fusion and optional reranking.

Mirrors Context-1's search_corpus tool backend: parallel BM25 and dense
queries fused via reciprocal rank fusion, then reranked with a cross-encoder.
"""

import functools
import os

import numpy as np
from tqdm import tqdm as _tqdm


def _silent_tqdm(*args, **kwargs):
    kwargs["disable"] = True
    return _tqdm(*args, **kwargs)


class HybridSearcher:

    def __init__(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        corpus_embeddings: np.ndarray,
        bm25,
        query_encoder,
        reranker=None,
        rrf_k: int = 60,
    ):
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.corpus_embeddings = corpus_embeddings
        self.bm25 = bm25
        self.query_encoder = query_encoder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self._id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    def _encode_query(self, query: str) -> np.ndarray:
        output = self.query_encoder.encode(
            [query], batch_size=1, max_length=1024,
        )
        if isinstance(output, dict):
            vec = np.array(output["dense_vecs"])
        else:
            vec = np.array(output)
        return vec[0]

    def encode_query(self, query: str) -> np.ndarray:
        vec = self._encode_query(query)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4, "encode_query must return unit-norm vector"
        return vec

    def search(
        self,
        query: str,
        top_n: int = 10,
        exclude_ids: set[str] | None = None,
        rerank: bool = True,
        rerank_top_k: int = 100,
    ) -> list[tuple[str, str, float]]:
        """Hybrid search returning (doc_id, doc_text, score) tuples."""
        exclude_ids = exclude_ids or set()
        n_docs = len(self.doc_ids)

        bm25_scores = self.bm25.transform(query)
        if hasattr(bm25_scores, "toarray"):
            bm25_scores = bm25_scores.toarray().flatten()
        else:
            bm25_scores = np.array(bm25_scores).flatten()

        query_emb = self._encode_query(query)
        dense_scores = query_emb @ self.corpus_embeddings.T

        bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
        dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
        rrf_scores = (
            1.0 / (self.rrf_k + bm25_ranks)
            + 1.0 / (self.rrf_k + dense_ranks)
        )

        ranked_indices = np.argsort(-rrf_scores)

        candidates = []
        for idx in ranked_indices:
            did = self.doc_ids[idx]
            if did in exclude_ids:
                continue
            candidates.append((idx, did, rrf_scores[idx]))
            if len(candidates) >= rerank_top_k:
                break

        if not candidates:
            return []

        if rerank and self.reranker is not None:
            cand_queries = [query] * len(candidates)
            cand_docs = [self.doc_texts[idx] for idx, _, _ in candidates]
            rerank_scores = self.reranker.score_pairs(
                cand_queries, cand_docs, batch_size=32,
            )
            scored = [
                (candidates[i][1], self.doc_texts[candidates[i][0]], float(s))
                for i, s in enumerate(rerank_scores)
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            return scored[:top_n]

        return [
            (did, self.doc_texts[idx], float(score))
            for idx, did, score in candidates[:top_n]
        ]
