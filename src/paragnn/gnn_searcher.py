"""StructGNN-based searcher for agentic retrieval.

Replaces HybridSearcher's RRF(BM25, dense) with StructGNN's native
alpha * gnn_scores + (1-alpha) * bm25_scores scoring. Corpus embeddings
are pre-computed via EUGAT; queries are encoded on-the-fly through a
tiny per-query graph.

Call preindex_queries() after construction to cache GNN embeddings for
original test queries using their correct precomputed RR edge labels,
matching the inference.py encoding exactly.

Usage:
    searcher = StructGNNSearcher(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        model_dir="outputs/paragnn/kuhperdata-exp/adapted_struct",
        bm25=bm25,
        bge_encoder=bge_model,
        rr_const_emb=rr_const_emb,
        alpha=0.7,
    )
    searcher.preindex_queries(qids, query_texts, para_store)
    results = searcher.search("ganti rugi wanprestasi", top_n=10)
"""
import numpy as np
import torch
import dgl

from paragnn import RR_LABELS
from paragnn.model import TestCaseGnn
from paragnn.structure import get_query_structure_features


class StructGNNSearcher:

    def __init__(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        corpus_embeddings: np.ndarray,
        bm25,
        bge_encoder,
        model_path: str,
        rr_const_emb: torch.Tensor,
        alpha: float = 0.8,
        structure_mode: str = "structural",
        act_dim: int = 64,
        pos_dim: int = 32,
        contranorm_scale: float = 0.0,
        contranorm_tau: float = 1.0,
        device: str = "cpu",
        reranker=None,
    ):
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.bm25 = bm25
        self.bge_encoder = bge_encoder
        self.alpha = alpha
        self.reranker = reranker
        self.device = torch.device(device)
        self.structure_mode = structure_mode

        self.corpus_embeddings = torch.tensor(corpus_embeddings, dtype=torch.float32)

        dim = 1024
        struct_input_dim = dim + act_dim + pos_dim
        self.model = TestCaseGnn(
            in_dim=dim, h_dim=dim, out_dim=dim,
            dropout=0.1, num_head=1,
            structure_mode=structure_mode,
            struct_input_dim=struct_input_dim,
            contranorm_scale=contranorm_scale,
            contranorm_tau=contranorm_tau,
        )
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.rr_const_emb = rr_const_emb
        self.rr_none_emb = rr_const_emb[RR_LABELS.index("NONE")]
        self._rr_label_to_idx = {lbl: i for i, lbl in enumerate(RR_LABELS)}

        if structure_mode == "structural":
            query_act, query_pos = get_query_structure_features(
                act_dim=act_dim, pos_dim=pos_dim,
            )
            self.query_struct_feat = torch.cat([query_act, query_pos])
        else:
            self.query_struct_feat = None

        # text → precomputed GNN query embedding (populated by preindex_queries)
        self._query_cache: dict[str, torch.Tensor] = {}

    def _get_rr_emb(self, label: str) -> torch.Tensor:
        idx = self._rr_label_to_idx.get(label, self._rr_label_to_idx["NONE"])
        return self.rr_const_emb[idx]

    @torch.no_grad()
    def _build_query_graph(
        self,
        para_emb: torch.Tensor,
        rr_labels: list[str],
    ) -> torch.Tensor:
        """Run GNN on a query's paragraph embeddings with given RR edge labels.

        Returns the query doc-node embedding (h[0]), raw unnormalized to match inference scoring.
        """
        doc_feat = para_emb.mean(dim=0)

        if self.structure_mode == "structural" and self.query_struct_feat is not None:
            sf = self.query_struct_feat
            doc_feat = torch.cat([doc_feat, sf])
            para_emb = torch.cat([
                para_emb,
                sf.unsqueeze(0).expand(para_emb.shape[0], -1),
            ], dim=1)

        n_paras = para_emb.shape[0]
        node_features = torch.cat([doc_feat.unsqueeze(0), para_emb], dim=0)

        u_ids = list(range(1, 1 + n_paras))
        v_ids = [0] * n_paras
        edge_feat = torch.stack([
            self._get_rr_emb(rr_labels[i] if i < len(rr_labels) else "NONE")
            for i in range(n_paras)
        ])

        g = dgl.graph((u_ids, v_ids), num_nodes=1 + n_paras)
        g = dgl.add_self_loop(g)

        n_self_loops = g.num_edges() - n_paras
        if n_self_loops > 0:
            edge_feat = torch.cat([
                edge_feat,
                torch.zeros(n_self_loops, edge_feat.shape[1]),
            ], dim=0)

        g.ndata["h"] = node_features
        g.edata["h"] = edge_feat
        g = g.to(self.device)

        node_h = g.ndata["h"]
        edge_h = g.edata["h"]
        if self.model.structure_mode == "structural":
            node_h = self.model.struct_proj(node_h)
        h = self.model.eugat_gnn(g, node_h, edge_h)

        return h[0].cpu()

    def preindex_queries(self, qids: list[str], texts: list[str], para_store) -> None:
        """Pre-encode original test queries using their exact precomputed embeddings
        and RR role labels from para_store, matching inference.py encoding exactly.

        Call this once before the agentic run. Agent sub-queries that aren't in
        qids will fall back to on-the-fly BGE encoding with NONE role labels.
        """
        print(f"  Pre-encoding {len(qids)} queries through StructGNN (faithful to inference)...")
        for qid, text in zip(qids, texts):
            para_emb = para_store.get_query_embedding(qid)   # precomputed BGE embeddings
            rr_labels = para_store.get_query_rr_labels(qid)  # actual role labels
            emb = self._build_query_graph(para_emb, rr_labels)
            self._query_cache[text] = emb
        print(f"  Cached {len(self._query_cache)} query embeddings.")

    @torch.no_grad()
    def _encode_query_gnn(self, query_text: str) -> torch.Tensor:
        """Encode query → GNN doc-node embedding.

        Uses preindexed embedding if available (exact inference match).
        Falls back to on-the-fly BGE + NONE role labels for sub-queries.
        """
        if query_text in self._query_cache:
            return self._query_cache[query_text]

        output = self.bge_encoder.encode(
            [query_text], batch_size=1, max_length=1024,
        )
        if isinstance(output, dict):
            para_emb = torch.tensor(np.array(output["dense_vecs"]), dtype=torch.float32)
        else:
            para_emb = torch.tensor(np.array(output), dtype=torch.float32)

        n_paras = para_emb.shape[0]
        rr_labels = ["NONE"] * n_paras
        return self._build_query_graph(para_emb, rr_labels)

    def search(
        self,
        query: str,
        top_n: int = 10,
        exclude_ids: set[str] | None = None,
        rerank: bool = True,
        rerank_top_k: int = 100,
    ) -> list[tuple[str, str, float]]:
        """Search using StructGNN scoring: alpha*gnn + (1-alpha)*bm25."""
        exclude_ids = exclude_ids or set()

        bm25_scores = self.bm25.transform(query)
        if hasattr(bm25_scores, "toarray"):
            bm25_scores = np.array(bm25_scores.toarray().flatten())
        else:
            bm25_scores = np.array(bm25_scores).flatten()

        query_encoded = self._encode_query_gnn(query)
        gnn_scores = (query_encoded @ self.corpus_embeddings.T).numpy()
        mean_s = gnn_scores.mean()
        std_s = gnn_scores.std()
        gnn_scores = (gnn_scores - mean_s) / (std_s + 1e-8)

        final_scores = self.alpha * gnn_scores + (1 - self.alpha) * bm25_scores

        ranked_indices = np.argsort(-final_scores)

        candidates = []
        for idx in ranked_indices:
            did = self.doc_ids[idx]
            if did in exclude_ids:
                continue
            candidates.append((idx, did, final_scores[idx]))
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
