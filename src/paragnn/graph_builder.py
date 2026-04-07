"""Graph construction for Para-GNN.

Builds DGL graphs from pre-computed paragraph embeddings and RR labels.
Adapted from IL-PCSR's GraphGenerator for BEIR-format datasets.

Graph layout:
  Nodes: [query_doc_0, ..., query_doc_N, cand_doc_0, ..., cand_doc_M,
          query_para_0_0, query_para_0_1, ..., cand_para_0_0, ...]
  Edges: paragraph_node → parent_doc_node
  Edge features: RR label embedding (1024d) for that paragraph
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import dgl

from paragnn import RR_LABELS


class ParagraphStore:
    """Loads and caches pre-computed paragraph embeddings and RR labels."""

    def __init__(self, output_dir: str, method: str = "adapted",
                 rr_labels_path: Optional[str] = None):
        """
        Args:
            output_dir: path to outputs/paragnn/{dataset}/
            method: "full" (LLM-labeled RR) or "adapted" (single para, NONE)
            rr_labels_path: path to rr_labels.json (method="full" only)
        """
        self.output_dir = Path(output_dir)
        self.emb_dir = self.output_dir / "embeddings"
        self.method = method

        # Load RR constant embeddings
        self.rr_const_emb = torch.load(self.emb_dir / "EMBD_CONST.pt")  # (13, 1024)
        self.rr_label_to_idx = {label: i for i, label in enumerate(RR_LABELS)}

        # Load paragraph metadata
        with open(self.output_dir / "query_paragraphs.json", "r", encoding="utf-8") as f:
            self.query_paras = json.load(f)  # {qid: [{sentence, role}, ...]}
        with open(self.output_dir / "corpus_paragraphs.json", "r", encoding="utf-8") as f:
            self.corpus_paras = json.load(f)  # {doc_id: [{sentence, role}, ...]}

        # If method="full", override query paragraph roles with LLM-labeled roles
        if method == "full" and rr_labels_path:
            with open(rr_labels_path, "r", encoding="utf-8") as f:
                llm_labels = json.load(f)  # {qid: [{sentence, role}, ...]}
            # Merge: use LLM roles where available
            for qid, paras in llm_labels.items():
                if qid in self.query_paras:
                    self.query_paras[qid] = paras

        # Embedding cache
        self._emb_cache: Dict[str, torch.Tensor] = {}

    def get_query_embedding(self, qid: str) -> torch.Tensor:
        """Load paragraph embeddings for a query. Returns (num_paras, 1024)."""
        if qid not in self._emb_cache:
            path = self.emb_dir / "queries" / f"{qid}.pt"
            self._emb_cache[qid] = torch.load(path, map_location="cpu")
        return self._emb_cache[qid]

    def get_corpus_embedding(self, doc_id: str) -> torch.Tensor:
        """Load paragraph embeddings for a corpus doc. Returns (num_paras, 1024)."""
        key = f"c_{doc_id}"
        if key not in self._emb_cache:
            path = self.emb_dir / "corpus" / f"{doc_id}.pt"
            self._emb_cache[key] = torch.load(path, map_location="cpu")
        return self._emb_cache[key]

    def get_query_rr_labels(self, qid: str) -> List[str]:
        """Get RR label strings for each paragraph of a query."""
        paras = self.query_paras.get(qid, [{"sentence": "", "role": "NONE"}])
        return [p["role"] for p in paras]

    def get_rr_embedding(self, label: str) -> torch.Tensor:
        """Get the embedding for an RR label string. Returns (1024,)."""
        idx = self.rr_label_to_idx.get(label, self.rr_label_to_idx["NONE"])
        return self.rr_const_emb[idx]

    def clear_cache(self):
        self._emb_cache.clear()


class GraphBuilder:
    """Builds a DGL graph from query IDs + candidate IDs using ParagraphStore.

    Matches IL-PCSR's GraphGenerator layout exactly:
      - Document nodes: [queries | candidates]
      - Paragraph nodes appended after document nodes
      - Edges: paragraph → parent document
      - Edge features: RR label embedding
    """

    def __init__(self, query_ids: List[str], candidate_ids: List[str],
                 para_store: ParagraphStore):
        self.query_ids = query_ids
        self.candidate_ids = candidate_ids
        self.all_doc_ids = query_ids + candidate_ids

        n_docs = len(self.all_doc_ids)

        # Collect paragraph info
        u_ids = []  # source (paragraph nodes)
        v_ids = []  # destination (document nodes)
        edge_features = []
        node_features = []
        query_para_count = 0
        candidate_para_count = 0

        # Document node features (mean of paragraph embeddings)
        doc_node_features = []

        para_offset = n_docs  # paragraph nodes start after all doc nodes

        for idx, doc_id in enumerate(self.all_doc_ids):
            is_query = idx < len(query_ids)

            if is_query:
                emb = para_store.get_query_embedding(doc_id)  # (num_paras, 1024)
                rr_labels = para_store.get_query_rr_labels(doc_id)
            else:
                emb = para_store.get_corpus_embedding(doc_id)  # (num_paras, 1024)
                rr_labels = ["NONE"] * emb.shape[0]  # statutes always NONE

            num_paras = emb.shape[0]

            # Document node = mean of paragraph embeddings
            doc_node_features.append(emb.mean(dim=0))

            # Paragraph nodes and edges
            for i in range(num_paras):
                node_features.append(emb[i])
                u_ids.append(para_offset)
                v_ids.append(idx)
                edge_features.append(para_store.get_rr_embedding(rr_labels[i] if i < len(rr_labels) else "NONE"))
                para_offset += 1

            if is_query:
                query_para_count += num_paras
            else:
                candidate_para_count += num_paras

        # Stack all node features: [doc_nodes | para_nodes]
        all_node_features = torch.stack(doc_node_features + node_features)  # (n_docs + n_paras, 1024)
        all_edge_features = torch.stack(edge_features)  # (n_edges, 1024)

        # Build DGL graph
        self.graph = dgl.graph((u_ids, v_ids), num_nodes=all_node_features.shape[0])
        self.graph = dgl.add_self_loop(self.graph)

        # Pad edge features for self-loops (zero vectors)
        n_self_loops = self.graph.num_edges() - len(edge_features)
        if n_self_loops > 0:
            pad = torch.zeros(n_self_loops, all_edge_features.shape[1])
            all_edge_features = torch.cat([all_edge_features, pad], dim=0)

        self.graph.ndata["h"] = all_node_features
        self.graph.edata["h"] = all_edge_features

        # Node masks
        n_total = all_node_features.shape[0]
        n_queries = len(query_ids)
        n_candidates = len(candidate_ids)

        query_mask = torch.zeros(n_total, dtype=torch.float32)
        query_mask[:n_queries] = 1.0
        self.graph.ndata["query_mask"] = query_mask

        candidate_mask = torch.zeros(n_total, dtype=torch.float32)
        candidate_mask[n_queries:n_docs] = 1.0
        self.graph.ndata["candidate_mask"] = candidate_mask

        query_para_mask = torch.zeros(n_total, dtype=torch.float32)
        query_para_mask[n_docs:n_docs + query_para_count] = 1.0
        self.graph.ndata["query_para_mask"] = query_para_mask

        candidate_para_mask = torch.zeros(n_total, dtype=torch.float32)
        candidate_para_mask[n_docs + query_para_count:] = 1.0
        self.graph.ndata["candidate_para_mask"] = candidate_para_mask
