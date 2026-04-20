"""Graph construction for Para-GNN / Prox-GNN / StructGNN.

Builds DGL graphs from pre-computed paragraph embeddings and RR labels.
Adapted from IL-PCSR's GraphGenerator for BEIR-format datasets.

Graph layout:
  Nodes: [query_doc_0, ..., query_doc_N, cand_doc_0, ..., cand_doc_M,
          query_para_0_0, query_para_0_1, ..., cand_para_0_0, ...]
  Edges: paragraph_node -> parent_doc_node
  Edge features: RR label embedding (1024d) for that paragraph

Structure modes:
  - "none": Para-GNN base. Node features = BGE-M3 only (1024d).
  - "proximity": Prox-GNN. Same node features + explicit proximity edges.
  - "structural": StructGNN. Node features = [BGE-M3 | act_hash | pos_enc]
    (1120d), projected to 1024d by the model before EUGAT.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import dgl

from paragnn import RR_LABELS, FACT_TYPES


class ParagraphStore:
    """Loads and caches pre-computed paragraph embeddings and RR/fact-type labels."""

    def __init__(self, output_dir: str, method: str = "adapted",
                 rr_labels_path: Optional[str] = None,
                 use_fact_types: bool = False):
        self.output_dir = Path(output_dir)
        self.emb_dir = self.output_dir / "embeddings"
        self.method = method
        self.use_fact_types = use_fact_types

        self.rr_const_emb = torch.load(self.emb_dir / "EMBD_CONST.pt")  # (13, 1024)
        self.rr_label_to_idx = {label: i for i, label in enumerate(RR_LABELS)}

        self.fact_type_emb = None
        self.fact_type_to_idx = {}
        if use_fact_types:
            ft_path = self.emb_dir / "EMBD_FACT_TYPES.pt"
            if ft_path.exists():
                self.fact_type_emb = torch.load(ft_path)  # (5, 1024)
                self.fact_type_to_idx = {label: i for i, label in enumerate(FACT_TYPES)}
            else:
                print(f"  WARNING: --use_fact_types but {ft_path} not found, falling back to NONE")
                self.use_fact_types = False

        if use_fact_types and (self.output_dir / "query_paragraphs_facts.json").exists():
            with open(self.output_dir / "query_paragraphs_facts.json", "r", encoding="utf-8") as f:
                self.query_paras = json.load(f)
        else:
            with open(self.output_dir / "query_paragraphs.json", "r", encoding="utf-8") as f:
                self.query_paras = json.load(f)

        with open(self.output_dir / "corpus_paragraphs.json", "r", encoding="utf-8") as f:
            self.corpus_paras = json.load(f)

        if method == "full" and rr_labels_path:
            with open(rr_labels_path, "r", encoding="utf-8") as f:
                llm_labels = json.load(f)
            for qid, paras in llm_labels.items():
                if qid in self.query_paras:
                    self.query_paras[qid] = paras

        self._emb_cache: Dict[str, torch.Tensor] = {}

    def get_query_embedding(self, qid: str) -> torch.Tensor:
        if qid not in self._emb_cache:
            if self.use_fact_types:
                path = self.emb_dir / "queries_facts" / f"{qid}.pt"
                if not path.exists():
                    path = self.emb_dir / "queries" / f"{qid}.pt"
            else:
                path = self.emb_dir / "queries" / f"{qid}.pt"
            self._emb_cache[qid] = torch.load(path, map_location="cpu")
        return self._emb_cache[qid]

    def get_corpus_embedding(self, doc_id: str) -> torch.Tensor:
        key = f"c_{doc_id}"
        if key not in self._emb_cache:
            path = self.emb_dir / "corpus" / f"{doc_id}.pt"
            self._emb_cache[key] = torch.load(path, map_location="cpu")
        return self._emb_cache[key]

    def get_query_rr_labels(self, qid: str) -> List[str]:
        paras = self.query_paras.get(qid, [{"sentence": "", "role": "NONE"}])
        labels = []
        for p in paras:
            if isinstance(p, dict):
                labels.append(p.get("role", "NONE"))
            elif isinstance(p, str):
                labels.append("NONE")
            else:
                labels.append("NONE")
        return labels

    def get_rr_embedding(self, label: str) -> torch.Tensor:
        if self.use_fact_types and label in self.fact_type_to_idx:
            return self.fact_type_emb[self.fact_type_to_idx[label]]
        idx = self.rr_label_to_idx.get(label, self.rr_label_to_idx["NONE"])
        return self.rr_const_emb[idx]

    def clear_cache(self):
        self._emb_cache.clear()


class GraphBuilder:
    """Builds a DGL graph from query IDs + candidate IDs.

    Supports three structure modes:
      - "none": Para-GNN base (paragraph->doc edges only)
      - "proximity": Prox-GNN (+ proximity edges between nearby statutes)
      - "structural": StructGNN (structural features concatenated to node embeddings)
    """

    def __init__(
        self,
        query_ids: List[str],
        candidate_ids: List[str],
        para_store: ParagraphStore,
        structure_mode: str = "none",
        proximity_radius: int = 50,
        structure_features: Optional[Dict[str, torch.Tensor]] = None,
        query_structure_feature: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            structure_mode: "none", "proximity", or "structural"
            proximity_radius: Prox-GNN only. Connect statutes within N articles.
            structure_features: StructGNN only. {doc_id: (act_dim+pos_dim,)} precomputed.
            query_structure_feature: StructGNN only. (act_dim+pos_dim,) for query nodes.
        """
        self.query_ids = query_ids
        self.candidate_ids = candidate_ids
        self.all_doc_ids = query_ids + candidate_ids

        n_docs = len(self.all_doc_ids)

        u_ids = []
        v_ids = []
        edge_features = []
        node_features = []
        query_para_count = 0
        candidate_para_count = 0
        doc_node_features = []

        para_offset = n_docs

        for idx, doc_id in enumerate(self.all_doc_ids):
            is_query = idx < len(query_ids)

            if is_query:
                emb = para_store.get_query_embedding(doc_id)
                rr_labels = para_store.get_query_rr_labels(doc_id)
            else:
                emb = para_store.get_corpus_embedding(doc_id)
                rr_labels = ["NONE"] * emb.shape[0]

            num_paras = emb.shape[0]

            doc_feat = emb.mean(dim=0)  # (1024,)

            if structure_mode == "structural":
                if is_query:
                    struct_feat = query_structure_feature
                else:
                    struct_feat = structure_features.get(doc_id)
                if struct_feat is not None:
                    doc_feat = torch.cat([doc_feat, struct_feat])

            doc_node_features.append(doc_feat)

            for i in range(num_paras):
                para_feat = emb[i]
                if structure_mode == "structural":
                    if is_query:
                        struct_feat = query_structure_feature
                    else:
                        struct_feat = structure_features.get(doc_id)
                    if struct_feat is not None:
                        para_feat = torch.cat([para_feat, struct_feat])
                node_features.append(para_feat)
                u_ids.append(para_offset)
                v_ids.append(idx)
                edge_features.append(
                    para_store.get_rr_embedding(rr_labels[i] if i < len(rr_labels) else "NONE")
                )
                para_offset += 1

            if is_query:
                query_para_count += num_paras
            else:
                candidate_para_count += num_paras

        # Prox-GNN: add proximity edges between nearby statute nodes
        if structure_mode == "proximity" and proximity_radius > 0:
            n_queries = len(query_ids)
            cand_nums = []
            for cid in candidate_ids:
                match = re.match(r'(\d+)', cid)
                cand_nums.append(int(match.group(1)) if match else -1)

            proximity_edge_feat = para_store.get_rr_embedding("NONE")

            for i in range(len(candidate_ids)):
                if cand_nums[i] < 0:
                    continue
                node_i = n_queries + i
                for j in range(i + 1, len(candidate_ids)):
                    if cand_nums[j] < 0:
                        continue
                    if abs(cand_nums[i] - cand_nums[j]) <= proximity_radius:
                        node_j = n_queries + j
                        u_ids.append(node_i)
                        v_ids.append(node_j)
                        edge_features.append(proximity_edge_feat)
                        u_ids.append(node_j)
                        v_ids.append(node_i)
                        edge_features.append(proximity_edge_feat)

        all_node_features = torch.stack(doc_node_features + node_features)
        all_edge_features = torch.stack(edge_features)

        self.graph = dgl.graph((u_ids, v_ids), num_nodes=all_node_features.shape[0])
        self.graph = dgl.add_self_loop(self.graph)

        n_self_loops = self.graph.num_edges() - len(edge_features)
        if n_self_loops > 0:
            pad = torch.zeros(n_self_loops, all_edge_features.shape[1])
            all_edge_features = torch.cat([all_edge_features, pad], dim=0)

        self.graph.ndata["h"] = all_node_features
        self.graph.edata["h"] = all_edge_features

        n_total = all_node_features.shape[0]
        n_queries = len(query_ids)

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
