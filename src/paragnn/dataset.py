"""Training dataset and collator for Para-GNN / Prox-GNN / StructGNN.

Adapted from IL-PCSR's SAILERDataset and SAILERDataCollator.
"""
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .graph_builder import GraphBuilder, ParagraphStore


class ParaGNNDataset(Dataset):
    """Training dataset: creates (query, positive, negatives) triples with BM25 scores."""

    def __init__(
        self,
        train_qids: List[str],
        qrels: Dict[str, Dict[str, int]],
        corpus_doc_ids: List[str],
        hard_negatives: Dict[str, List[str]],
        bm25_scores: torch.Tensor,
        num_negatives: int = 299,
    ):
        self.dataset = []
        self.corpus_doc_ids = corpus_doc_ids
        doc_id_to_idx = {d: i for i, d in enumerate(corpus_doc_ids)}

        print(f"Creating ParaGNN training dataset...")
        for qi, qid in enumerate(train_qids):
            if qid not in qrels:
                continue

            relevant_docs = [d for d in qrels[qid] if qrels[qid][d] > 0 and d in doc_id_to_idx]
            if not relevant_docs:
                continue

            for pos_did in relevant_docs:
                neg_candidates = [
                    d for d in hard_negatives.get(qid, [])[:400]
                    if d not in set(relevant_docs) and d in doc_id_to_idx
                ]

                if len(neg_candidates) < num_negatives:
                    remaining = [d for d in corpus_doc_ids if d not in set(relevant_docs) and d not in set(neg_candidates)]
                    extra = min(num_negatives - len(neg_candidates), len(remaining))
                    neg_candidates += random.sample(remaining, extra)

                sampled_negs = random.sample(neg_candidates, min(num_negatives, len(neg_candidates)))

                all_cand_ids = [pos_did] + sampled_negs
                all_cand_indices = [doc_id_to_idx[d] for d in all_cand_ids]
                bm25_slice = bm25_scores[qi][all_cand_indices]

                self.dataset.append({
                    "qid": qid,
                    "positive_did": pos_did,
                    "negative_dids": sampled_negs,
                    "bm25_scores": bm25_slice,
                })

        print(f"  Created {len(self.dataset)} training examples from {len(train_qids)} queries")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ParaGNNCollator:
    """Collates batch items into a DGL graph + training tensors.

    Supports all three structure modes via GraphBuilder.
    """

    def __init__(
        self,
        para_store: ParagraphStore,
        structure_mode: str = "none",
        proximity_radius: int = 50,
        structure_features: Optional[Dict[str, torch.Tensor]] = None,
        query_structure_feature: Optional[torch.Tensor] = None,
    ):
        self.para_store = para_store
        self.structure_mode = structure_mode
        self.proximity_radius = proximity_radius
        self.structure_features = structure_features
        self.query_structure_feature = query_structure_feature

    def __call__(self, batch: List[dict]) -> dict:
        query_id_list = []
        candidate_id_list = []

        for item in batch:
            if item["qid"] not in query_id_list:
                query_id_list.append(item["qid"])
            for did in [item["positive_did"]] + item["negative_dids"]:
                if did not in candidate_id_list:
                    candidate_id_list.append(did)

        graph_builder = GraphBuilder(
            query_id_list,
            candidate_id_list,
            self.para_store,
            structure_mode=self.structure_mode,
            proximity_radius=self.proximity_radius,
            structure_features=self.structure_features,
            query_structure_feature=self.query_structure_feature,
        )

        query_pos = []
        candidate_pos = []
        bm25_scores = []

        for item in batch:
            query_pos.append(query_id_list.index(item["qid"]))

            cand_ids = [item["positive_did"]] + item["negative_dids"]
            cand_pos = [candidate_id_list.index(d) for d in cand_ids]
            candidate_pos.append(cand_pos)
            bm25_scores.append(item["bm25_scores"])

        num_cands = len(candidate_pos[0])
        labels = torch.zeros(len(batch), num_cands)
        labels[:, 0] = 1

        return {
            "query_pos": torch.tensor(query_pos, dtype=torch.long),
            "candidate_pos": torch.tensor(candidate_pos, dtype=torch.long),
            "bm25_scores": torch.stack(bm25_scores),
            "candidate_relevance_labels": labels,
            "graph": graph_builder.graph,
        }
