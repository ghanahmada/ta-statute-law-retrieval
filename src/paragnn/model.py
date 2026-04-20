"""Para-GNN / Prox-GNN / StructGNN model for statute retrieval.

Adapted from IL-PCSR (Paul et al., EMNLP 2025).
Changed embedding dimension from 768 (all-mpnet-base-v2) to 1024 (BGE-M3).

StructGNN adds a projection layer (LayerNorm -> Linear) to map concatenated
[semantic | act_hash | pos_enc] features (1120d) back to 1024d before EUGAT.
"""
import math

import torch
import torch.nn as nn

from .eugat import EUGATGNN


class SimpleFFNN(nn.Module):
    """Learns per-query alpha weight for BM25/GNN score blending."""

    def __init__(self, input_dim=1024, output_dim=1, dropout_rate=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.dropout(self.fc(x)))


class StructureProjection(nn.Module):
    """Projects concatenated [semantic | structure] features to EUGAT input dim.

    Used by StructGNN only. Para-GNN and Prox-GNN bypass this layer.
    """

    def __init__(self, input_dim=1120, output_dim=1024):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class CaseGnn(nn.Module):
    """Training model: computes loss from query-candidate pairs + BM25 scores."""

    def __init__(self, in_dim=1024, h_dim=1024, out_dim=1024, dropout=0.1,
                 num_head=1, structure_mode="none", struct_input_dim=1120):
        super().__init__()
        self.structure_mode = structure_mode

        if structure_mode == "structural":
            self.struct_proj = StructureProjection(struct_input_dim, in_dim)

        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head)
        self.ffnn = SimpleFFNN(input_dim=in_dim)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, query_pos, candidate_pos, bm25_scores,
                candidate_relevance_labels, graph, ips_weights=None):
        node_h = graph.ndata["h"]
        edge_h = graph.edata["h"]

        if self.structure_mode == "structural":
            node_h = self.struct_proj(node_h)

        h = self.eugat_gnn(graph, node_h, edge_h)

        query_encoded = h[graph.ndata["query_mask"].bool()][query_pos]
        alphas = self.ffnn(query_encoded)

        h_candidate = h[graph.ndata["candidate_mask"].bool()]
        candidate_encoded = torch.gather(
            h_candidate.unsqueeze(0).repeat(query_encoded.size(0), 1, 1),
            1,
            candidate_pos.unsqueeze(2).repeat(1, 1, h_candidate.size(1)),
        )

        scores = torch.bmm(
            query_encoded.unsqueeze(1),
            candidate_encoded.permute(0, 2, 1),
        ).squeeze(1)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / (std_scores + 1e-8)

        scores = alphas * normalized_scores + (1 - alphas) * bm25_scores

        per_sample_loss = self.loss(scores, candidate_relevance_labels.argmax(dim=1))
        if ips_weights is not None:
            per_sample_loss = per_sample_loss * ips_weights
        return per_sample_loss.mean()


class TestCaseGnn(nn.Module):
    """Inference model: scores all candidates for all queries."""

    def __init__(self, in_dim=1024, h_dim=1024, out_dim=1024, dropout=0.1,
                 num_head=1, structure_mode="none", struct_input_dim=1120):
        super().__init__()
        self.structure_mode = structure_mode

        if structure_mode == "structural":
            self.struct_proj = StructureProjection(struct_input_dim, in_dim)

        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head)
        self.ffnn = SimpleFFNN(input_dim=in_dim)

    def forward(self, bm25_scores, graph):
        node_h = graph.ndata["h"]
        edge_h = graph.edata["h"]

        if self.structure_mode == "structural":
            node_h = self.struct_proj(node_h)

        h = self.eugat_gnn(graph, node_h, edge_h)

        query_encoded = h[graph.ndata["query_mask"].bool()]
        alphas = self.ffnn(query_encoded)
        candidate_encoded = h[graph.ndata["candidate_mask"].bool()]

        scores = torch.matmul(query_encoded, candidate_encoded.T)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / (std_scores + 1e-8)

        scores = alphas * normalized_scores + (1 - alphas) * bm25_scores

        return scores, alphas
