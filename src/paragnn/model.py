"""Para-GNN model for statute retrieval.

Adapted from IL-PCSR (Paul et al., EMNLP 2025).
Changed embedding dimension from 768 (all-mpnet-base-v2) to 1024 (BGE-M3).
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


class CaseGnn(nn.Module):
    """Training model: computes loss from query-candidate pairs + BM25 scores."""

    def __init__(self, in_dim=1024, h_dim=1024, out_dim=1024, dropout=0.1, num_head=1):
        super().__init__()
        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head)
        self.ffnn = SimpleFFNN(input_dim=in_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, query_pos, candidate_pos, bm25_scores,
                candidate_relevance_labels, graph):
        """
        Args:
            query_pos: (batch_size,) indices into query doc nodes
            candidate_pos: (batch_size, num_cands) indices into candidate doc nodes
            bm25_scores: (batch_size, num_cands) pre-computed BM25 scores
            candidate_relevance_labels: (batch_size, num_cands) one-hot labels
            graph: DGL graph with ndata["h"], edata["h"], ndata masks
        Returns:
            loss: scalar CrossEntropyLoss
        """
        h = self.eugat_gnn(graph, graph.ndata["h"], graph.edata["h"])

        # Extract query representations
        query_encoded = h[graph.ndata["query_mask"].bool()][query_pos]
        alphas = self.ffnn(query_encoded)  # (batch_size, 1)

        # Extract candidate representations
        h_candidate = h[graph.ndata["candidate_mask"].bool()]
        candidate_encoded = torch.gather(
            h_candidate.unsqueeze(0).repeat(query_encoded.size(0), 1, 1),
            1,
            candidate_pos.unsqueeze(2).repeat(1, 1, h_candidate.size(1)),
        )  # (batch_size, num_cands, embed_dim)

        # Dot product scores
        scores = torch.bmm(
            query_encoded.unsqueeze(1),
            candidate_encoded.permute(0, 2, 1),
        ).squeeze(1)  # (batch_size, num_cands)

        # Z-normalize GNN scores
        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / (std_scores + 1e-8)

        # Hybrid: alpha * GNN + (1-alpha) * BM25
        scores = alphas * normalized_scores + (1 - alphas) * bm25_scores

        return self.loss(scores, candidate_relevance_labels.argmax(dim=1))


class TestCaseGnn(nn.Module):
    """Inference model: scores all candidates for all queries."""

    def __init__(self, in_dim=1024, h_dim=1024, out_dim=1024, dropout=0.1, num_head=1):
        super().__init__()
        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head)
        self.ffnn = SimpleFFNN(input_dim=in_dim)

    def forward(self, bm25_scores, graph):
        """
        Args:
            bm25_scores: (num_queries, num_candidates) pre-computed BM25 scores
            graph: DGL graph (full test set)
        Returns:
            scores: (num_queries, num_candidates) final hybrid scores
            alphas: (num_queries, 1) learned alpha weights per query
        """
        h = self.eugat_gnn(graph, graph.ndata["h"], graph.edata["h"])

        query_encoded = h[graph.ndata["query_mask"].bool()]  # (num_queries, embed_dim)
        alphas = self.ffnn(query_encoded)  # (num_queries, 1)
        candidate_encoded = h[graph.ndata["candidate_mask"].bool()]  # (num_candidates, embed_dim)

        scores = torch.matmul(query_encoded, candidate_encoded.T)  # (num_queries, num_candidates)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / (std_scores + 1e-8)

        scores = alphas * normalized_scores + (1 - alphas) * bm25_scores

        return scores, alphas
