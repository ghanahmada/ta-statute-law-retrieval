"""Training loop for Para-GNN / Prox-GNN / StructGNN.

Adapted from IL-PCSR's train_paragnn_plus_bm25_for_secs.py.
"""
import json
import os
from math import ceil
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from util import set_seed
from .model import CaseGnn, TestCaseGnn
from .graph_builder import ParagraphStore, GraphBuilder


class ParaGNNTrainer:
    """Trains Para-GNN / Prox-GNN / StructGNN and evaluates on test set."""

    def __init__(self, config, para_store: ParagraphStore,
                 structure_features: Optional[Dict[str, torch.Tensor]] = None,
                 query_structure_feature: Optional[torch.Tensor] = None):
        self.config = config
        self.para_store = para_store
        self.structure_features = structure_features
        self.query_structure_feature = query_structure_feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_dataset,
        collator,
        bm25_test_scores: torch.Tensor,
        test_query_ids: list,
        test_corpus_ids: list,
        test_gold: dict,
    ):
        set_seed(42)

        # Output to method-specific subfolder
        mode = self.config.structure_mode
        method_suffix = self.config.method
        if mode == "proximity":
            method_suffix = f"{method_suffix}_prox{self.config.proximity_radius}"
        elif mode == "structural":
            method_suffix = f"{method_suffix}_struct"
        output_dir = f"{self.config.output_dir}/{self.config.dataset}/{method_suffix}"
        os.makedirs(output_dir, exist_ok=True)

        dim = self.config.embed_dim
        struct_input_dim = dim + self.config.act_dim + self.config.pos_dim
        model = CaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                        dropout=self.config.dropout, num_head=self.config.num_heads,
                        structure_mode=mode, struct_input_dim=struct_input_dim)

        # Resume from checkpoint if exists
        start_epoch = 0
        best_metric = 0
        log = []
        resume_path = f"{output_dir}/resume_checkpoint.pt"

        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_metric = checkpoint.get("best_metric", 0)
            log = checkpoint.get("log", [])
            print(f"  Resumed at epoch {start_epoch}, best metric={best_metric:.4f}")

        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        if os.path.exists(resume_path):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # IL-PCSR LR schedule: linear warmup + 1.2x total steps (never fully decays)
        steps_per_epoch = ceil(len(train_dataset) / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=int(total_steps * 1.2),
        )

        # Early stopping
        patience = 10
        epochs_without_improvement = 0

        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        # Handle BM25 NaN values
        bm25_test_scores = torch.nan_to_num(bm25_test_scores, nan=0.0)

        # Build test graph once
        print("Building test graph...")
        test_graph = GraphBuilder(
            test_query_ids, test_corpus_ids, self.para_store,
            structure_mode=mode,
            proximity_radius=self.config.proximity_radius,
            structure_features=self.structure_features,
            query_structure_feature=self.query_structure_feature,
        ).graph
        test_graph = test_graph.to(self.device)
        bm25_test_scores = bm25_test_scores.to(self.device)

        # Build gold matrix
        gold_matrix = torch.zeros(len(test_query_ids), len(test_corpus_ids))
        doc_id_to_idx = {d: i for i, d in enumerate(test_corpus_ids)}
        for qi, qid in enumerate(test_query_ids):
            if qid in test_gold:
                for did, score in test_gold[qid].items():
                    if did in doc_id_to_idx and score > 0:
                        gold_matrix[qi][doc_id_to_idx[did]] = 1

        mode_name = {"none": "Para-GNN", "proximity": "Prox-GNN", "structural": "StructGNN"}[mode]
        print(f"Training {mode_name} for epochs {start_epoch+1}-{self.config.epochs}...")
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
                torch.cuda.empty_cache()

                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if hasattr(batch_on_device["graph"], "to"):
                    batch_on_device["graph"] = batch_on_device["graph"].to(self.device)

                # OOM handling (from IL-PCSR)
                try:
                    loss = model(
                        query_pos=batch_on_device["query_pos"],
                        candidate_pos=batch_on_device["candidate_pos"],
                        bm25_scores=batch_on_device["bm25_scores"],
                        candidate_relevance_labels=batch_on_device["candidate_relevance_labels"],
                        graph=batch_on_device["graph"],
                    )
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    print("  OOM, skipping batch")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Evaluate: get raw GNN scores then grid search alpha
            gnn_scores, avg_alpha = self._get_gnn_scores(model, test_graph)
            best_alpha, mrr, recall, hit_rate = self._grid_search_alpha(
                gnn_scores, bm25_test_scores.cpu(), gold_matrix
            )

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "mrr@10": mrr,
                "recall@10": recall,
                "hit_rate": hit_rate,
                "best_alpha": best_alpha,
                "learned_alpha": avg_alpha,
            }
            log.append(log_entry)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} MRR={mrr:.4f} R@10={recall:.4f} Hit={hit_rate:.1%} alpha={best_alpha:.2f} (learned={avg_alpha:.3f})")

            if mrr > best_metric:
                best_metric = mrr
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                print(f"  → New best MRR={mrr:.4f} at alpha={best_alpha:.2f}, saved model")
            else:
                epochs_without_improvement += 1

            # Save resume checkpoint
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "log": log,
            }, f"{output_dir}/resume_checkpoint.pt")

            with open(f"{output_dir}/training_log.json", "w") as f:
                json.dump(log, f, indent=2)

            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Final grid search with best model
        print(f"\n{'='*60}")
        print(f"  Final evaluation with best model + alpha grid search")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(f"{output_dir}/best_model.pt", map_location="cpu"))
        model = model.to(self.device)
        gnn_scores, _ = self._get_gnn_scores(model, test_graph)

        print(f"\n  Alpha grid search results:")
        print(f"  {'Alpha':<8} {'MRR@10':<10} {'R@10':<10} {'Hit':<10}")
        print(f"  {'-'*38}")
        best_alpha, best_mrr, best_recall, best_hit = 0, 0, 0, 0
        for alpha in np.arange(0.0, 1.05, 0.1):
            scores = alpha * gnn_scores + (1 - alpha) * bm25_test_scores.cpu()
            mrr, recall, hit_rate = self._compute_metrics(scores, gold_matrix)
            marker = " ←" if mrr > best_mrr else ""
            print(f"  {alpha:<8.1f} {mrr:<10.4f} {recall:<10.4f} {hit_rate:<10.1%}{marker}")
            if mrr > best_mrr:
                best_mrr = mrr
                best_recall = recall
                best_hit = hit_rate
                best_alpha = alpha

        print(f"\n  Best: alpha={best_alpha:.1f} MRR={best_mrr:.4f} R@10={best_recall:.4f} Hit={best_hit:.1%}")
        print(f"\nTraining complete. Best MRR@10: {best_mrr:.4f}")
        return best_mrr

    @torch.no_grad()
    def _get_gnn_scores(self, train_model, test_graph):
        """Get raw GNN scores (before alpha blending)."""
        dim = self.config.embed_dim
        mode = self.config.structure_mode
        struct_input_dim = dim + self.config.act_dim + self.config.pos_dim
        test_model = TestCaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                                  dropout=self.config.dropout, num_head=self.config.num_heads,
                                  structure_mode=mode, struct_input_dim=struct_input_dim)
        test_model.load_state_dict(train_model.state_dict(), strict=False)
        test_model = test_model.to(self.device)
        test_model.eval()

        node_h = test_graph.ndata["h"]
        edge_h = test_graph.edata["h"]

        if mode == "structural":
            node_h = test_model.struct_proj(node_h)

        h = test_model.eugat_gnn(test_graph, node_h, edge_h)
        query_encoded = h[test_graph.ndata["query_mask"].bool()]
        candidate_encoded = h[test_graph.ndata["candidate_mask"].bool()]
        alphas = test_model.ffnn(query_encoded)

        scores = torch.matmul(query_encoded, candidate_encoded.T)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        gnn_scores = (scores - mean_scores) / (std_scores + 1e-8)

        return gnn_scores.cpu(), alphas.mean().item()

    def _grid_search_alpha(self, gnn_scores, bm25_scores, gold_matrix):
        """Find optimal alpha via grid search."""
        best_alpha, best_mrr, best_recall, best_hit = 0, 0, 0, 0
        for alpha in np.arange(0.0, 1.05, 0.1):
            scores = alpha * gnn_scores + (1 - alpha) * bm25_scores
            mrr, recall, hit_rate = self._compute_metrics(scores, gold_matrix)
            if mrr > best_mrr:
                best_mrr = mrr
                best_recall = recall
                best_hit = hit_rate
                best_alpha = alpha
        return best_alpha, best_mrr, best_recall, best_hit

    def _compute_metrics(self, scores, gold_matrix, k=10):
        """Compute MRR@k, Recall@k, Hit Rate from score matrix."""
        mrr_sum = 0
        recall_sum = 0
        hit_count = 0
        n_queries = gold_matrix.shape[0]

        for qi in range(n_queries):
            relevant = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
            if not relevant:
                continue

            ranked = torch.argsort(scores[qi], descending=True)[:k].tolist()

            for rank, idx in enumerate(ranked):
                if idx in relevant:
                    mrr_sum += 1.0 / (rank + 1)
                    break

            hits = len(set(ranked) & set(relevant))
            recall_sum += hits / len(relevant)
            if hits > 0:
                hit_count += 1

        return mrr_sum / n_queries, recall_sum / n_queries, hit_count / n_queries
