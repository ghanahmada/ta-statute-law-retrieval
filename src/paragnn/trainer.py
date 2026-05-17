"""Training loop for Para-GNN / Prox-GNN / StructGNN.

Adapted from IL-PCSR's train_paragnn_plus_bm25_for_secs.py.

Uses validation set for early stopping and alpha selection.
Test set metrics are reported but never used for model/hyperparameter decisions.
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
                 query_structure_feature: Optional[torch.Tensor] = None,
                 use_fact_types: bool = False):
        self.config = config
        self.para_store = para_store
        self.structure_features = structure_features
        self.query_structure_feature = query_structure_feature
        self.use_fact_types = use_fact_types
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_graph(self, query_ids, corpus_ids):
        mode = self.config.structure_mode
        return GraphBuilder(
            query_ids, corpus_ids, self.para_store,
            structure_mode=mode,
            proximity_radius=self.config.proximity_radius,
            structure_features=self.structure_features,
            query_structure_feature=self.query_structure_feature,
        ).graph

    def _build_gold_matrix(self, query_ids, corpus_ids, gold):
        gold_matrix = torch.zeros(len(query_ids), len(corpus_ids))
        doc_id_to_idx = {d: i for i, d in enumerate(corpus_ids)}
        for qi, qid in enumerate(query_ids):
            if qid in gold:
                for did, score in gold[qid].items():
                    if did in doc_id_to_idx and score > 0:
                        gold_matrix[qi][doc_id_to_idx[did]] = 1
        return gold_matrix

    def train(
        self,
        train_dataset,
        collator,
        bm25_val_scores: torch.Tensor,
        val_query_ids: list,
        val_corpus_ids: list,
        val_gold: dict,
        bm25_test_scores: torch.Tensor,
        test_query_ids: list,
        test_corpus_ids: list,
        test_gold: dict,
    ):
        set_seed(42)

        mode = self.config.structure_mode
        method_suffix = self.config.method
        if mode == "proximity":
            method_suffix = f"{method_suffix}_prox{self.config.proximity_radius}"
        elif mode == "structural":
            method_suffix = f"{method_suffix}_struct"
        if self.use_fact_types:
            method_suffix = f"{method_suffix}_facts"
        output_dir = f"{self.config.output_dir}/{self.config.dataset}/{method_suffix}"
        os.makedirs(output_dir, exist_ok=True)

        dim = self.config.embed_dim
        struct_input_dim = dim + self.config.act_dim + self.config.pos_dim
        model = CaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                        dropout=self.config.dropout, num_head=self.config.num_heads,
                        structure_mode=mode, struct_input_dim=struct_input_dim,
                        contranorm_scale=self.config.contranorm_scale,
                        contranorm_tau=self.config.contranorm_tau)

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

        steps_per_epoch = ceil(len(train_dataset) / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=int(total_steps * 1.2),
        )

        patience = self.config.patience
        epochs_without_improvement = 0

        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        bm25_val_scores = torch.nan_to_num(bm25_val_scores, nan=0.0)
        bm25_test_scores = torch.nan_to_num(bm25_test_scores, nan=0.0)

        # Build val and test graphs once
        print("Building val graph...")
        val_graph = self._build_graph(val_query_ids, val_corpus_ids)
        val_graph = val_graph.to(self.device)
        bm25_val_scores = bm25_val_scores.to(self.device)
        val_gold_matrix = self._build_gold_matrix(val_query_ids, val_corpus_ids, val_gold)

        print("Building test graph...")
        test_graph = self._build_graph(test_query_ids, test_corpus_ids)
        test_graph = test_graph.to(self.device)
        bm25_test_scores = bm25_test_scores.to(self.device)
        test_gold_matrix = self._build_gold_matrix(test_query_ids, test_corpus_ids, test_gold)

        mode_name = {"none": "Para-GNN", "proximity": "Prox-GNN", "structural": "StructGNN"}[mode]
        print(f"Training {mode_name} for epochs {start_epoch+1}-{self.config.epochs}...")
        print(f"  Early stopping on VAL set ({len(val_query_ids)} queries)")

        # Pre-warm corpus embedding cache to avoid per-batch disk I/O during training
        all_corpus_ids = set(val_corpus_ids) | set(test_corpus_ids)
        print(f"Pre-warming corpus embedding cache ({len(all_corpus_ids)} docs)...")
        for doc_id in all_corpus_ids:
            self.para_store.get_corpus_embedding(doc_id)
        print("  Cache ready.")

        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if hasattr(batch_on_device["graph"], "to"):
                    batch_on_device["graph"] = batch_on_device["graph"].to(self.device)

                try:
                    loss = model(
                        query_pos=batch_on_device["query_pos"],
                        candidate_pos=batch_on_device["candidate_pos"],
                        bm25_scores=batch_on_device["bm25_scores"],
                        candidate_relevance_labels=batch_on_device["candidate_relevance_labels"],
                        graph=batch_on_device["graph"],
                        ips_weights=batch_on_device["ips_weights"],
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

            # Evaluate on VAL with alpha grid search (blended score, consistent with final eval)
            gnn_val, avg_alpha = self._get_gnn_scores(model, val_graph)
            gnn_val_debiased = gnn_val - gnn_val.mean(dim=0, keepdim=True)
            best_alpha_v, val_mrr, val_recall, val_hit = self._grid_search_alpha(
                gnn_val, bm25_val_scores.cpu(), val_gold_matrix
            )
            best_alpha_vd, val_mrr_d, val_recall_d, val_hit_d = self._grid_search_alpha(
                gnn_val_debiased, bm25_val_scores.cpu(), val_gold_matrix
            )
            if val_recall_d > val_recall:
                val_mrr, val_recall, val_hit, best_alpha_v = val_mrr_d, val_recall_d, val_hit_d, best_alpha_vd

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_mrr@10": val_mrr,
                "val_recall@10": val_recall,
                "val_hit_rate": val_hit,
                "val_alpha": float(best_alpha_v),
                "learned_alpha": avg_alpha,
            }
            log.append(log_entry)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} val_MRR={val_mrr:.4f} val_R@10={val_recall:.4f} val_Hit={val_hit:.1%} alpha={best_alpha_v:.1f} (learned={avg_alpha:.3f})")

            if val_recall > best_metric:
                best_metric = val_recall
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                print(f"  → New best val Recall@10={val_recall:.4f}, saved model")
            else:
                epochs_without_improvement += 1

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
                print(f"  Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

        # === Post-training: alpha sweep on VAL, then apply frozen to TEST ===
        print(f"\n{'='*60}")
        print(f"  Post-training alpha selection (on VAL) and final evaluation (on TEST)")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(f"{output_dir}/best_model.pt", map_location="cpu"))
        model = model.to(self.device)

        gnn_val, _ = self._get_gnn_scores(model, val_graph)
        gnn_test, _ = self._get_gnn_scores(model, test_graph)

        gnn_val_debiased = gnn_val - gnn_val.mean(dim=0, keepdim=True)
        gnn_test_debiased = gnn_test - gnn_test.mean(dim=0, keepdim=True)

        # Sweep alpha on VAL (original scores)
        print(f"\n  Alpha sweep on VAL (original):")
        print(f"  {'Alpha':<8} {'R@10':<10} {'MRR@10':<10} {'Hit':<10}")
        print(f"  {'-'*38}")
        best_val_alpha, best_val_recall = 0, 0
        for alpha in np.arange(0.0, 1.05, 0.1):
            scores = alpha * gnn_val + (1 - alpha) * bm25_val_scores.cpu()
            mrr, recall, hit = self._compute_metrics(scores, val_gold_matrix)
            marker = " ←" if recall > best_val_recall else ""
            print(f"  {alpha:<8.1f} {recall:<10.4f} {mrr:<10.4f} {hit:<10.1%}{marker}")
            if recall > best_val_recall:
                best_val_recall = recall
                best_val_alpha = alpha

        # Sweep alpha on VAL (debiased scores)
        print(f"\n  Alpha sweep on VAL (debiased):")
        print(f"  {'Alpha':<8} {'R@10':<10} {'MRR@10':<10} {'Hit':<10}")
        print(f"  {'-'*38}")
        best_val_alpha_d, best_val_recall_d = 0, 0
        for alpha in np.arange(0.0, 1.05, 0.1):
            scores = alpha * gnn_val_debiased + (1 - alpha) * bm25_val_scores.cpu()
            mrr, recall, hit = self._compute_metrics(scores, val_gold_matrix)
            marker = " ←" if recall > best_val_recall_d else ""
            print(f"  {alpha:<8.1f} {recall:<10.4f} {mrr:<10.4f} {hit:<10.1%}{marker}")
            if recall > best_val_recall_d:
                best_val_recall_d = recall
                best_val_alpha_d = alpha

        # Decide: use original or debiased based on val performance
        use_debiased = best_val_recall_d > best_val_recall
        chosen_alpha = best_val_alpha_d if use_debiased else best_val_alpha
        chosen_val_recall = max(best_val_recall, best_val_recall_d)
        variant = "debiased" if use_debiased else "original"

        print(f"\n  Val-selected: alpha={chosen_alpha:.1f} ({variant}), val Recall@10={chosen_val_recall:.4f}")

        # Apply frozen alpha to TEST
        if use_debiased:
            test_scores = chosen_alpha * gnn_test_debiased + (1 - chosen_alpha) * bm25_test_scores.cpu()
        else:
            test_scores = chosen_alpha * gnn_test + (1 - chosen_alpha) * bm25_test_scores.cpu()
        test_mrr, test_recall, test_hit = self._compute_metrics(test_scores, test_gold_matrix)

        # Build top-100 rankings for save_predictions
        top_k = min(100, test_scores.shape[1])
        rankings: dict = {}
        pred_scores: dict = {}
        for qi, qid in enumerate(test_query_ids):
            idx = torch.argsort(test_scores[qi], descending=True)[:top_k].tolist()
            rankings[qid] = [test_corpus_ids[i] for i in idx]
            pred_scores[qid] = {test_corpus_ids[i]: float(test_scores[qi, i]) for i in idx}

        print(f"\n  TEST results (alpha={chosen_alpha:.1f} frozen from val):")
        print(f"    Recall@10: {test_recall:.4f}")
        print(f"    MRR@10:    {test_mrr:.4f}")
        print(f"    Hit Rate:  {test_hit:.1%}")

        print(f"\nTraining complete. Test Recall@10: {test_recall:.4f}")
        ground_truth = {qid: list(test_gold[qid].keys()) for qid in test_query_ids if qid in test_gold}
        return test_recall, rankings, ground_truth, pred_scores

    @torch.no_grad()
    def _get_gnn_scores(self, train_model, graph):
        """Get raw GNN scores (before alpha blending)."""
        dim = self.config.embed_dim
        mode = self.config.structure_mode
        struct_input_dim = dim + self.config.act_dim + self.config.pos_dim
        test_model = TestCaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                                  dropout=self.config.dropout, num_head=self.config.num_heads,
                                  structure_mode=mode, struct_input_dim=struct_input_dim,
                                  contranorm_scale=self.config.contranorm_scale,
                                  contranorm_tau=self.config.contranorm_tau)
        test_model.load_state_dict(train_model.state_dict(), strict=False)
        test_model = test_model.to(self.device)
        test_model.eval()

        node_h = graph.ndata["h"]
        edge_h = graph.edata["h"]

        if mode == "structural":
            node_h = test_model.struct_proj(node_h)

        h = test_model.eugat_gnn(graph, node_h, edge_h)
        query_encoded = h[graph.ndata["query_mask"].bool()]
        candidate_encoded = h[graph.ndata["candidate_mask"].bool()]
        alphas = test_model.ffnn(query_encoded)

        scores = torch.matmul(query_encoded, candidate_encoded.T)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        gnn_scores = (scores - mean_scores) / (std_scores + 1e-8)

        return gnn_scores.cpu(), alphas.mean().item()

    def _grid_search_alpha(self, gnn_scores, bm25_scores, gold_matrix):
        """Find optimal alpha via grid search (optimizes Recall@10)."""
        best_alpha, best_mrr, best_recall, best_hit = 0, 0, 0, 0
        for alpha in np.arange(0.0, 1.05, 0.1):
            scores = alpha * gnn_scores + (1 - alpha) * bm25_scores
            mrr, recall, hit_rate = self._compute_metrics(scores, gold_matrix)
            if recall > best_recall:
                best_mrr = mrr
                best_recall = recall
                best_hit = hit_rate
                best_alpha = alpha
        return best_alpha, best_mrr, best_recall, best_hit

    def _compute_metrics(self, scores, gold_matrix, k=10):
        """Compute MRR@k, Recall@k, Hit Rate — fully vectorized."""
        n_queries = gold_matrix.shape[0]
        top_k_idx = torch.argsort(scores, dim=1, descending=True)[:, :k]   # [Q, k]
        top_k_rel = gold_matrix.gather(1, top_k_idx).float()                # [Q, k]

        ranks = torch.arange(1, k + 1, dtype=torch.float32)
        first_hit = (top_k_rel.cumsum(dim=1) <= 1) & top_k_rel.bool()
        mrr = (first_hit.float() / ranks).sum(dim=1).mean().item()

        n_rel = gold_matrix.sum(dim=1).clamp(min=1)
        recall = (top_k_rel.sum(dim=1) / n_rel).mean().item()
        hit_rate = (top_k_rel.sum(dim=1) > 0).float().mean().item()

        return mrr, recall, hit_rate
