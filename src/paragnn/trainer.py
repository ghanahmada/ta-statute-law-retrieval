"""Training loop for Para-GNN.

Adapted from IL-PCSR's train_paragnn_plus_bm25_for_secs.py.
"""
import json
import os
from math import ceil

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from util import set_seed
from .model import CaseGnn, TestCaseGnn
from .graph_builder import ParagraphStore, GraphBuilder


class ParaGNNTrainer:
    """Trains Para-GNN and evaluates on test set."""

    def __init__(self, config, para_store: ParagraphStore):
        self.config = config
        self.para_store = para_store
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
        """
        Args:
            train_dataset: ParaGNNDataset
            collator: ParaGNNCollator
            bm25_test_scores: (num_test_queries, num_corpus) tensor
            test_query_ids: ordered list of test query IDs
            test_corpus_ids: ordered list of corpus doc IDs
            test_gold: {qid: {doc_id: score}} for test evaluation
        """
        set_seed(42)

        output_dir = f"{self.config.output_dir}/{self.config.dataset}"
        os.makedirs(output_dir, exist_ok=True)

        dim = self.config.embed_dim
        model = CaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                        dropout=self.config.dropout, num_head=self.config.num_heads)

        # Resume from checkpoint if exists
        start_epoch = 0
        best_mrr = 0
        log = []
        checkpoint_path = f"{output_dir}/best_model.pt"
        resume_path = f"{output_dir}/resume_checkpoint.pt"

        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_mrr = checkpoint["best_mrr"]
            log = checkpoint.get("log", [])
            print(f"  Resumed at epoch {start_epoch}, best MRR={best_mrr:.4f}")

        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        if os.path.exists(resume_path):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        steps_per_epoch = ceil(len(train_dataset) / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Cosine annealing with warmup: ramps up then decays smoothly
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        # Early stopping
        patience = 10  # stop if no improvement for 10 epochs
        epochs_without_improvement = 0

        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        # Build test graph once
        print("Building test graph...")
        test_graph = GraphBuilder(test_query_ids, test_corpus_ids, self.para_store).graph
        test_graph = test_graph.to(self.device)
        bm25_test_scores = bm25_test_scores.to(self.device)

        # Build gold matrix for evaluation
        gold_matrix = torch.zeros(len(test_query_ids), len(test_corpus_ids))
        doc_id_to_idx = {d: i for i, d in enumerate(test_corpus_ids)}
        for qi, qid in enumerate(test_query_ids):
            if qid in test_gold:
                for did, score in test_gold[qid].items():
                    if did in doc_id_to_idx and score > 0:
                        gold_matrix[qi][doc_id_to_idx[did]] = 1

        print(f"Training Para-GNN for epochs {start_epoch+1}-{self.config.epochs}...")
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
                batch_on_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                # Move graph to device
                if hasattr(batch_on_device["graph"], "to"):
                    batch_on_device["graph"] = batch_on_device["graph"].to(self.device)

                loss = model(
                    query_pos=batch_on_device["query_pos"],
                    candidate_pos=batch_on_device["candidate_pos"],
                    bm25_scores=batch_on_device["bm25_scores"],
                    candidate_relevance_labels=batch_on_device["candidate_relevance_labels"],
                    graph=batch_on_device["graph"],
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Evaluate every epoch
            mrr, recall, hit_rate, avg_alpha = self._evaluate(
                model, bm25_test_scores, test_graph, gold_matrix
            )

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "mrr@10": mrr,
                "recall@10": recall,
                "hit_rate": hit_rate,
                "avg_alpha": avg_alpha,
            }
            log.append(log_entry)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} MRR={mrr:.4f} R@10={recall:.4f} Hit={hit_rate:.1%} alpha={avg_alpha:.3f}")

            if mrr > best_mrr:
                best_mrr = mrr
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
                print(f"  → New best MRR={mrr:.4f}, saved model")
            else:
                epochs_without_improvement += 1

            # Save resume checkpoint
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mrr": best_mrr,
                "log": log,
            }, f"{output_dir}/resume_checkpoint.pt")

            with open(f"{output_dir}/training_log.json", "w") as f:
                json.dump(log, f, indent=2)

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        print(f"\nTraining complete. Best MRR@10: {best_mrr:.4f}")
        return best_mrr

    @torch.no_grad()
    def _evaluate(self, train_model, bm25_scores, test_graph, gold_matrix):
        """Run TestCaseGnn and compute MRR@10."""
        dim = self.config.embed_dim
        test_model = TestCaseGnn(in_dim=dim, h_dim=dim, out_dim=dim,
                                  dropout=self.config.dropout, num_head=self.config.num_heads)
        test_model.load_state_dict(train_model.state_dict(), strict=False)
        test_model = test_model.to(self.device)
        test_model.eval()

        scores, alphas = test_model(bm25_scores, test_graph)
        scores = scores.cpu()
        alphas = alphas.cpu()

        # Compute MRR@10, Recall@10, Hit Rate
        k = 10
        mrr_sum = 0
        recall_sum = 0
        hit_count = 0
        n_queries = gold_matrix.shape[0]

        for qi in range(n_queries):
            relevant = gold_matrix[qi].nonzero(as_tuple=True)[0].tolist()
            if not relevant:
                continue

            ranked = torch.argsort(scores[qi], descending=True)[:k].tolist()

            # MRR
            for rank, idx in enumerate(ranked):
                if idx in relevant:
                    mrr_sum += 1.0 / (rank + 1)
                    break

            # Recall@k
            hits = len(set(ranked) & set(relevant))
            recall_sum += hits / len(relevant)

            # Hit rate
            if hits > 0:
                hit_count += 1

        mrr = mrr_sum / n_queries
        recall = recall_sum / n_queries
        hit_rate = hit_count / n_queries
        avg_alpha = alphas.mean().item()

        return mrr, recall, hit_rate, avg_alpha
