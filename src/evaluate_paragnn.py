"""Evaluate Para-GNN on statute retrieval datasets.

Usage:
  # Phase 1: Pre-compute BM25 + embeddings (needs GPU for BGE-M3)
  python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted

  # Phase 2: Train and evaluate Para-GNN
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --method adapted

  # Method 1 (full): requires LLM-labeled RR (run label_rhetorical_roles.py first)
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --method full
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paragnn import DATASETS, ParaGNNConfig
from paragnn.graph_builder import ParagraphStore
from paragnn.dataset import ParaGNNDataset, ParaGNNCollator
from paragnn.trainer import ParaGNNTrainer
from util.dataloader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Para-GNN for statute retrieval")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--method", default="adapted", choices=["full", "adapted"])
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=299)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--proximity_radius", type=int, default=0,
                        help="Connect statute nodes within N articles (0=disabled, 50=recommended)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, cfg in datasets.items():
        prox_label = f", prox={args.proximity_radius}" if args.proximity_radius > 0 else ""
        print(f"\n{'='*60}")
        print(f"  Para-GNN ({args.method}{prox_label}): {name}")
        print(f"{'='*60}")

        config = ParaGNNConfig(
            dataset=name,
            data_path=cfg["path"],
            lang=cfg["lang"],
            method=args.method,
            max_relevant=args.max_relevant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
            learning_rate=args.lr,
            proximity_radius=args.proximity_radius,
        )
        output_dir = f"{config.output_dir}/{name}"

        # Check pre-computed files exist
        required = [
            f"{output_dir}/bm25_train_scores.pt",
            f"{output_dir}/bm25_test_scores.pt",
            f"{output_dir}/train_query_ids.json",
            f"{output_dir}/test_query_ids.json",
            f"{output_dir}/corpus_doc_ids.json",
            f"{output_dir}/bm25_hard_negatives.json",
            f"{output_dir}/query_paragraphs.json",
            f"{output_dir}/corpus_paragraphs.json",
            f"{output_dir}/embeddings/EMBD_CONST.pt",
        ]
        missing = [f for f in required if not Path(f).exists()]
        if missing:
            print(f"Missing pre-computed files:")
            for f in missing:
                print(f"  {f}")
            print(f"\nRun first: python src/paragnn/precompute.py --dataset {name} --method {args.method}")
            continue

        # Load pre-computed data
        print("Loading pre-computed data...")
        bm25_train_scores = torch.load(f"{output_dir}/bm25_train_scores.pt")
        bm25_test_scores = torch.load(f"{output_dir}/bm25_test_scores.pt")
        with open(f"{output_dir}/train_query_ids.json") as f:
            train_qids = json.load(f)
        with open(f"{output_dir}/test_query_ids.json") as f:
            test_qids = json.load(f)
        with open(f"{output_dir}/corpus_doc_ids.json") as f:
            corpus_doc_ids = json.load(f)
        with open(f"{output_dir}/bm25_hard_negatives.json") as f:
            hard_negatives = json.load(f)

        # Load qrels
        train_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_train.tsv",
        ).load()
        test_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_test.tsv",
        ).load()
        if config.max_relevant > 0:
            train_loader.filter_max_relevant(config.max_relevant)
            test_loader.filter_max_relevant(config.max_relevant)

        # RR labels path (Method 1 only)
        rr_labels_path = None
        if args.method == "full":
            rr_labels_path = f"{output_dir}/rr_labels.json"
            if not Path(rr_labels_path).exists():
                print(f"Missing RR labels: {rr_labels_path}")
                print(f"Run first: python experiment/label_rhetorical_roles.py --dataset {name}")
                continue

        # Initialize paragraph store
        print("Loading paragraph store...")
        para_store = ParagraphStore(
            output_dir=output_dir,
            method=args.method,
            rr_labels_path=rr_labels_path,
        )

        # Create training dataset
        print("Creating training dataset...")
        train_dataset = ParaGNNDataset(
            train_qids=train_qids,
            qrels=train_loader.qrels,
            corpus_doc_ids=corpus_doc_ids,
            hard_negatives=hard_negatives,
            bm25_scores=bm25_train_scores,
            num_negatives=config.num_negatives,
        )

        collator = ParaGNNCollator(para_store=para_store,
                                    proximity_radius=config.proximity_radius)

        # Train
        trainer = ParaGNNTrainer(config=config, para_store=para_store)
        best_mrr = trainer.train(
            train_dataset=train_dataset,
            collator=collator,
            bm25_test_scores=bm25_test_scores,
            test_query_ids=test_qids,
            test_corpus_ids=corpus_doc_ids,
            test_gold=test_loader.qrels,
        )

        print(f"\n  Final best MRR@10: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
