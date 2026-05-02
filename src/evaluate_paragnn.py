"""Evaluate Para-GNN / Prox-GNN / StructGNN on statute retrieval datasets.

Usage:
  # Phase 1: Pre-compute BM25 + embeddings (needs GPU for BGE-M3)
  python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted

  # Phase 2: Train and evaluate
  # Para-GNN (base):
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode none

  # Prox-GNN (proximity edges, KUHPerdata only):
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode proximity

  # StructGNN (structural node features, all datasets):
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural

  # Any mode + fact-type query edges (requires annotate_subsumption.py + precompute --encode_fact_types):
  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural --use_fact_types
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
from paragnn.structure import precompute_structure_features, get_query_structure_features
from util.dataloader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Para-GNN / Prox-GNN / StructGNN")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--method", default="adapted", choices=["full", "adapted"])
    parser.add_argument("--structure_mode", default="none", choices=["none", "proximity", "structural"])
    parser.add_argument("--proximity_radius", type=int, default=50)
    parser.add_argument("--act_dim", type=int, default=64)
    parser.add_argument("--pos_dim", type=int, default=32)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=299)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_fact_types", action="store_true",
                        help="Use fact-type edge features for query paragraphs (requires precompute --encode_fact_types)")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    ft_tag = " + FactTypes" if args.use_fact_types else ""
    mode_labels = {
        "none": f"Para-GNN{ft_tag}",
        "proximity": f"Prox-GNN (r={args.proximity_radius}){ft_tag}",
        "structural": f"StructGNN (act={args.act_dim}d, pos={args.pos_dim}d){ft_tag}",
    }

    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {mode_labels[args.structure_mode]} [{args.method}]: {name}")
        print(f"{'='*60}")

        config = ParaGNNConfig(
            dataset=name,
            data_path=cfg["path"],
            lang=cfg["lang"],
            method=args.method,
            structure_mode=args.structure_mode,
            proximity_radius=args.proximity_radius,
            act_dim=args.act_dim,
            pos_dim=args.pos_dim,
            max_relevant=args.max_relevant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
            learning_rate=args.lr,
        )
        output_dir = f"{config.output_dir}/{name}"

        required = [
            f"{output_dir}/bm25_train_scores.pt",
            f"{output_dir}/bm25_val_scores.pt",
            f"{output_dir}/bm25_test_scores.pt",
            f"{output_dir}/train_query_ids.json",
            f"{output_dir}/val_query_ids.json",
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

        print("Loading pre-computed data...")
        bm25_train_scores = torch.load(f"{output_dir}/bm25_train_scores.pt")
        bm25_val_scores = torch.load(f"{output_dir}/bm25_val_scores.pt")
        bm25_test_scores = torch.load(f"{output_dir}/bm25_test_scores.pt")
        with open(f"{output_dir}/train_query_ids.json") as f:
            train_qids = json.load(f)
        with open(f"{output_dir}/val_query_ids.json") as f:
            val_qids = json.load(f)
        with open(f"{output_dir}/test_query_ids.json") as f:
            test_qids = json.load(f)
        with open(f"{output_dir}/corpus_doc_ids.json") as f:
            corpus_doc_ids = json.load(f)
        with open(f"{output_dir}/bm25_hard_negatives.json") as f:
            hard_negatives = json.load(f)

        train_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_train.tsv",
        ).load()
        val_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_val.tsv",
        ).load()
        test_loader = DataLoader(
            f"{config.data_path}/corpus.jsonl",
            f"{config.data_path}/queries.jsonl",
            f"{config.data_path}/qrels_test.tsv",
        ).load()
        if config.max_relevant > 0:
            train_loader.filter_max_relevant(config.max_relevant)
            val_loader.filter_max_relevant(config.max_relevant)
            test_loader.filter_max_relevant(config.max_relevant)

        rr_labels_path = None
        if args.method == "full":
            rr_labels_path = f"{output_dir}/rr_labels.json"
            if not Path(rr_labels_path).exists():
                print(f"Missing RR labels: {rr_labels_path}")
                print(f"Run first: python experiment/label_rhetorical_roles.py --dataset {name}")
                continue

        print("Loading paragraph store...")
        para_store = ParagraphStore(
            output_dir=output_dir,
            method=args.method,
            rr_labels_path=rr_labels_path,
            use_fact_types=args.use_fact_types,
        )

        # Precompute structure features for StructGNN
        structure_features = None
        query_structure_feature = None
        if args.structure_mode == "structural":
            corpus_path = f"{config.data_path}/corpus.jsonl"
            print(f"Computing structure features (act_dim={args.act_dim}, pos_dim={args.pos_dim})...")
            structure_features = precompute_structure_features(
                corpus_path, name, act_dim=args.act_dim, pos_dim=args.pos_dim
            )
            query_act, query_pos = get_query_structure_features(
                act_dim=args.act_dim, pos_dim=args.pos_dim
            )
            query_structure_feature = torch.cat([query_act, query_pos])
            print(f"  {len(structure_features)} corpus docs, feature dim={args.act_dim + args.pos_dim}")

        print("Creating training dataset...")
        train_dataset = ParaGNNDataset(
            train_qids=train_qids,
            qrels=train_loader.qrels,
            corpus_doc_ids=corpus_doc_ids,
            hard_negatives=hard_negatives,
            bm25_scores=bm25_train_scores,
            num_negatives=config.num_negatives,
        )

        collator = ParaGNNCollator(
            para_store=para_store,
            structure_mode=args.structure_mode,
            proximity_radius=config.proximity_radius,
            structure_features=structure_features,
            query_structure_feature=query_structure_feature,
        )

        trainer = ParaGNNTrainer(
            config=config,
            para_store=para_store,
            structure_features=structure_features,
            query_structure_feature=query_structure_feature,
            use_fact_types=args.use_fact_types,
        )
        best_mrr = trainer.train(
            train_dataset=train_dataset,
            collator=collator,
            bm25_val_scores=bm25_val_scores,
            val_query_ids=val_qids,
            val_corpus_ids=corpus_doc_ids,
            val_gold=val_loader.qrels,
            bm25_test_scores=bm25_test_scores,
            test_query_ids=test_qids,
            test_corpus_ids=corpus_doc_ids,
            test_gold=test_loader.qrels,
        )

        print(f"\n  Final best MRR@10: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
