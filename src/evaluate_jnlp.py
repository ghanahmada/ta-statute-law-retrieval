import argparse

from jnlp import Config, PipelineOrchestrator

DATASETS = {
    "kuhperdata": {"path": "data/kuhperdata", "max_length": 1024, "batch_size": 64},
    "bsard": {"path": "data/bsard", "max_length": 1024, "batch_size": 64},
    "ilpcsr": {"path": "data/ilpcsr", "max_length": 8192, "batch_size": 8},
    "stard": {"path": "data/stard", "max_length": 1024, "batch_size": 64},
}

parser = argparse.ArgumentParser(description="JNLP Stage 1 evaluation")
parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
parser.add_argument("--feature_type", type=str, default="product",
                    choices=["histogram", "product"],
                    help="CatBoost feature type: histogram (paper) or product (element-wise q*d)")
parser.add_argument("--reranker", action="store_true", help="Enable re-ranker (slow)")
args = parser.parse_args()

datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

for name, cfg in datasets.items():
    data_dir = cfg["path"]

    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} — JNLP Stage 1 ({args.feature_type}, max_len={cfg['max_length']})")
    print(f"{'=' * 60}")

    config = Config(
        corpus_path=f"{data_dir}/corpus.jsonl",
        queries_path=f"{data_dir}/queries.jsonl",
        qrels_path=f"{data_dir}/qrels_train.tsv",
        qrels_train_path=f"{data_dir}/qrels_train.tsv",
        qrels_test_path=f"{data_dir}/qrels_test.tsv",
        stage1_feature_type=args.feature_type,
        encode_max_length=cfg["max_length"],
        encode_batch_size=cfg["batch_size"],
        output_dir=f"outputs/jnlp/{name}",
    )
    pipeline = PipelineOrchestrator(config)

    metrics = pipeline.evaluate_stage1_only(
        verbose=True,
        use_reranker=args.reranker,
        use_test_split=True,
    )
