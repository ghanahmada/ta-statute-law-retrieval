import argparse

from jnlp import Config, PipelineOrchestrator
from util.metrics import save_predictions

DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "max_length": 1024, "batch_size": 64},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "max_length": 1024, "batch_size": 64},
    "kuhperdata-exp": {"path": "data/kuhperdata-exp", "max_length": 1024, "batch_size": 64},
    "kuhperdata-summ-exp": {"path": "data/kuhperdata-summ-exp", "max_length": 1024, "batch_size": 16},
    "bsard": {"path": "data/bsard", "max_length": 1024, "batch_size": 16},
    "ilpcsr": {"path": "data/ilpcsr", "max_length": 8192, "batch_size": 8},
    "stard": {"path": "data/stard", "max_length": 1024, "batch_size": 64},
}

LLM_MODELS = {
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}

parser = argparse.ArgumentParser(description="JNLP pipeline evaluation")
parser.add_argument("--dataset", type=str, default="kuhperdata", choices=[*DATASETS, "all"])
parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
parser.add_argument("--feature_type", type=str, default="product", choices=["histogram", "product"])
parser.add_argument("--llm_model", type=str, default="qwen2.5", choices=LLM_MODELS.keys())
parser.add_argument("--reranker", action="store_true", help="Enable re-ranker (slow)")
parser.add_argument("--max_relevant", type=int, default=5,
                    help="Max ground-truth docs per query (queries with more are excluded)")
parser.add_argument("--max_length", type=int, default=None,
                    help="Override encode max length (default: per-dataset value)")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Override encode batch size (default: per-dataset value)")
parser.add_argument("--save_predictions", type=str, default=None,
                    help="Path to save per-query top-100 predictions as JSONL")
args = parser.parse_args()

datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

for name, cfg in datasets.items():
    data_dir = cfg["path"]
    llm_name = LLM_MODELS[args.llm_model]

    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} — JNLP Stage {args.stage} ({args.feature_type})")
    if args.stage == 2:
        print(f"  LLM: {llm_name}")
    print(f"{'=' * 60}")

    # Stage 1 always in base dir; Stage 2 in LLM-specific subdir
    base_dir = f"outputs/jnlp/{name}"

    encode_max_length = args.max_length if args.max_length is not None else cfg["max_length"]
    encode_batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]

    config = Config(
        corpus_path=f"{data_dir}/corpus.jsonl",
        queries_path=f"{data_dir}/queries.jsonl",
        qrels_path=f"{data_dir}/qrels_train.tsv",
        qrels_train_path=f"{data_dir}/qrels_train.tsv",
        qrels_test_path=f"{data_dir}/qrels_test.tsv",
        stage1_feature_type=args.feature_type,
        encode_max_length=encode_max_length,
        encode_batch_size=encode_batch_size,
        llm_model_name=llm_name,
        max_relevant=args.max_relevant,
        output_dir=base_dir,
    )
    pipeline = PipelineOrchestrator(config)

    if args.stage == 1:
        metrics = pipeline.evaluate_stage1_only(
            verbose=True,
            use_reranker=args.reranker,
            use_test_split=True,
        )
    else:
        metrics = pipeline.evaluate_stage2_only(
            verbose=True,
            use_reranker=args.reranker,
        )

    if "_rankings" in metrics:
        save_predictions(
            metrics["_rankings"], metrics["_ground_truth"],
            args.save_predictions.format(dataset=name) if args.save_predictions else None,
            method=f"jnlp_stage{args.stage}", dataset=name, scores=metrics.get("_scores"),
        )
