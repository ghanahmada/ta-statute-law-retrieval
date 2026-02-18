import argparse

from jnlp import Config, PipelineOrchestrator

parser = argparse.ArgumentParser(description="JNLP Stage 1 evaluation")
parser.add_argument("--feature_type", type=str, default="product",
                    choices=["histogram", "product"],
                    help="CatBoost feature type: histogram (paper) or product (element-wise q*d)")
parser.add_argument("--reranker", action="store_true", help="Enable re-ranker (slow)")
args = parser.parse_args()

config = Config(stage1_feature_type=args.feature_type)
pipeline = PipelineOrchestrator(config)

metrics = pipeline.evaluate_stage1_only(
    verbose=True,
    use_reranker=args.reranker,
    use_test_split=True
)
