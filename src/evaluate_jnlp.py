from jnlp import Config, PipelineOrchestrator

config = Config()
pipeline = PipelineOrchestrator(config)

metrics = pipeline.evaluate_stage1_only(
    verbose=True,
    use_reranker=True,  
    use_test_split=True  
)