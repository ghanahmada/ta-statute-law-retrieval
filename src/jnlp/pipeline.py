import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any

from . import Config
from . import seed_everything
from util.dataloader import DataLoader
from util.metrics import evaluate_ranking
from .stage1_retriever import Stage1Retriever
from .stage2_finetuner import Stage2FineTuner
from .stage3_ensemble import Stage3Ensemble

class PipelineOrchestrator:
    def __init__(self, config: "Config"):
        self.config = config
        
        self.data_loader = None  # General data loader (corpus + all queries)
        self.train_data_loader = None  # Train qrels only
        self.test_data_loader = None  # Test qrels only
        
        self.stage1 = None
        self.stage2 = None
        self.ensemble = None
        
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.stage1_results: Dict[str, List[Tuple[str, float]]] = {}
    
    def load_data(self, mode: str = "all") -> "PipelineOrchestrator":
        """
        Load dataset.
        
        Args:
            mode: "all" for full dataset, "train" for train split, "test" for test split
        """
        self.data_loader = DataLoader(
            corpus_path=self.config.corpus_path,
            queries_path=self.config.queries_path,
            qrels_path=self.config.qrels_path
        ).load()
        
        return self
    
    def load_train_data(self) -> "PipelineOrchestrator":
        self.train_data_loader = DataLoader(
            corpus_path=self.config.corpus_path,
            queries_path=self.config.queries_path,
            qrels_path=self.config.qrels_train_path
        ).load()
        
        return self
    
    def load_test_data(self) -> "PipelineOrchestrator":
        self.test_data_loader = DataLoader(
            corpus_path=self.config.corpus_path,
            queries_path=self.config.queries_path,
            qrels_path=self.config.qrels_test_path
        ).load()
        
        return self
    
    def run_stage1_training(
        self,
        encode_batch_size: int = 64,
        max_length: int = 1024,
        classifier_iterations: int = 1000,
        verbose: bool = True,
        use_train_split: bool = True
    ) -> "PipelineOrchestrator":
        """
        Train Stage 1: BGE-M3 encoding + CatBoost classifier.
        
        Args:
            use_train_split: If True, uses train qrels only for training.
                           If False, uses all qrels (legacy behavior).
        """
        if use_train_split:
            if self.train_data_loader is None:
                self.load_train_data()
            training_loader = self.train_data_loader
            if verbose:
                print(f"Training on TRAIN split: {len(training_loader.qrels)} queries with qrels")
        else:
            if self.data_loader is None:
                self.load_data()
            training_loader = self.data_loader
            if verbose:
                print(f"Training on ALL data: {len(training_loader.qrels)} queries with qrels")
        
        self.stage1 = Stage1Retriever(
            bge_model_name=self.config.bge_model_name,
            feature_type=self.config.stage1_feature_type,
            n_bins=self.config.n_histogram_bins,
            oversample_ratio=self.config.stage1_oversample_ratio
        )

        if verbose:
            print(f"Feature type: {self.config.stage1_feature_type}")
            print("Encoding corpus...")
        doc_ids, doc_texts = training_loader.get_corpus_texts()
        self.stage1.encode_corpus(doc_ids, doc_texts, batch_size=encode_batch_size, max_length=max_length)

        if verbose:
            print("Encoding training queries...")
        train_query_ids = list(training_loader.qrels.keys())
        train_query_texts = [training_loader.queries[qid]["text"] for qid in train_query_ids]
        query_embs = self.stage1.encode_bge(train_query_texts, batch_size=encode_batch_size, max_length=max_length)
        train_query_embeddings = {qid: emb for qid, emb in zip(train_query_ids, query_embs)}
        self.query_embeddings.update(train_query_embeddings)

        # Calibrate L1 histogram range (only needed for histogram feature type)
        if self.config.stage1_feature_type == "histogram":
            if verbose:
                print("Calibrating L1 histogram range...")
            self.stage1.calibrate_l1_range(train_query_embeddings)

        # Compute BM25 hard negatives (adjustment from paper's random negatives)
        if verbose:
            print("Computing BM25 hard negatives...")
        from util.bm25 import BM25
        bm25 = BM25()
        bm25.fit(doc_texts)
        hard_negatives = {}
        for qid in train_query_ids:
            scores = bm25.transform(training_loader.queries[qid]["text"])
            top_indices = np.argsort(scores)[::-1][:100]
            hard_negatives[qid] = [doc_ids[i] for i in top_indices]

        if verbose:
            print("Preparing training data...")
        X, y = self.stage1.prepare_training_data(
            training_loader, train_query_embeddings, hard_negatives=hard_negatives
        )
        
        if verbose:
            print("Training CatBoost classifier...")
        self.stage1.train_classifier(X, y, iterations=classifier_iterations)
        
        self.stage1.save(f"{self.config.output_dir}/stage1")
        if verbose:
            print(f"Stage 1 model saved to {self.config.output_dir}/stage1")
        
        return self
    
    def run_stage1_inference(
        self,
        load_reranker: bool = True
    ) -> "PipelineOrchestrator":
        if self.stage1 is None:
            self.stage1 = Stage1Retriever().load(f"{self.config.output_dir}/stage1")
            self.stage1.load_bge_model()
        
        if load_reranker:
            self.stage1.load_reranker(
                reranker_type=self.config.reranker_type,
                model_name=self.config.bge_reranker_name if self.config.reranker_type == "bge" else self.config.rankllama_name
            )
        
        for qid, query_data in tqdm(self.data_loader.queries.items(), desc="Stage 1 Inference"):
            if qid not in self.query_embeddings:
                emb = self.stage1.encode_bge([query_data["text"]])[0]
                self.query_embeddings[qid] = emb
            
            results = self.stage1.retrieve(
                query_text=query_data["text"],
                query_embedding=self.query_embeddings[qid],
                corpus=self.data_loader.corpus,
                stage1_topk=self.config.stage1_topk,
                rerank_topk=self.config.stage1_rerank_topk
            )
            self.stage1_results[qid] = results
        
        return self
    
    def run_stage2_training(self) -> "PipelineOrchestrator":
        self.stage2 = Stage2FineTuner(
            model_name=self.config.llm_model_name,
            max_seq_length=self.config.max_seq_length,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            use_4bit=self.config.use_4bit,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
        )

        self.stage2.setup_model()
        
        train_dataset = self.stage2.prepare_data(
            self.data_loader,
            self.stage1_results,
            upsample_positive=self.config.stage2_upsample_ratio
        )
        
        self.stage2.train(
            train_dataset=train_dataset,
            output_dir=f"{self.config.output_dir}/stage2",
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            warmup_ratio=self.config.warmup_ratio
        )
        
        return self
    
    def run_full_pipeline(self) -> Dict[str, List[str]]:
        seed_everything(self.config.seed)
        
        self.load_data()
        
        self.run_stage1_training()
        self.run_stage1_inference()
        
        self.run_stage2_training()
        
        predictions = {}
        for qid, candidates in self.stage1_results.items():
            query_text = self.data_loader.queries[qid]["text"]
            articles = [(did, self.data_loader.corpus[did]["text"]) for did, _ in candidates]
            
            preds = self.stage2.predict(query_text, articles)
            predictions[qid] = [aid for aid, score in preds if score >= 0.5]
        
        return predictions
    
    def evaluate_predictions(
        self,
        predictions: Dict[str, List[str]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        ground_truth = {
            qid: list(docs.keys())
            for qid, docs in self.data_loader.qrels.items()
        }
        
        ranking_metrics = evaluate_ranking(predictions, ground_truth, top_k=top_k)
        
        ensemble = Stage3Ensemble()
        classification_metrics = ensemble.evaluate(predictions, ground_truth)
        
        return {**ranking_metrics, **classification_metrics}
    
    def evaluate_trained_model(
        self,
        stage1_path: str = None,
        stage2_path: str = None,
        top_k: int = 10,
        use_reranker: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate already-trained models (minimal code for evaluation).
        
        Args:
            stage1_path: Path to Stage 1 artifacts (default: {output_dir}/stage1)
            stage2_path: Path to Stage 2 adapter (default: {output_dir}/stage2)
            top_k: Cutoff for ranking metrics
            use_reranker: Whether to use re-ranker in Stage 1
            verbose: Print detailed results
            
        Returns:
            Dict with all evaluation metrics
        """
        seed_everything(self.config.seed)
        stage1_path = stage1_path or f"{self.config.output_dir}/stage1"
        stage2_path = stage2_path or f"{self.config.output_dir}/stage2"
        
        if verbose:
            print("=" * 60)
            print("JNLP Pipeline Evaluation")
            print("=" * 60)
        
        if self.data_loader is None:
            self.load_data()
            if verbose:
                print(f"Loaded {len(self.data_loader.corpus)} documents, {len(self.data_loader.queries)} queries")
        
        if verbose:
            print("\nLoading Stage 1 (BGE-M3 + CatBoost)...")
        self.stage1 = Stage1Retriever(
            bge_model_name=self.config.bge_model_name,
            feature_type=self.config.stage1_feature_type,
            n_bins=self.config.n_histogram_bins
        ).load(stage1_path)
        self.stage1.load_bge_model()
        
        if use_reranker:
            self.stage1.load_reranker(
                reranker_type=self.config.reranker_type,
                model_name=self.config.bge_reranker_name if self.config.reranker_type == "bge" else self.config.rankllama_name
            )
        
        if verbose:
            print(f"Loading Stage 2 (QLoRA from {stage2_path})...")
        self.stage2 = Stage2FineTuner(
            model_name=self.config.llm_model_name,
            max_seq_length=self.config.max_seq_length,
            use_4bit=self.config.use_4bit,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
        )
        self.stage2.load_adapter(stage2_path, self.config.llm_model_name)
        
        if verbose:
            print("\nRunning inference...")
        
        stage1_rankings = {}  
        final_rankings = {}   
        
        iterator = self.data_loader.queries.items()
        if verbose:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for qid, query_data in iterator:
            query_text = query_data["text"]
            
            query_emb = self.stage1.encode_bge([query_text])[0]
            
            stage1_results = self.stage1.retrieve(
                query_text=query_text,
                query_embedding=query_emb,
                corpus=self.data_loader.corpus,
                stage1_topk=self.config.stage1_topk,
                rerank_topk=self.config.stage1_rerank_topk
            )
            stage1_rankings[qid] = [doc_id for doc_id, _ in stage1_results]
            
            articles = [(did, self.data_loader.corpus[did]["text"]) for did, _ in stage1_results]
            stage2_scores = self.stage2.predict(query_text, articles)
            
            stage2_scores.sort(key=lambda x: x[1], reverse=True)
            final_rankings[qid] = [doc_id for doc_id, _ in stage2_scores]
        
        ground_truth = {
            qid: list(docs.keys())
            for qid, docs in self.data_loader.qrels.items()
        }
        
        stage1_metrics = evaluate_ranking(stage1_rankings, ground_truth, top_k=top_k)
        final_metrics = evaluate_ranking(final_rankings, ground_truth, top_k=top_k)
        
        final_predictions = {
            qid: [did for did, score in self.stage2.predict(
                self.data_loader.queries[qid]["text"],
                [(did, self.data_loader.corpus[did]["text"]) for did in final_rankings[qid]]
            ) if score >= 0.5]
            for qid in final_rankings
        }
        classification_metrics = Stage3Ensemble().evaluate(final_predictions, ground_truth)
        
        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"\nStage 1 (BGE-M3 + CatBoost + Re-ranker):")
            print(f"  MRR@{top_k}: {stage1_metrics[f'mrr@{top_k}']:.4f}")
            print(f"  Recall@{top_k}: {stage1_metrics[f'recall@{top_k}']:.4f}")
            print(f"  Hit Rate: {stage1_metrics['hit_rate']:.2%}")
            print(f"\nFull Pipeline (Stage 1 + Stage 2 QLoRA):")
            print(f"  MRR@{top_k}: {final_metrics[f'mrr@{top_k}']:.4f}")
            print(f"  Recall@{top_k}: {final_metrics[f'recall@{top_k}']:.4f}")
            print(f"  Precision@{top_k}: {final_metrics[f'precision@{top_k}']:.4f}")
            print(f"  F2: {classification_metrics['f2.0']:.4f}")
            print(f"  Hit Rate: {final_metrics['hit_rate']:.2%}")
            print("-" * 60)
        
        return {
            "stage1": stage1_metrics,
            "full_pipeline": {**final_metrics, **classification_metrics}
        }
    
    def evaluate_stage1_only(
        self,
        stage1_path: str = None,
        top_k: int = 10,
        use_reranker: bool = False,  # Default OFF for speed
        verbose: bool = True,
        train_if_missing: bool = True,
        use_test_split: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate Stage 1 only (faster, no LLM required).
        
        Args:
            stage1_path: Path to Stage 1 artifacts
            top_k: Cutoff for ranking metrics
            use_reranker: Whether to use re-ranker (slower, ~1-2 min per query)
            verbose: Print detailed results
            train_if_missing: If True, train Stage 1 if no saved model found
            use_test_split: If True, evaluate only on TEST queries (proper evaluation).
                          If False, evaluate on all queries (legacy behavior).
        """        
        seed_everything(self.config.seed)
        stage1_path = stage1_path or f"{self.config.output_dir}/stage1"
        
        if verbose:
            print("=" * 60)
            print("Stage 1 Evaluation (BGE-M3 + CatBoost)")
            if use_reranker:
                print(f"  + Re-ranker: {self.config.reranker_type}")
            else:
                print("  (Re-ranker disabled for speed)")
            if use_test_split:
                print("  Evaluating on TEST split (unseen queries)")
            else:
                print("  Evaluating on ALL queries (may include train)")
            print("=" * 60)
        
        if self.data_loader is None:
            self.load_data()
        
        if use_test_split:
            if self.test_data_loader is None:
                self.load_test_data()
            eval_loader = self.test_data_loader
        else:
            eval_loader = self.data_loader
        
        if verbose:
            print(f"Corpus: {len(self.data_loader.corpus)} documents")
            print(f"Evaluating: {len(eval_loader.qrels)} queries")
        
        model_exists = Path(stage1_path).exists() and (Path(stage1_path) / "catboost_model.cbm").exists()
        
        if not model_exists and train_if_missing:
            if verbose:
                print("\nNo trained model found. Training Stage 1...")
            self.run_stage1_training(use_train_split=use_test_split)
        elif not model_exists:
            raise ValueError(f"No trained model at {stage1_path}. Set train_if_missing=True to train.")
        else:
            self.stage1 = Stage1Retriever(
                bge_model_name=self.config.bge_model_name,
                feature_type=self.config.stage1_feature_type,
                n_bins=self.config.n_histogram_bins
            ).load(stage1_path)
            self.stage1.load_bge_model()
        
        if use_reranker:
            self.stage1.load_reranker(
                reranker_type=self.config.reranker_type,
                model_name=self.config.bge_reranker_name if self.config.reranker_type == "bge" else self.config.rankllama_name
            )
        
        test_query_ids = list(eval_loader.qrels.keys())
        test_query_texts = [self.data_loader.queries[qid]["text"] for qid in test_query_ids]
        
        if verbose:
            print(f"Encoding {len(test_query_ids)} test queries...")
        query_cache_path = f"{self.config.output_dir}/stage1/query_embeddings.npy"
        query_embeddings = self.stage1.encode_queries_batch(
            test_query_ids, test_query_texts, cache_path=query_cache_path
        )
        
        queries_batch = {
            qid: (self.data_loader.queries[qid]["text"], query_embeddings[qid])
            for qid in test_query_ids
        }
        
        if verbose:
            print("Running retrieval...")
        results = self.stage1.retrieve_batch(
            queries_batch,
            self.data_loader.corpus,
            stage1_topk=self.config.stage1_topk,
            rerank_topk=self.config.stage1_rerank_topk,
            use_reranker=use_reranker
        )
        
        rankings = {
            qid: [doc_id for doc_id, _ in docs]
            for qid, docs in results.items()
        }
        
        ground_truth = {
            qid: list(docs.keys())
            for qid, docs in eval_loader.qrels.items()
        }
        metrics = evaluate_ranking(rankings, ground_truth, top_k=top_k)
        
        if verbose:
            split_name = "TEST" if use_test_split else "ALL"
            print("\n" + "=" * 60)
            print(f"RESULTS ({split_name} split)")
            print("=" * 60)
            print(f"MRR@{top_k}: {metrics[f'mrr@{top_k}']:.4f}")
            print(f"Recall@{top_k}: {metrics[f'recall@{top_k}']:.4f}")
            print(f"Precision@{top_k}: {metrics[f'precision@{top_k}']:.4f}")
            print(f"Queries with hits: {int(metrics['hit_rate'] * metrics['n_queries'])}/{metrics['n_queries']} ({metrics['hit_rate']:.1%})")
            print("-" * 60)
        
        return metrics
