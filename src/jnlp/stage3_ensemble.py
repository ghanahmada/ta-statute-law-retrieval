import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from tqdm import tqdm


class Stage3Ensemble:
    """
    Paper Section 4.3: Weighted sum ensemble mechanism.
    Optimizes weights using Optuna to maximize F2-score (prioritizes recall).
    """
    
    def __init__(
        self,
        models: List[Any] = None,
        weights: List[float] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            models: List of Stage2FineTuner instances (optional, can add later)
            weights: Initial weights for ensemble (default: uniform)
            threshold: Classification threshold
        """
        self.models = models or []
        self.weights = weights
        self.threshold = threshold
        self.optimized_weights = None
        self.optimized_threshold = None
    
    def add_model(self, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0]
        else:
            self.weights.append(weight)
        return self
    
    def _normalize_weights(self, weights: List[float]) -> np.ndarray:
        """Normalize weights to sum to 1."""
        weights = np.array(weights)
        return weights / weights.sum()
    
    def predict_scores(
        self,
        query: str,
        articles: List[Tuple[str, str]],
        weights: List[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get ensemble scores for query-article pairs.
        
        Args:
            query: Query text
            articles: List of (article_id, article_text) tuples
            weights: Custom weights (uses self.weights if None)
            
        Returns:
            List of (article_id, ensemble_score) tuples
        """
        if not self.models:
            raise ValueError("No models added to ensemble.")
        
        weights = weights or self.optimized_weights or self.weights
        weights = self._normalize_weights(weights)
        
        # Collect predictions from all models
        all_predictions = []
        for model in self.models:
            preds = model.predict(query, articles)
            pred_dict = {aid: score for aid, score in preds}
            all_predictions.append(pred_dict)
        
        # Weighted average
        results = []
        for aid, _ in articles:
            weighted_score = sum(
                w * preds.get(aid, 0.0)
                for w, preds in zip(weights, all_predictions)
            )
            results.append((aid, weighted_score))
        
        return results
    
    def predict(
        self,
        query: str,
        articles: List[Tuple[str, str]],
        threshold: float = None
    ) -> List[str]:
        """
        Get binary predictions (relevant article IDs).
        
        Returns:
            List of article IDs predicted as relevant
        """
        threshold = threshold or self.optimized_threshold or self.threshold
        scores = self.predict_scores(query, articles)
        return [aid for aid, score in scores if score >= threshold]
    
    def evaluate(
        self,
        predictions: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        beta: float = 2.0
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Dict mapping query_id to list of predicted doc_ids
            ground_truth: Dict mapping query_id to list of relevant doc_ids
            beta: Beta parameter for F-score (default 2.0 for F2)
            
        Returns:
            Dict with precision, recall, f_beta metrics
        """
        all_y_true = []
        all_y_pred = []
        
        # Collect all unique doc_ids
        all_docs = set()
        for docs in predictions.values():
            all_docs.update(docs)
        for docs in ground_truth.values():
            all_docs.update(docs)
        all_docs = sorted(all_docs)
        doc_to_idx = {d: i for i, d in enumerate(all_docs)}
        
        for qid in ground_truth.keys():
            gt_docs = set(ground_truth.get(qid, []))
            pred_docs = set(predictions.get(qid, []))
            
            for doc in all_docs:
                all_y_true.append(1 if doc in gt_docs else 0)
                all_y_pred.append(1 if doc in pred_docs else 0)
        
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f_beta = fbeta_score(all_y_true, all_y_pred, beta=beta, zero_division=0)
        
        return {
            "precision": precision,
            "recall": recall,
            f"f{beta}": f_beta
        }
    
    def optimize_weights(
        self,
        dev_queries: List[Tuple[str, str, List[Tuple[str, str]]]],
        ground_truth: Dict[str, List[str]],
        n_trials: int = 100,
        metric: str = "f2"
    ) -> Tuple[List[float], float]:
        """
        Use Optuna to optimize ensemble weights and threshold.
        
        Args:
            dev_queries: List of (query_id, query_text, articles) tuples
            ground_truth: Dict mapping query_id to list of relevant doc_ids
            n_trials: Number of Optuna trials
            metric: Optimization metric ("f2", "recall", "precision")
            
        Returns:
            (optimized_weights, optimized_threshold)
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        n_models = len(self.models)
        
        def objective(trial):
            # Sample weights
            weights = [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n_models)]
            threshold = trial.suggest_float("threshold", 0.3, 0.7)
            
            # Make predictions
            predictions = {}
            for qid, query_text, articles in dev_queries:
                scores = self.predict_scores(query_text, articles, weights=weights)
                predictions[qid] = [aid for aid, s in scores if s >= threshold]
            
            # Evaluate
            metrics = self.evaluate(predictions, ground_truth)
            
            return metrics.get(metric, metrics.get("f2"))
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best parameters
        best_params = study.best_params
        self.optimized_weights = [best_params[f"w{i}"] for i in range(n_models)]
        self.optimized_threshold = best_params["threshold"]
        
        return self.optimized_weights, self.optimized_threshold
    
    def save(self, path: str):
        """Save ensemble configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        config = {
            "weights": self.weights,
            "optimized_weights": self.optimized_weights,
            "threshold": self.threshold,
            "optimized_threshold": self.optimized_threshold,
            "n_models": len(self.models)
        }
        
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        """Load ensemble configuration (models must be loaded separately)."""
        path = Path(path)
        
        with open(path / "ensemble_config.json", "r") as f:
            config = json.load(f)
        
        self.weights = config.get("weights")
        self.optimized_weights = config.get("optimized_weights")
        self.threshold = config.get("threshold", 0.5)
        self.optimized_threshold = config.get("optimized_threshold")
        
        return self


