import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from tqdm import tqdm
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler

from util.dataloader import DataLoader


class Stage1Retriever:
    """
    Paper Section 4.3 - Pre-retrieval Stage:
    1. BGE-M3 dense embeddings -> 1024 embed dims
    2. L1 distance histogram features (76 bins)
    3. CatBoost classifier with oversampling
    4. Re-ranking with cross-encoder

    Adjustments from original COLIEE 2025 paper:
    1. Feature Extraction:
      - Histogram range: calibrated from data instead of fixed (0, 2)
      - Element-wise product features added as alternative to histograms (configurable via feature_type)
    2. Oversampling: reduced from 300x (paper) to 10x to reduce overfitting
    3. Negatives: BM25 top 100 non positive as hard negatives instead of random sampling
    """

    FEATURE_TYPES = ("histogram", "product")

    def __init__(
        self,
        bge_model_name: str = "BAAI/bge-m3",
        feature_type: str = "product",
        n_bins: int = 76,
        oversample_ratio: int = 10,
        device: str = None
    ):
        if feature_type not in self.FEATURE_TYPES:
            raise ValueError(f"feature_type must be one of {self.FEATURE_TYPES}, got '{feature_type}'")

        self.bge_model_name = bge_model_name
        self.feature_type = feature_type
        self.n_bins = n_bins
        self.oversample_ratio = oversample_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.bge_model = None
        self.classifier: Optional[CatBoostClassifier] = None
        self.reranker = None
        self.reranker_type = None

        # Store embeddings
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.doc_ids: Optional[List[str]] = None

        # L1 histogram range — calibrated from data (only used for feature_type="histogram")
        self.l1_hist_range: Optional[Tuple[float, float]] = None

        # Query embedding cache for faster evaluation
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        
    def load_bge_model(self):
        """Load BGE-M3 model for dense retrieval (Paper Section 4.3)."""
        import util.compat  # noqa: F401 — patch transformers for FlagEmbedding compat
        from FlagEmbedding import BGEM3FlagModel
        self.bge_model = BGEM3FlagModel(
            self.bge_model_name,
            use_fp16=True,
            devices=self.device
        )
        return self
    
    def encode_bge(
        self, 
        texts: List[str], 
        batch_size: int = 64,
        max_length: int = 1024
    ) -> np.ndarray:
        """
        Batch encode texts using BGE-M3.
        Returns dense embeddings (N, D).
        """
        if self.bge_model is None:
            self.load_bge_model()
        
        # BGE-M3 encode handles batching internally
        output = self.bge_model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # BGE-M3 returns dict with 'dense_vecs'
        if isinstance(output, dict):
            embeddings = output["dense_vecs"]
        else:
            embeddings = output
        
        return np.array(embeddings)
    
    def encode_corpus(
        self, 
        doc_ids: List[str], 
        texts: List[str], 
        batch_size: int = 64,
        max_length: int = 1024
    ):
        """Encode and store corpus embeddings."""
        self.doc_ids = doc_ids
        self.corpus_embeddings = self.encode_bge(texts, batch_size, max_length)
        return self
    
    def encode_queries_batch(
        self,
        query_ids: List[str],
        query_texts: List[str],
        batch_size: int = 64,
        max_length: int = 1024,
        cache_path: str = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batch encode queries with in-memory + optional disk caching.
        Disk cache avoids re-encoding on repeated evaluation runs.
        """
        # Try loading from disk cache
        if cache_path and Path(cache_path).exists():
            data = np.load(cache_path, allow_pickle=True).item()
            self._query_embedding_cache.update(data)
            print(f"Loaded {len(data)} cached query embeddings from {cache_path}")

        # Filter out already cached
        to_encode_ids = []
        to_encode_texts = []
        for qid, text in zip(query_ids, query_texts):
            if qid not in self._query_embedding_cache:
                to_encode_ids.append(qid)
                to_encode_texts.append(text)

        if to_encode_texts:
            print(f"Encoding {len(to_encode_texts)} queries...")
            embeddings = self.encode_bge(to_encode_texts, batch_size, max_length)
            for qid, emb in zip(to_encode_ids, embeddings):
                self._query_embedding_cache[qid] = emb

            # Save to disk cache
            if cache_path:
                result = {qid: self._query_embedding_cache[qid] for qid in query_ids}
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, result)
                print(f"Saved query embeddings to {cache_path}")
        else:
            print(f"All {len(query_ids)} query embeddings loaded from cache")

        return {qid: self._query_embedding_cache[qid] for qid in query_ids}
    
    def calibrate_l1_range(self, query_embeddings: Dict[str, np.ndarray], n_samples: int = 50):
        """
        Calibrate L1 histogram range from actual data distribution.

        Adjustment: paper uses fixed range=(0, 2), but for L2-normalized 1024-dim
        vectors the element-wise L1 values are much smaller (~0.02 mean).
        Fixed range wastes 90%+ of bins on empty space.
        """
        sample_qids = list(query_embeddings.keys())[:min(n_samples, len(query_embeddings))]
        all_maxes = []
        for qid in sample_qids:
            l1 = np.abs(self.corpus_embeddings - query_embeddings[qid].reshape(1, -1))
            all_maxes.append(l1.max())
        range_max = float(np.max(all_maxes) * 1.1)  # 10% buffer
        self.l1_hist_range = (0.0, range_max)
        print(f"Calibrated L1 histogram range: (0, {range_max:.4f})")
        return self

    def compute_l1_histograms(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray = None
    ) -> np.ndarray:
        """
        Paper Section 4.3: Discretize L1 distance into histogram bins.

        Uses calibrated range if available, otherwise falls back to (0, 2).
        """
        if doc_embeddings is None:
            doc_embeddings = self.corpus_embeddings

        query_embedding = query_embedding.reshape(1, -1)
        l1_distances = np.abs(doc_embeddings - query_embedding)

        hist_range = self.l1_hist_range or (0, 2)

        histograms = []
        for doc_l1 in l1_distances:
            hist, _ = np.histogram(doc_l1, bins=self.n_bins, range=hist_range)
            histograms.append(hist)

        return np.array(histograms, dtype=np.float32)

    def compute_features(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray = None
    ) -> np.ndarray:
        """Compute pairwise features based on configured feature_type."""
        if doc_embeddings is None:
            doc_embeddings = self.corpus_embeddings

        if self.feature_type == "histogram":
            return self.compute_l1_histograms(query_embedding, doc_embeddings)
        else:  # product
            q = query_embedding.reshape(1, -1)
            return (doc_embeddings * q).astype(np.float32)

    def prepare_training_data(
        self,
        data_loader: DataLoader,
        query_embeddings: Dict[str, np.ndarray],
        neg_per_query: int = 50,
        hard_negatives: Dict[str, List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for CatBoost classifier.

        Adjustments from paper:
        - hard_negatives: BM25-ranked docs used as negatives instead of random.
          Paper uses random sampling; we use BM25 top-K to force the classifier
          to learn harder distinctions.
        - Oversampling reduced from 300x to 10x (configured via oversample_ratio).

        Args:
            data_loader: Loaded DataLoader with qrels
            query_embeddings: Dict mapping query_id to embedding
            neg_per_query: Number of negative samples per query
            hard_negatives: Dict mapping query_id to ranked doc_ids (e.g. from BM25).
                           If None, falls back to random negatives (paper behavior).
        """
        X_list = []
        y_list = []

        doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        all_doc_ids = set(self.doc_ids)

        for qid, embedding in tqdm(query_embeddings.items(), desc="Preparing features"):
            if qid not in data_loader.qrels:
                continue

            relevant_docs = set(data_loader.qrels[qid].keys())

            # Positive pairs: only relevant docs
            for doc_id in relevant_docs:
                if doc_id not in doc_id_to_idx:
                    continue
                idx = doc_id_to_idx[doc_id]
                feat = self.compute_features(embedding, self.corpus_embeddings[idx:idx+1])
                X_list.append(feat[0])
                y_list.append(1)

            # Negative sampling: hard negatives (BM25) or random fallback
            if hard_negatives and qid in hard_negatives:
                neg_candidates = [
                    did for did in hard_negatives[qid]
                    if did not in relevant_docs and did in doc_id_to_idx
                ]
                sampled_negatives = neg_candidates[:neg_per_query]
                # fill with random if not enough hard negatives
                if len(sampled_negatives) < neg_per_query:
                    remaining = list(all_doc_ids - relevant_docs - set(sampled_negatives))
                    extra = min(neg_per_query - len(sampled_negatives), len(remaining))
                    sampled_negatives += random.sample(remaining, extra)
            else:
                non_relevant = list(all_doc_ids - relevant_docs)
                sampled_negatives = random.sample(non_relevant, min(neg_per_query, len(non_relevant)))

            for doc_id in sampled_negatives:
                idx = doc_id_to_idx[doc_id]
                feat = self.compute_features(embedding, self.corpus_embeddings[idx:idx+1])
                X_list.append(feat[0])
                y_list.append(0)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Training pairs: {len(y)} ({int(y.sum())} positive, {len(y) - int(y.sum())} negative)")

        # Oversampling (paper: 300x, adjusted: 10x)
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count

        if self.oversample_ratio > 1 and pos_count > 0:
            target_pos = min(pos_count * self.oversample_ratio, neg_count)
            sampling_strategy = {0: neg_count, 1: int(target_pos)}

            oversampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
            X, y = oversampler.fit_resample(X, y)
            print(f"After {self.oversample_ratio}x oversampling: {len(y)} ({int(y.sum())} pos, {len(y) - int(y.sum())} neg)")

        return X, y
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        verbose: int = 100
    ):
        """
        Train CatBoost classifier on histogram features.
        """
        self.classifier = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=verbose,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            devices="0" if torch.cuda.is_available() else None
        )
        
        self.classifier.fit(X, y)
        return self
    
    def predict_topk(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k documents for a query.
        
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")
        
        features = self.compute_features(query_embedding)
        scores = self.classifier.predict_proba(features)[:, 1]
        
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(self.doc_ids[i], float(scores[i])) for i in top_indices]
        
        return results
    
    def load_reranker(
        self,
        reranker_type: str = "bge",
        model_name: str = None
    ):
        """
        Load re-ranker model.
        
        Args:
            reranker_type: "bge" for BGE-reranker-v2-m3, "rankllama" for RankLLaMA
            model_name: Override default model name
        """
        self.reranker_type = reranker_type
        
        if reranker_type == "bge":
            import util.compat  # noqa: F401
            from FlagEmbedding import FlagReranker
            model_name = model_name or "BAAI/bge-reranker-v2-m3"
            self.reranker = FlagReranker(
                model_name,
                use_fp16=True,
                device=self.device
            )
        elif reranker_type == "rankllama":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = model_name or "castorini/rankllama-v1-7b-lora-passage"
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unknown reranker_type: {reranker_type}")
        
        return self
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        topk: int = 50,
        batch_size: int = 64
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of (doc_id, doc_text, initial_score) tuples
            topk: Number of top results to return
            batch_size: Batch size for inference
            
        Returns:
            List of (doc_id, reranker_score) tuples
        """
        if self.reranker is None:
            raise ValueError("Reranker not loaded. Call load_reranker first.")
        
        doc_ids = [c[0] for c in candidates]
        doc_texts = [c[1] for c in candidates]
        
        if self.reranker_type == "bge":
            pairs = [[query, doc] for doc in doc_texts]
            scores = self.reranker.compute_score(pairs, batch_size=batch_size)
            if not isinstance(scores, list):
                scores = [scores]
        
        elif self.reranker_type == "rankllama":
            scores = []
            for i in range(0, len(doc_texts), batch_size):
                batch_docs = doc_texts[i:i + batch_size]
                inputs = self.reranker_tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.reranker.device)
                
                with torch.no_grad():
                    outputs = self.reranker(**inputs)
                    batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)
        
        # Sort by score and return top-k
        scored = list(zip(doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:topk]
    
    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        corpus: Dict[str, Dict[str, str]],
        stage1_topk: int = 100,
        rerank_topk: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Full Stage 1 pipeline: CatBoost retrieval + re-ranking.
        
        Args:
            query_text: Query text for re-ranking
            query_embedding: BGE-M3 embedding of query
            corpus: Dict mapping doc_id to {"text": str, "title": str}
            stage1_topk: Top-k from CatBoost
            rerank_topk: Top-k after re-ranking
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Step 1: CatBoost retrieval
        candidates = self.predict_topk(query_embedding, k=stage1_topk)
        
        # Prepare for re-ranking
        candidates_with_text = [
            (doc_id, corpus[doc_id]["text"], score)
            for doc_id, score in candidates
            if doc_id in corpus
        ]
        
        # Step 2: Re-ranking
        if self.reranker is not None:
            results = self.rerank(query_text, candidates_with_text, topk=rerank_topk)
        else:
            results = [(did, score) for did, _, score in candidates_with_text[:rerank_topk]]
        
        return results
    
    def retrieve_batch(
        self,
        queries: Dict[str, Tuple[str, np.ndarray]],  # qid -> (text, embedding)
        corpus: Dict[str, Dict[str, str]],
        stage1_topk: int = 100,
        rerank_topk: int = 50,
        use_reranker: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch retrieval for multiple queries (much faster than individual calls).
        
        Args:
            queries: Dict mapping query_id to (text, embedding) tuple
            corpus: Document corpus
            stage1_topk: Top-k from CatBoost
            rerank_topk: Top-k after re-ranking
            use_reranker: Whether to use re-ranker
            
        Returns:
            Dict mapping query_id to list of (doc_id, score) tuples
        """
        results = {}
        
        # Step 1: CatBoost prediction for all queries (fast)
        for qid, (_, emb) in tqdm(queries.items(), desc="CatBoost retrieval"):
            results[qid] = self.predict_topk(emb, k=stage1_topk)
        
        # Step 2: Re-ranking (optional, slow)
        if use_reranker and self.reranker is not None:
            reranked = {}
            for qid, candidates in tqdm(results.items(), desc="Re-ranking"):
                query_text = queries[qid][0]
                candidates_with_text = [
                    (doc_id, corpus[doc_id]["text"], score)
                    for doc_id, score in candidates
                    if doc_id in corpus
                ]
                reranked[qid] = self.rerank(query_text, candidates_with_text, topk=rerank_topk)
            return reranked
        else:
            # No re-ranking, just truncate
            return {qid: cands[:rerank_topk] for qid, cands in results.items()}
    
    def save(self, path: str):
        """Save classifier, embeddings, and calibration metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.classifier is not None:
            self.classifier.save_model(str(path / "catboost_model.cbm"))

        if self.corpus_embeddings is not None:
            np.save(path / "corpus_embeddings.npy", self.corpus_embeddings)

        if self.doc_ids is not None:
            with open(path / "doc_ids.json", "w", encoding="utf-8") as f:
                json.dump(self.doc_ids, f)

        # Persist feature_type and calibrated histogram range
        meta = {"feature_type": self.feature_type}
        if self.l1_hist_range is not None:
            meta["l1_hist_range"] = list(self.l1_hist_range)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, path: str):
        """Load classifier, embeddings, and calibration metadata."""
        path = Path(path)

        model_path = path / "catboost_model.cbm"
        if model_path.exists():
            self.classifier = CatBoostClassifier()
            self.classifier.load_model(str(model_path))

        emb_path = path / "corpus_embeddings.npy"
        if emb_path.exists():
            self.corpus_embeddings = np.load(emb_path)

        ids_path = path / "doc_ids.json"
        if ids_path.exists():
            with open(ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)

        # Restore feature_type and calibrated histogram range
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "feature_type" in meta:
                self.feature_type = meta["feature_type"]
            if "l1_hist_range" in meta:
                self.l1_hist_range = tuple(meta["l1_hist_range"])

        return self
