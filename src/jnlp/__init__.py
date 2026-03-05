"""
JNLP COLIEE 2025 Statute Law Retrieval Pipeline
Based on: "JNLP at COLIEE 2025: Hybrid Large Language Model-based Framework 
          for Legal Information Retrieval and Entailment"

Three-Stage Pipeline:
  Stage 1: Pre-retrieval (BGE-M3 + CatBoost + Re-ranker)
  Stage 2: QLoRA Fine-tuning (Qwen2/Qwen3)
  Stage 3: Weighted Ensemble + Optuna
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from transformers import set_seed


def seed_everything(seed: int = 42):
    """Ensure reproducibility across all libraries."""
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@dataclass
class Config:
    """Pipeline configuration - dataset agnostic."""
    
    # Data paths (BEIR format)
    corpus_path: str = "data/kuhperdata/corpus.jsonl"
    queries_path: str = "data/kuhperdata/queries.jsonl"
    qrels_path: str = "data/kuhperdata/qrels.tsv"  # All qrels (for reference)
    qrels_train_path: str = "data/kuhperdata/qrels_train.tsv"  # Train split
    qrels_test_path: str = "data/kuhperdata/qrels_test.tsv"  # Test split
    
    # Stage 1: Pre-retrieval
    bge_model_name: str = "BAAI/bge-m3"
    reranker_type: str = "bge"  # "bge" or "rankllama"
    bge_reranker_name: str = "BAAI/bge-reranker-v2-m3"
    rankllama_name: str = "castorini/rankllama-v1-7b-lora-passage"
    
    # Stage 1 feature type: "histogram" (paper L1 bins) or "product" (element-wise q*d)
    stage1_feature_type: str = "product"
    # Paper Section 4.3: 76 bins for L1 distance histogram (only used when feature_type="histogram")
    n_histogram_bins: int = 76
    # Oversampling: reduced from 300x (paper) to 10x to reduce overfitting on small datasets
    stage1_oversample_ratio: int = 10
    stage1_topk: int = 100  # after CatBoost
    stage1_rerank_topk: int = 50  # after re-ranking
    encode_max_length: int = 1024  # BGE-M3 supports up to 8192
    encode_batch_size: int = 64
    
    # Stage 2: QLoRA Fine-tuning
    llm_model_name: str = "Qwen/Qwen2-7B-Instruct"  # or "Qwen/Qwen3-8B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    # Paper Section 4.3: 3x upsample positives
    stage2_upsample_ratio: int = 3
    # Hard Negative Mining for fast training (~30 min vs ~2h with full top-50)
    stage2_hard_neg_k: int = 4           # hard negatives per query (ranks 1-14)
    stage2_random_neg_k: int = 1         # random negatives per query (ranks 50-99)
    stage2_hard_neg_range: Tuple[int, int] = field(default_factory=lambda: (1, 15))
    stage2_random_neg_range: Tuple[int, int] = field(default_factory=lambda: (50, 100))

    # Training — effective batch = batch_size × gradient_accumulation_steps = 16
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_seq_length: int = 1536
    warmup_ratio: float = 0.1
    
    # Quantization (QLoRA via Unsloth)
    load_in_4bit: bool = True
    
    # Stage 3: Ensemble
    ensemble_metric: str = "f2"  # F2 prioritizes recall (legal retrieval)
    optuna_n_trials: int = 50
    
    # Output
    output_dir: str = "outputs/jnlp"
    seed: int = 42
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute based on project root."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / relative_path


from util.dataloader import DataLoader
from util.metrics import (
    calculate_mrr,
    calculate_recall_at_k,
    calculate_precision_at_k,
    evaluate_ranking,
)
from .stage1_retriever import Stage1Retriever
from .stage2_finetuner import QueryArticleDataset, Stage2FineTuner
from .stage3_ensemble import Stage3Ensemble
from .pipeline import PipelineOrchestrator

__all__ = [
    "seed_everything",
    "Config",
    "DataLoader",
    "Stage1Retriever",
    "QueryArticleDataset",
    "Stage2FineTuner",
    "Stage3Ensemble",
    "PipelineOrchestrator",
    "calculate_mrr",
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "evaluate_ranking",
]
