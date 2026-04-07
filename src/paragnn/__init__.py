"""Para-GNN: Paragraph-level Graph Neural Network for Statute Retrieval.

Adapted from IL-PCSR (Paul et al., EMNLP 2025) for BEIR-format multilingual datasets.
Uses BGE-M3 (1024d) instead of all-mpnet-base-v2 (768d) for multilingual support.

Two methods:
  - method="full": LLM-labeled query rhetorical roles + statute "NONE"
  - method="adapted": query as single node + statute "NONE" (no LLM needed)
"""
from dataclasses import dataclass, field
from typing import Optional, List


# Rhetorical role labels from IL-PCSR / LegalSeg (Nigam et al., NAACL 2025)
RR_LABELS = [
    "Argument by Petitioner",
    "Argument by Respondent",
    "Conclusion",
    "Court Disclosure",
    "Court Reasoning",
    "Facts",
    "Issue",
    "NONE",
    "Precedent",
    "Section",
    "Statute",
    "Cites",
    "Paragraph Cites",
]


@dataclass
class ParaGNNConfig:
    # Data
    dataset: str = "kuhperdata-humanized"
    data_path: str = "data/kuhperdata-humanized"
    output_dir: str = "outputs/paragnn"
    lang: str = "id"
    max_relevant: int = 5

    # Method
    method: str = "adapted"  # "full" or "adapted"
    rr_labels_path: Optional[str] = None  # path to rr_labels.json (method="full" only)

    # Graph structure
    proximity_radius: int = 0  # 0 = no proximity edges, N = connect statutes within N articles

    # Model
    embed_dim: int = 1024  # BGE-M3 output dimension
    h_dim: int = 1024
    out_dim: int = 1024
    num_heads: int = 1
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 256
    num_negatives: int = 299  # candidates per query = 1 positive + num_negatives
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    grad_accumulation: int = 1

    # BM25
    bm25_b: float = 0.75
    bm25_k1: float = 1.5
    bm25_ngram: int = 1

    # Encoding
    bge_model_name: str = "BAAI/bge-m3"
    encode_batch_size: int = 32
    encode_max_length: int = 512


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}
