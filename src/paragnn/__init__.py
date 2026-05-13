"""Paragraph-level Graph Neural Network for Statute Retrieval.

Adapted from IL-PCSR (Paul et al., EMNLP 2025) for BEIR-format multilingual datasets.
Uses BGE-M3 (1024d) instead of all-mpnet-base-v2 (768d) for multilingual support.

Query methods:
  - method="full": LLM-labeled query rhetorical roles + statute "NONE"
  - method="adapted": query as single node + statute "NONE" (no LLM needed)

Structure modes (for ablation):
  - structure_mode="none": Para-GNN base (no inter-statute signal)
  - structure_mode="proximity": Prox-GNN (proximity edges by article number distance)
  - structure_mode="structural": StructGNN (act hash + positional encoding as node features)
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

# Fact types for query paragraphs — used as edge features to make query
# embeddings composition-aware (different fact compositions → different
# attention patterns → more discriminative query representations).
FACT_TYPES = [
    "CIRCUMSTANCE",  # background situation, context, status
    "ACTION",        # what someone did or failed to do
    "DAMAGE",        # harm, loss, injury suffered
    "DISPUTE",       # what is contested, the legal question
    "GENERAL",       # catch-all
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
    structure_mode: str = "none"  # "none" (Para-GNN), "proximity" (Prox-GNN), "structural" (StructGNN)
    proximity_radius: int = 50  # Prox-GNN only: connect statutes within N articles
    act_dim: int = 64  # StructGNN only: act hash embedding dimension
    pos_dim: int = 32  # StructGNN only: sinusoidal position encoding dimension

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
    patience: int = 10

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
    "kuhperdata-exp": {"path": "data/kuhperdata-exp", "lang": "id"},
    "kuhperdata-summ-exp": {"path": "data/kuhperdata-summ-exp", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}
