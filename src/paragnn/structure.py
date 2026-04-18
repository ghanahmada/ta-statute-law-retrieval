"""Structural metadata extraction for statute retrieval.

Parses act membership and positional rank from corpus titles.
Language-agnostic: uses index rank (order of appearance) instead of
parsing article numbers, which vary across legal systems.

Supported datasets: KUHPerdata (ID), BSARD (FR), STARD (ZH), IL-PCSR (EN).
"""
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def parse_act_name(title: str, dataset: str) -> str:
    """Extract the act/code name from a statute title.

    Returns a normalized string identifying which legal instrument
    this statute belongs to.
    """
    if dataset.startswith("kuhperdata"):
        return "KUHPerdata"

    if dataset == "bsard":
        m = re.search(r",\s*(.+?)\s*\(", title)
        return m.group(1).strip() if m else "unknown"

    if dataset == "stard":
        m = re.match(r"(.+?)第", title)
        return m.group(1).strip() if m else "unknown"

    if dataset == "ilpcsr":
        m = re.search(r"of (.+?)(?:,\s*\d|$)", title)
        return m.group(1).strip() if m else "unknown"

    return "unknown"


def build_structure_metadata(
    corpus_path: str, dataset: str
) -> Dict[str, Dict]:
    """Parse corpus and build structural metadata for each document.

    Returns:
        {doc_id: {"act_name": str, "act_id": int, "position": float}}
        where position = index_within_act / total_articles_in_act (0-1).
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    act_groups: Dict[str, List[str]] = {}
    doc_act: Dict[str, str] = {}

    for doc in docs:
        doc_id = doc["_id"]
        act_name = parse_act_name(doc.get("title", ""), dataset)
        doc_act[doc_id] = act_name
        if act_name not in act_groups:
            act_groups[act_name] = []
        act_groups[act_name].append(doc_id)

    act_vocab = {name: idx for idx, name in enumerate(sorted(act_groups.keys()))}

    metadata = {}
    for doc_id, act_name in doc_act.items():
        group = act_groups[act_name]
        idx_in_act = group.index(doc_id)
        total_in_act = len(group)
        position = idx_in_act / max(total_in_act - 1, 1)

        metadata[doc_id] = {
            "act_name": act_name,
            "act_id": act_vocab[act_name],
            "position": position,
        }

    return metadata


def deterministic_hash_vector(name: str, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a deterministic pseudo-random vector from a string.

    Same input always produces the same output. Different inputs produce
    near-orthogonal vectors in high dimensions.
    """
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = torch.Generator()
    rng.manual_seed(int.from_bytes(h[:8], "big"))
    vec = torch.randn(dim, generator=rng)
    return vec / vec.norm()


def sinusoidal_position_encoding(position: float, dim: int = 32) -> torch.Tensor:
    """Sinusoidal positional encoding at multiple frequencies.

    position: normalized 0-1 value (index rank within act).
    dim: output dimension (must be even).
    """
    assert dim % 2 == 0
    encoding = torch.zeros(dim)
    for i in range(dim // 2):
        freq = 1.0 / (10000 ** (2 * i / dim))
        encoding[2 * i] = math.sin(position * freq * math.pi)
        encoding[2 * i + 1] = math.cos(position * freq * math.pi)
    return encoding


QUERY_ACT_HASH: Optional[torch.Tensor] = None
QUERY_POS_ENCODING: Optional[torch.Tensor] = None


def get_query_structure_features(act_dim: int = 64, pos_dim: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get structure features for query nodes.

    Uses a dedicated query hash (not zeros) and a fixed 0.5 position
    to give the GNN a distinct signature for query nodes.
    """
    global QUERY_ACT_HASH, QUERY_POS_ENCODING
    if QUERY_ACT_HASH is None or QUERY_ACT_HASH.shape[0] != act_dim:
        QUERY_ACT_HASH = deterministic_hash_vector("QUERY_NODE", act_dim)
    if QUERY_POS_ENCODING is None or QUERY_POS_ENCODING.shape[0] != pos_dim:
        QUERY_POS_ENCODING = sinusoidal_position_encoding(0.5, pos_dim)
    return QUERY_ACT_HASH, QUERY_POS_ENCODING


def precompute_structure_features(
    corpus_path: str, dataset: str, act_dim: int = 64, pos_dim: int = 32
) -> Dict[str, torch.Tensor]:
    """Precompute concatenated structure features for all corpus documents.

    Returns:
        {doc_id: tensor of shape (act_dim + pos_dim,)}
    """
    metadata = build_structure_metadata(corpus_path, dataset)

    act_hash_cache: Dict[str, torch.Tensor] = {}
    features = {}

    for doc_id, meta in metadata.items():
        act_name = meta["act_name"]
        if act_name not in act_hash_cache:
            act_hash_cache[act_name] = deterministic_hash_vector(act_name, act_dim)

        act_vec = act_hash_cache[act_name]
        pos_vec = sinusoidal_position_encoding(meta["position"], pos_dim)
        features[doc_id] = torch.cat([act_vec, pos_vec])

    return features
