# StructGNN Evaluation Guide

Extension of Para-GNN with structural node features (act membership + positional rank).
Designed as a language-agnostic, scalable alternative to Prox-GNN's explicit proximity edges.

---

## Architecture Overview

### Problem with Prox-GNN
Prox-GNN adds explicit edges between statutes within N articles of each other. This works for KUHPerdata (single code, sequential numbering) but breaks on other datasets:
- BSARD: 35 different codes, article numbering varies per code
- STARD: 1,257 acts, non-sequential IDs
- IL-PCSR: 95 acts, random doc IDs (e.g., "37054924")

### StructGNN Solution
Instead of explicit edges, encode structural metadata as **node features** and let the GNN's attention mechanism learn which structural patterns matter.

Each statute node gets:
```
node_feature = Linear(LayerNorm([semantic | act_hash | pos_enc]))
                                 1024d     64d       32d      → 1024d
```

| Component | Dimension | Source | Purpose |
|---|---|---|---|
| Semantic | 1024d | BGE-M3 embedding (precomputed) | Content meaning |
| Act hash | 64d | Deterministic hash of act name | Same act → identical vector, different act → near-orthogonal |
| Position | 32d | Sinusoidal encoding of index rank within act | Multi-scale within-act proximity |

### Why This Works
The EUGAT attention computes: `attention = f(source_node, dest_node, edge_feat)`.

With structural features concatenated into node features:
- **Same act, close position**: act_hash dot product ≈ 1, position vectors similar → high attention
- **Same act, far position**: act_hash dot product ≈ 1, position vectors different → moderate attention
- **Different act**: act_hash dot product ≈ 0 → low attention

The linear projection (LayerNorm → Linear) preserves the geometric orthogonality of act hashes. The GNN's attention handles nonlinear routing.

### Query Nodes
Queries don't belong to any act. Instead of zeros (which causes unstable LayerNorm scaling), they receive:
- Act component: `deterministic_hash("QUERY_NODE")` — distinct signature meaning "I am a query"
- Position component: sinusoidal encoding of 0.5 — neutral midpoint

### Parameter Cost
- `StructureProjection`: LayerNorm(1120) + Linear(1120, 1024) = ~1.15M parameters
- For comparison: EUGAT has ~12M parameters (+9.6%)
- Para-GNN and Prox-GNN skip this layer entirely

---

## Three Methods for Ablation

| Method | `--structure_mode` | Node features | Edges | Works on |
|---|---|---|---|---|
| Para-GNN | `none` | BGE-M3 (1024d) | paragraph→doc only | All datasets |
| Prox-GNN | `proximity` | BGE-M3 (1024d) | + proximity edges | KUHPerdata only |
| StructGNN | `structural` | BGE-M3 + act + pos (1120d→1024d) | paragraph→doc only | All datasets |

All three share the same precomputed embeddings and BM25 scores.

Output directories:
```
outputs/paragnn/{dataset}/adapted/           ← Para-GNN
outputs/paragnn/{dataset}/adapted_prox50/    ← Prox-GNN
outputs/paragnn/{dataset}/adapted_struct/    ← StructGNN
```

---

## Act Parsing Per Dataset

| Dataset | Title format | Parsed act name | # Acts | Largest act |
|---|---|---|---|---|
| KUHPerdata | `Pasal 1365` | `KUHPerdata` (single) | 1 | 2,127 |
| BSARD | `Art. 1.1.1, Code Bruxellois... (Livre 1er)` | `Code Bruxellois de l'Air...` | 35 | 2,618 |
| STARD | `中华人民共和国民法典第四百六十三条` | `中华人民共和国民法典` | 1,257 | 1,260 |
| IL-PCSR | `Section 171 of The Motor Vehicles Act, 1988` | `The Motor Vehicles Act` | 95 | 146 |

Position is computed as **index rank** (order of appearance in corpus), not parsed article numbers. This avoids parsing "Art. 1.1.1", Chinese numerals, or amendment letters.

Formula: `position = index_within_act / (total_articles_in_act - 1)`

---

## Files Modified/Created

| File | Change |
|---|---|
| `src/paragnn/structure.py` | **New.** Act parsing, hash vectors, sinusoidal PE, precompute function |
| `src/paragnn/__init__.py` | Added `structure_mode`, `act_dim`, `pos_dim` to ParaGNNConfig |
| `src/paragnn/graph_builder.py` | Supports 3 modes. StructGNN concatenates structure features to all nodes |
| `src/paragnn/model.py` | Added `StructureProjection` (LayerNorm→Linear). CaseGnn/TestCaseGnn accept `structure_mode` |
| `src/paragnn/dataset.py` | Collator passes structure_features + query_structure_feature to GraphBuilder |
| `src/paragnn/trainer.py` | Wires structure_mode throughout. Method-specific output folders |
| `src/evaluate_paragnn.py` | New CLI with `--structure_mode`. Precomputes structure features when structural |

---

## Commands

### Prerequisites
Same precomputed data as Para-GNN (embeddings + BM25 scores). No additional precomputation needed.

```bash
# If not already done:
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted
python src/paragnn/precompute.py --dataset kuhperdata-summarized --method adapted
python src/paragnn/precompute.py --dataset bsard --method adapted
python src/paragnn/precompute.py --dataset stard --method adapted
python src/paragnn/precompute.py --dataset ilpcsr --method adapted
```

### Run Ablation (all three methods)

```bash
# Para-GNN (base)
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode none --epochs 50
python src/evaluate_paragnn.py --dataset kuhperdata-summarized --structure_mode none --epochs 50
python src/evaluate_paragnn.py --dataset bsard --structure_mode none --epochs 50
python src/evaluate_paragnn.py --dataset stard --structure_mode none --epochs 50

# Prox-GNN (KUHPerdata only — proximity edges don't generalize)
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode proximity --epochs 50
python src/evaluate_paragnn.py --dataset kuhperdata-summarized --structure_mode proximity --epochs 50

# StructGNN (all datasets)
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode structural --epochs 50
python src/evaluate_paragnn.py --dataset kuhperdata-summarized --structure_mode structural --epochs 50
python src/evaluate_paragnn.py --dataset bsard --structure_mode structural --epochs 50
python src/evaluate_paragnn.py --dataset stard --structure_mode structural --epochs 50
python src/evaluate_paragnn.py --dataset ilpcsr --structure_mode structural --epochs 50
```

### Custom Hyperparameters

```bash
# Adjust structure feature dimensions
python src/evaluate_paragnn.py --dataset bsard --structure_mode structural \
    --act_dim 64 --pos_dim 32 --epochs 50

# Adjust proximity radius (Prox-GNN only)
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --structure_mode proximity \
    --proximity_radius 20 --epochs 50
```

---

## Expected Results Table

Run all experiments and fill in:

### KUHPerdata-humanized

| Method | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Para-GNN | 0.456 | 0.542 | 67.8% | 0.9 |
| Prox-GNN (r=50) | 0.486 | 0.551 | 68.3% | 0.9 |
| StructGNN | | | | |

### KUHPerdata-summarized

| Method | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Para-GNN | 0.458 | 0.522 | 71.1% | 0.8 |
| Prox-GNN (r=50) | 0.498 | 0.516 | 69.2% | 0.9 |
| StructGNN | | | | |

### BSARD

| Method | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Para-GNN | 0.493 | 0.487 | 68.5% | 0.7 |
| StructGNN | | | | |

### STARD

| Method | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Para-GNN | 0.527 | 0.617 | 72.4% | 0.8 |
| StructGNN | | | | |

### IL-PCSR

| Method | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Para-GNN | | | | |
| StructGNN | | | | |

---

## What to Watch For

1. **StructGNN vs Para-GNN on multi-act datasets (BSARD, STARD, IL-PCSR)**: This is where StructGNN should shine — act membership is a strong signal that Para-GNN can't use.

2. **StructGNN vs Prox-GNN on KUHPerdata**: StructGNN has position encoding but no explicit edges. If StructGNN matches or beats Prox-GNN here, it validates that learned structural attention > hardcoded edges.

3. **Alpha shift**: If StructGNN finds different optimal alpha than Para-GNN, it means structural features change the GNN's need for BM25 signal.

4. **Training stability**: The StructureProjection adds parameters. Check if training converges similarly (same epoch count, similar loss curves) or if it needs more/fewer epochs.

---

## Environment Setup

Same as Para-GNN — no additional dependencies:

```bash
conda create -n paragnn python=3.11 -y
conda activate paragnn
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install FlagEmbedding==1.3.5 transformers==4.44.2 sentence-transformers
pip install numpy scipy scikit-learn tqdm jieba accelerate pydantic pyyaml
```
