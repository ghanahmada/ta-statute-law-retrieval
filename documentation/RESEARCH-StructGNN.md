# StructGNN Retrieval Flow

## Overview

StructGNN extends Para-GNN by incorporating **statutory structure** into node features. Each document node is augmented with a deterministic act-membership hash and a sinusoidal position encoding that encodes where the article sits within its parent act. This structural signal is concatenated to the BGE-M3 semantic embedding before being projected back to 1024d for the EUGAT encoder.

**Model:** EUGAT (2-layer Edge-Updated Graph Attention Network) + StructureProjection  
**Base encoder:** BAAI/bge-m3 (1024-dimensional dense vectors)  
**Structure mode:** `structural` — semantic (1024d) + act hash (64d) + position (32d) = 1120d → projected to 1024d  
**BM25 blend:** Learned per-query alpha (grid-searched on validation)  
**Graph library:** DGL (Deep Graph Library)

---

## Architecture

```
Query text
  │
  ▼
BGE-M3 Encode (paragraph-level)
  │
  ▼
Concatenate structure features                 [StructGNN addition]
  │  query act hash = hash("QUERY_NODE", 64)   fixed sentinel
  │  query pos enc  = sinusoidal(0.5, 32)      fixed middle position
  │  node feat: (1024 + 64 + 32) = (1120,)
  │
  ▼
Build Query Graph
  │  doc node (1120,)
  │  para nodes (N_para × 1120,)
  │  edges: para_i → doc_node (RR label as edge feat, 1024d)
  │
  ▼
Build Candidate Graphs (same structure, with act/pos features per doc)
  │
  ▼
Merge into single DGL batch graph
  │
  ▼
StructureProjection: LayerNorm(1120) → Linear(1120 → 1024)   [StructGNN addition]
  │
  ▼
EUGAT (2 layers, 1024d, 1 head)
  │
  ▼
Extract query_encoded [Q, 1024]
Extract candidate_encoded [C, 1024]
  │
  ▼
GNN scores = query_encoded @ candidate_encoded.T   [Q × C]
  │
  ▼
Z-score normalise per query
  │
  ▼
Alpha blend: α * gnn_scores + (1−α) * bm25_scores
  │
  ▼
Ranked doc list → evaluation
```

---

## Step 1 — Pre-computation

Same as Para-GNN for BGE-M3 embeddings, RR label embeddings, and BM25 scores (see Para-GNN doc). StructGNN additionally computes structural features.

### 1a. Structural Features (StructGNN-only)

For each corpus document, two structure signals are computed:

#### Act Hash (64d)

A deterministic, dataset-agnostic representation of which act a document belongs to:

```python
def deterministic_hash_vector(name: str, dim: int) -> Tensor:
    seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return normalize(vec)   # L2 unit norm
```

- **Input:** act name string (e.g., "民法典", "Code Civil")
- **Output:** (64,) unit-norm vector, identical across runs for same input
- Documents from the same act get the same hash → model learns act-level similarity
- Acts are parsed from document titles using dataset-specific regex

#### Position Encoding (32d)

A sinusoidal encoding of the article's rank within its parent act:

```python
def sinusoidal_position_encoding(position: float, dim: int) -> Tensor:
    # position ∈ [0, 1] = (article_rank - 1) / (n_articles_in_act - 1)
    encoding = []
    for i in range(dim // 2):
        freq = 1.0 / (10000 ** (2 * i / dim))
        encoding.append(sin(position * freq * π))
        encoding.append(cos(position * freq * π))
    return Tensor(encoding)
```

- Articles at the start of an act get position ≈ 0.0; end ≈ 1.0
- Encodes relative proximity: nearby articles have similar encodings

#### Query Structure Features (Fixed Sentinels)

Query nodes cannot belong to any real act, so fixed sentinel values are used:

```python
QUERY_ACT_HASH = deterministic_hash_vector("QUERY_NODE", act_dim=64)
QUERY_POS_ENCODING = sinusoidal_position_encoding(0.5, pos_dim=32)
```

The `"QUERY_NODE"` sentinel is distinct from any real act name, so the model learns to distinguish query nodes from document nodes in the structural space.

#### Concatenation

```
node_feature = [BGE-M3_emb (1024) | act_hash (64) | pos_encoding (32)]
             = (1120,) per node
```

---

## Step 2 — Graph Construction

Identical to Para-GNN except node feature dimension is **1120** instead of 1024.

### Node Types

| Node | Feature dim | Semantic (1024) | Act hash (64) | Pos enc (32) |
|------|-------------|-----------------|---------------|--------------|
| Query doc node | 1120 | mean of query paras | QUERY_NODE sentinel | 0.5 fixed |
| Query para nodes | 1120 | sentence embedding | QUERY_NODE sentinel | 0.5 fixed |
| Candidate doc node | 1120 | mean of cand paras | act of this doc | rank in act |
| Candidate para nodes | 1120 | sentence embedding | act of this doc | rank in act |

### Edge Types

Same as Para-GNN: paragraph → parent doc edges with RR label embeddings (1024d). Edge features are **not** augmented with structural signals.

---

## Step 3 — Structure Projection

Before the EUGAT encoder, the 1120d node features are projected back to 1024d:

```python
class StructureProjection(nn.Module):
    def __init__(self, input_dim=1120, output_dim=1024):
        self.norm = LayerNorm(input_dim)
        self.proj = Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(self.norm(x))
```

This is the only added module in StructGNN compared to Para-GNN. After projection, the pipeline is identical: EUGAT → scoring → blending.

---

## Step 4 — EUGAT Encoder

Identical to Para-GNN (see Para-GNN doc, Step 3). Input dimension is now 1024 (after projection).

```
2-layer EUGATConv stack
  in_feats=1024, edge_feats=1024, out_feats=1024
  num_heads=1, dropout=0.1, residual=True
```

---

## Step 5 — Scoring and BM25 Blending

Identical to Para-GNN (see Para-GNN doc, Step 4).

```
gnn_scores = query_encoded @ candidate_encoded.T
gnn_scores = (gnn_scores - mean) / (std + 1e-8)
alpha[q]   = FFNN(query_encoded[q])   # SimpleFFNN: Linear(1024→1) → Sigmoid()
final_score = alpha * gnn_scores + (1−alpha) * bm25_scores
```

---

## Step 6 — Training

Identical to Para-GNN training (see Para-GNN doc, Step 5), with one addition:

The **StructureProjection** parameters are included in the optimiser parameter group. All other hyperparameters and procedures (IPS weighting, early stopping, alpha grid search) are the same.

---

## Step 7 — Inference

### Batch Inference

Same as Para-GNN. The key difference is the `struct_proj` call before EUGAT:

```
1. Load best_model.pt → TestCaseGnn (with struct_proj weights)
2. Build DGL graph (all test queries + all corpus, node feats 1120d)
3. node_h = model.struct_proj(graph.ndata["h"])   ← StructGNN addition
4. h = model.eugat_gnn(graph, node_h, edge_h)
5. Extract query/candidate embeddings → scores → alpha blend → rankings
```

### Online Inference (per-query)

```
Pre-encoding (once):
  For each corpus doc:
    1. Concatenate [BGE-M3 emb | act_hash | pos_enc] → (1120,)
    2. Build tiny doc graph, run struct_proj + EUGAT
    3. Store encoded embedding (1024,)
  corpus_embeddings.npy  (N_docs × 1024)

Per query:
  1. BGE-M3 encode query sentences
  2. Concatenate QUERY_NODE sentinel structure features → (1120,)
  3. Build tiny 1-doc query graph
  4. struct_proj(node_feats) → (N_nodes, 1024)
  5. EUGAT → query_encoded (1024,)
  6. gnn_scores = query_encoded @ corpus_embeddings.T
  7. final_scores = alpha * gnn_scores + (1-alpha) * bm25_scores
  8. Return top-N
```

---

## Difference from Para-GNN

| Aspect | Para-GNN | StructGNN |
|--------|----------|-----------|
| Node feature dim | 1024 | 1120 |
| Act membership | — | 64d deterministic hash |
| Position in act | — | 32d sinusoidal encoding |
| Extra module | — | StructureProjection (1120→1024) |
| Query structure | — | Fixed sentinel (`"QUERY_NODE"`, pos=0.5) |
| Edge features | RR labels (1024d) | Same — edges not augmented |
| EUGAT input | 1024d | 1024d (after projection) |
| All else | — | Identical |

The structural features give the model information about legal hierarchy: articles from the same act share an act hash, and the position encoding lets the model reason about proximity within an act (e.g., general provisions typically appear at the start).

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 1024 (BGE-M3) |
| Act hash dim | 64 |
| Position enc dim | 32 |
| Total node feat dim | 1120 (before projection) |
| EUGAT input/hidden/output | 1024 |
| Attention heads | 1 |
| GNN layers | 2 |
| Dropout | 0.1 |
| Batch size | 256 |
| Negatives per query | 299 |
| Learning rate | 1e-4 |
| Warmup ratio | 10% |
| Max epochs | 100 |
| Early stopping patience | 10 epochs |
| BM25 b / k1 | 0.75 / 1.5 |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/paragnn/structure.py` | Act parsing, act hash, sinusoidal position encoding |
| `src/paragnn/model.py` | CaseGnn + StructureProjection; struct_proj applied before EUGAT |
| `src/paragnn/graph_builder.py` | GraphBuilder: concatenates struct features to node feats when structure_mode="structural" |
| `src/paragnn/eugat.py` | EUGATConv, EUGATGNN (same as Para-GNN) |
| `src/paragnn/trainer.py` | Training loop (same as Para-GNN, struct_proj in optimiser) |
| `src/paragnn/precompute.py` | BGE-M3 + BM25 + structure feature pre-computation |
| `src/paragnn/inference.py` | Batch inference (struct_proj before EUGAT) |
| `src/paragnn/gnn_searcher.py` | Online inference (struct features on query at runtime) |
| `src/evaluate_paragnn.py` | Training entry point (`--structure_mode structural`) |
| `src/inference/infer_paragnn.py` | Inference entry point |
