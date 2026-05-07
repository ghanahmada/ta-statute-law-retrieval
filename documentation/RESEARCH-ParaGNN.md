# Para-GNN Retrieval Flow

## Overview

Para-GNN is a graph neural network retrieval model adapted from IL-PCSR (Paul et al., EMNLP 2025) for multilingual statute retrieval. It encodes both query and corpus documents as paragraph-level graphs, then uses an Edge-Updated Graph Attention Network (EUGAT) to produce embeddings that are scored via dot product and blended with BM25.

**Model:** EUGAT (2-layer Edge-Updated Graph Attention Network)  
**Base encoder:** BAAI/bge-m3 (1024-dimensional dense vectors)  
**Structure mode:** `none` — semantic embeddings only, no structural features  
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
Build Query Graph
  │  doc node (mean of para embs)
  │  para nodes (one per sentence)
  │  edges: para_i → doc_node (RR label as edge feat)
  │
  ▼
Build Candidate Graphs (same structure per corpus doc)
  │
  ▼
Merge into single DGL batch graph
  │
  ▼
EUGAT (2 layers, 1024d, 1 head)
  │  node features updated by attention over neighbours
  │  edge features updated as src + dst + edge_feat
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
  │  α ∈ [0,1] learned per-query via FFNN, grid-searched on val
  │
  ▼
Ranked doc list → evaluation
```

---

## Step 1 — Pre-computation

Before training or inference, three artefacts are computed offline and cached to disk.

### 1a. BGE-M3 Paragraph Embeddings

Each document and query is split into sentences (paragraphs), then each sentence is encoded with BGE-M3:

- **Model:** `BAAI/bge-m3`, output dim 1024
- **Sentence splitting:** language-aware regex
  - Chinese: split on `。！？\n`
  - Other (Indonesian, French, English): split on `.!?;` followed by space/newline
  - Minimum sentence length: 10 characters; max encode length: 512 tokens
- Stored per document/query in `embeddings/{corpus,queries}/*.pt`
- **Doc node feature:** mean pool of all paragraph embeddings → (1024,)
- **Para node features:** individual sentence embeddings → (N_para, 1024)

### 1b. Rhetorical Role (RR) Label Embeddings

13 role labels are encoded once with BGE-M3 and stored as a constant embedding matrix:

```
RR_LABELS = [
    "Argument by Petitioner", "Argument by Respondent", "Conclusion",
    "Court Disclosure", "Court Reasoning", "Facts", "Issue", "NONE",
    "Precedent", "Section", "Statute", "Cites", "Paragraph Cites"
]
```

Stored as `EMBD_CONST.pt` (13 × 1024). Each paragraph→doc edge uses the RR label embedding for that paragraph as its edge feature. In the **adapted** query encoding method (default), all query paragraphs receive the `"NONE"` label.

### 1c. BM25 Scores

BM25 is fitted on the full corpus and pre-scored against all train/val/test queries:

- Parameters: `b=0.75, k1=1.5, n_gram=1`
- Language-aware tokenisation: `jieba` for Chinese, whitespace for others
- Stored as `bm25_{split}_scores.pt` (N_queries × N_docs)
- Used both in training (loss blending) and inference (alpha blending)

---

## Step 2 — Graph Construction

For each batch of queries and candidate documents, a **dynamic DGL graph** is built.

### Node Types

| Node | Feature | Source |
|------|---------|--------|
| Doc node (query) | mean of query para embeddings (1024,) | BGE-M3 |
| Doc node (candidate) | mean of candidate para embeddings (1024,) | BGE-M3 |
| Para node (query) | individual sentence embedding (1024,) | BGE-M3 |
| Para node (candidate) | individual sentence embedding (1024,) | BGE-M3 |

### Edge Types

| Edge | Feature |
|------|---------|
| para_i → parent doc | RR label embedding (1024,) for that paragraph |
| self-loops (all nodes) | zero-padded to match edge feature dim |

Edges go **from paragraph to parent doc node** only. There are no query→candidate edges — the model learns to align them implicitly through the embedding space.

### Node Masks

Four boolean masks stored in `graph.ndata`:

| Mask | Meaning |
|------|---------|
| `query_mask` | 1.0 for query doc nodes |
| `candidate_mask` | 1.0 for candidate doc nodes |
| `query_para_mask` | 1.0 for query paragraph nodes |
| `candidate_para_mask` | 1.0 for candidate paragraph nodes |

These are used after the EUGAT forward pass to extract the relevant node embeddings.

---

## Step 3 — EUGAT Encoder

### EUGATConv (Single Layer)

Based on SCENE (https://arxiv.org/pdf/2301.03512.pdf). Updates both node and edge features simultaneously.

```
Attention computation:
  e_l = (feat_src * attn_l).sum(dim=-1)        # source node contribution
  e_r = (feat_dst * attn_r).sum(dim=-1)        # destination node contribution
  e_e = (feat_edge * attn_edge).sum(dim=-1)    # edge contribution
  e   = LeakyReLU(e_l + e_r + e_e, slope=0.2) # combined logit
  a   = softmax(e)                             # normalised attention weight

Message aggregation:
  m     = (feat_src + feat_edge) * a           # weighted message
  h_new = sum(m over neighbours)              # aggregate

Edge update:
  f_out = LeakyReLU(feat_src + feat_dst + feat_edge)
```

**Parameters per layer:**
```
in_feats=1024, edge_feats=1024, out_feats=1024, out_edge_feats=1024
num_heads=1, feat_drop=0.0, attn_drop=0.0, residual=True
```

### EUGATGNN (2-Layer Stack)

```
Layer 1:
  h_node_1, h_edge_1 = EUGATConv1(g, node_feats, edge_feats)
  h_node_1 = Dropout(ReLU(h_node_1)) + node_feats   # residual from input
  h_edge_1 = Dropout(ReLU(h_edge_1)) + edge_feats   # residual from input

Layer 2:
  h_node_2, _ = EUGATConv2(g, h_node_1, h_edge_1)
  h_node_2 = ReLU(h_node_2) + node_feats             # residual from original input

Output: h_node_2  (N_nodes, 1024)
```

Dropout rate: 0.1.

---

## Step 4 — Scoring and BM25 Blending

After the EUGAT forward pass, query and candidate doc node embeddings are extracted using the masks.

### GNN Score

```
query_encoded     = h[query_mask]       # (N_queries, 1024)
candidate_encoded = h[candidate_mask]   # (N_cands, 1024)

gnn_scores = query_encoded @ candidate_encoded.T    # (N_queries, N_cands)
gnn_scores = (gnn_scores - mean) / (std + 1e-8)    # Z-score normalise per query
```

### Learned Alpha Blending

A small FFNN produces a per-query blending weight:

```
SimpleFFNN: Linear(1024 → 1) → Sigmoid()
alpha[q] = FFNN(query_encoded[q])    # scalar ∈ [0, 1]

final_score[q, c] = alpha[q] * gnn_scores[q, c]
                  + (1 − alpha[q]) * bm25_scores[q, c]
```

During inference, alpha is grid-searched (0.0 to 1.0 in steps of 0.1) on the validation set and frozen for test evaluation.

---

## Step 5 — Training

### Dataset and Negative Sampling

Each training sample is a (query, positive doc, 299 negative docs) triple. Negatives are sampled randomly from the corpus.

```
batch_size = 256
num_negatives = 299    # 1 positive + 299 negatives per sample
```

### Loss Function

```
CrossEntropyLoss(reduction="none")

labels[b] = 0   # first candidate is always the positive

Per-query loss weighted by IPS (Inverse Propensity Score):
  ips_weight[q] = 1.0 / log(1 + doc_freq[positive_doc])
  doc_freq[d] = number of queries for which d is relevant

total_loss = mean(ips_weight * cross_entropy_loss)
```

IPS down-weights queries whose positive is a frequently-cited document, preventing the model from over-fitting to common citations.

### Optimiser and Schedule

```
Optimiser: AdamW, lr=1e-4
Scheduler: linear warmup (10% of steps) then constant
Total training steps: epochs * n_batches * 1.2  (slight overshoot)
```

### Early Stopping

Validation MRR@10 is computed after every epoch using pure GNN scores (no blending). If no improvement for **10 consecutive epochs**, training stops. Best checkpoint is saved to `best_model.pt`.

### Post-training Alpha Selection

After training, alpha is selected by grid search on the validation set:

1. Try α ∈ {0.0, 0.1, …, 1.0}
2. Compute `final_scores = α * gnn_scores + (1−α) * bm25_scores`
3. Also try debiased: `α * (gnn_scores − mean) + (1−α) * bm25_scores`
4. Pick α maximising val MRR@10
5. Apply frozen α to test set — test-optimal α is logged but NOT used for selection

---

## Step 6 — Inference

### Batch Inference (full test set)

```
1. Load best_model.pt → TestCaseGnn
2. Build single DGL graph: all test queries + all corpus docs
3. EUGAT forward pass → (N_queries, 1024) and (N_docs, 1024)
4. GNN scores = query_encoded @ candidate_encoded.T
5. Z-score normalise
6. Grid search alpha on validation set
7. Apply frozen alpha: final_scores = α * gnn + (1−α) * bm25
8. Top-K per query → rankings.jsonl
```

### Online Inference (per-query, via StructGNNSearcher)

For integration with the agentic harness:

```
Pre-encoding (once):
  corpus_embeddings = EUGAT_encode(all docs)   # (N_docs, 1024)

Per query:
  1. BGE-M3 encode query sentences → para embeddings
  2. Build tiny 1-doc graph (query doc + para nodes)
  3. EUGAT forward on tiny graph → query_encoded (1024,)
  4. gnn_scores = query_encoded @ corpus_embeddings.T
  5. gnn_scores = (gnn_scores - mean) / (std + 1e-8)
  6. bm25_scores = bm25.transform(query)
  7. final_scores = alpha * gnn_scores + (1-alpha) * bm25_scores
  8. Return top-N by final_scores
```

Pre-encoded corpus embeddings are stored in `gnn_corpus_embeddings.npy`.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 1024 (BGE-M3) |
| EUGAT hidden dim | 1024 |
| EUGAT output dim | 1024 |
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
| `src/paragnn/__init__.py` | Config, dataset metadata, RR/fact-type label definitions |
| `src/paragnn/model.py` | CaseGnn (train), TestCaseGnn (inference), alpha FFNN |
| `src/paragnn/eugat.py` | EUGATConv, EUGATGNN (2-layer graph encoder) |
| `src/paragnn/trainer.py` | Training loop, validation, early stopping, alpha selection |
| `src/paragnn/graph_builder.py` | ParagraphStore, GraphBuilder (DGL graph construction) |
| `src/paragnn/dataset.py` | ParaGNNDataset, ParaGNNCollator (training batches) |
| `src/paragnn/precompute.py` | BGE-M3 encoding, BM25 scoring, RR label embeddings |
| `src/paragnn/inference.py` | Batch inference on test set |
| `src/paragnn/gnn_searcher.py` | Online per-query inference searcher |
| `src/evaluate_paragnn.py` | Training entry point |
| `src/inference/infer_paragnn.py` | Inference entry point |
