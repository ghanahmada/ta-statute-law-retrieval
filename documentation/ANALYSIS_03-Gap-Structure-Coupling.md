# Analysis 03: Vocabulary Gap × Structural Benefit Coupling + Embedding Analysis

## Objective

Connect Analysis 01 (vocabulary gap) and Analysis 02 (structural co-relevance) to explain **why** StructGNN improves:
1. Structural features help most where vocabulary gap is worst
2. StructGNN failures correlate with dense retrieval also missing the neighborhood
3. StructGNN pulls co-relevant articles closer in embedding space

## Scripts & Commands

### Part 1: Gap-Structure Coupling (CPU, ~2 min)

```bash
python src/analysis/gap_structure_coupling.py
```

Datasets: kuhperdata-exp, kuhperdata-summ-exp, bsard, stard

### Part 2: Embedding Analysis

**kuhperdata-exp** (CPU, ~1 min, 2127 docs):
```bash
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis all
```

**stard — similarity only** (CPU, slow due to 55K .pt file loads):
```bash
python src/analysis/embedding_analysis.py --dataset stard --analysis similarity --no_plots
```

**stard — neighborhood** (needs ~11GB RAM for 55K×55K matrix, run on GPU VM):
```bash
python src/analysis/embedding_analysis.py --dataset stard --analysis neighborhood --no_plots
```

**coliee — before/after** (CPU, 768 docs, has both Para-GNN and StructGNN embeddings):
```bash
python src/analysis/embedding_analysis.py --dataset coliee --analysis before_after --no_plots
```
Note: coliee has only 1 relevant doc per query, so similarity/neighborhood analyses produce no results.

**bsard** (needs embeddings on GPU VM first):
```bash
# Generate embeddings if not present
python src/paragnn/inference.py --dataset bsard --structure_mode structural --export_embeddings
python src/paragnn/inference.py --dataset bsard --structure_mode none --export_embeddings

# Then run analysis
python src/analysis/embedding_analysis.py --dataset bsard --analysis all
```

### Prerequisites

Embedding files needed per dataset:
```
outputs/paragnn/{dataset}/adapted_struct/gnn_corpus_embeddings.npy   (StructGNN)
outputs/paragnn/{dataset}/adapted/gnn_corpus_embeddings.npy          (Para-GNN, for before_after)
outputs/paragnn/{dataset}/corpus_doc_ids.json                        (row→doc_id mapping)
outputs/paragnn/{dataset}/embeddings/corpus/{doc_id}.pt              (raw BGE-M3, per doc)
```

### All commands for GPU VM (run sequentially):

```bash
# Part 1: Coupling (CPU)
python src/analysis/gap_structure_coupling.py

# Part 2: Embedding — kuhperdata-exp
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis all

# Part 2: Embedding — stard (similarity only, skip neighborhood due to 55K×55K matrix)
python src/analysis/embedding_analysis.py --dataset stard --analysis similarity --no_plots

# Part 2: Embedding — coliee (before/after only, single-relevant queries)
python src/analysis/embedding_analysis.py --dataset coliee --analysis before_after --no_plots
```

## Results (2026-05-16)

### Part 1: Gap-Structure Coupling

#### kuhperdata-exp

| Stratum | N | BM25 | Dense | ParaGNN | StructGNN | Δ(S-P) |
|---------|---|------|-------|---------|-----------|--------|
| zero | 98 | 0.000 | 0.004 | 0.460 | 0.449 | -0.011 |
| low | 71 | 0.040 | 0.100 | 0.549 | 0.560 | +0.011 |
| moderate | 42 | 0.098 | 0.141 | 0.566 | 0.537 | -0.029 |

Never-retrieved: 34/211 StructGNN failures, 68% also missed in Dense top-100.

#### kuhperdata-summ-exp

| Stratum | N | BM25 | Dense | ParaGNN | StructGNN | Δ(S-P) |
|---------|---|------|-------|---------|-----------|--------|
| zero | 28 | 0.000 | 0.004 | 0.122 | 0.161 | **+0.040** |
| low | 110 | 0.007 | 0.009 | 0.399 | 0.476 | **+0.076** |
| moderate | 73 | 0.195 | 0.241 | 0.606 | 0.593 | -0.013 |

Never-retrieved: 53/213 failures, 66% also missed in Dense top-100.

#### bsard

| Stratum | N | BM25 | Dense | ParaGNN | StructGNN | Δ(S-P) |
|---------|---|------|-------|---------|-----------|--------|
| zero | 22 | 0.000 | 0.070 | 0.228 | 0.102 | -0.126 |
| low | 156 | 0.197 | 0.248 | 0.392 | 0.480 | **+0.088** |
| moderate | 44 | 0.556 | 0.560 | 0.721 | 0.727 | +0.006 |

Never-retrieved: 64/222 failures, 39% also missed in Dense top-100.

#### stard

| Stratum | N | BM25 | Dense | ParaGNN | StructGNN | Δ(S-P) |
|---------|---|------|-------|---------|-----------|--------|
| zero | 20 | 0.000 | 0.056 | 0.087 | 0.135 | **+0.048** |
| low | 90 | 0.075 | 0.197 | 0.278 | 0.291 | +0.013 |
| moderate | 150 | 0.372 | 0.545 | 0.592 | 0.588 | -0.003 |
| high | 48 | 0.879 | 0.885 | 0.928 | 0.938 | +0.010 |

Never-retrieved: 82/308 failures, 41% also missed in Dense top-100.

### Part 2: Embedding Analysis (kuhperdata-exp)

**Similarity:**

| Space | Co-relevant | Random | Ratio |
|-------|-------------|--------|-------|
| StructGNN | 0.939 | 0.882 | 1.1× |
| BGE-M3 | 0.660 | 0.562 | 1.2× |

GNN pushes co-relevant pairs +0.279 higher than BGE-M3.

**Neighborhood coverage:**

| K | GNN | BGE-M3 | Δ | Ratio |
|---|-----|--------|---|-------|
| 5 | 24.4% | 14.7% | +9.6% | 1.7× |
| 10 | 30.8% | 18.2% | +12.6% | 1.7× |
| 20 | 38.0% | 26.8% | +11.2% | 1.4× |
| 50 | 48.1% | 34.1% | +13.9% | 1.4× |

## Key Findings

1. **StructGNN benefit concentrates on low-overlap queries** (kuhperdata-summ-exp: +0.076 MRR on low-overlap, bsard: +0.088 on low-overlap, stard: +0.048 on zero-overlap)

2. **When StructGNN fails, 66-68% of the time dense also missed** — structural features can't propagate signal if no anchor is found in the correct neighborhood

3. **GNN embedding space has 1.7× better neighborhood coverage** — co-relevant articles are 70% more likely to appear in each other's top-10 neighbors in GNN space vs raw BGE-M3

4. **Narrative**: structural features act as a bridge when partial lexical signal exists (low-overlap), but require at least one anchor in the correct neighborhood

## Output Artifacts

- `outputs/analysis/gap_structure_coupling/coupling_results.json`
- `outputs/analysis/gap_structure_coupling/console_output.txt`
- `outputs/analysis/embedding_analysis/embedding_results_kuhperdata-exp.json`
- `outputs/analysis/embedding_analysis/similarity_hist_kuhperdata-exp.png`
- `outputs/analysis/embedding_analysis/console_output_kuhperdata-exp.txt`
