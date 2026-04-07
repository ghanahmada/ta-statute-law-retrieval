# Para-GNN Evaluation Results

Adapted from IL-PCSR (Paul et al., EMNLP 2025) for multilingual statute retrieval.

---

## Method Overview

Para-GNN builds a graph per query-document pair:
- **Document nodes**: query + statute candidates (embedding = mean of paragraph embeddings)
- **Paragraph nodes**: per-sentence embeddings (BGE-M3, 1024d)
- **Edges**: paragraph → parent document, with rhetorical role embedding as edge feature
- **GNN**: 2-layer Edge-Updated Graph Attention Network (EUGAT)
- **Scoring**: `alpha × GNN_score + (1-alpha) × BM25_score`

Two variants:
- **Adapted**: query as single paragraph (no LLM needed)
- **Full**: query split into sentences, each labeled with rhetorical role by LLM (Qwen 3.5 9B)

---

## What Actually Improved Performance (Step-by-Step)

### 1. Base Para-GNN with learned alpha (first run)
Directly adapted from IL-PCSR with BGE-M3 (1024d). The learned alpha converged to 0.95-0.97 (nearly pure GNN, almost no BM25). MRR peaked early then oscillated.

| Dataset | MRR@10 | Issue |
|---|---|---|
| KUHPerdata-humanized | 0.431 | Alpha too high, lost BM25 signal after epoch 5 |
| KUHPerdata-summarized | 0.339 | Same issue, MRR peaked at epoch 4 then dropped |

**Problem identified**: The FFNN learning alpha pushes it to ~0.97, but the GNN alone isn't precise enough. The 3% BM25 signal isn't enough for lexical grounding.

### 2. Added IL-PCSR LR schedule (linear warmup + 1.2x total steps)
Changed from cosine annealing to IL-PCSR's original `get_linear_schedule_with_warmup` with `num_training_steps = total_steps * 1.2`. This means the LR never fully decays to 0, maintaining a minimum learning rate throughout training.

Also added: OOM handling per batch, `torch.nan_to_num` for BM25 scores, `torch.cuda.empty_cache()` per batch.

**Impact**: Training more stable, but alpha still too high.

### 3. Post-training alpha grid search (the key improvement)
Instead of using the learned alpha (0.97), sweep alpha from 0.0 to 1.0 in steps of 0.1 after each epoch. Pure arithmetic on pre-computed score matrices — takes <1 second.

This decouples "how good is the GNN" from "how much BM25 to blend." The GNN trains freely, then the optimal blend is found post-hoc.

**Impact**: Massive improvement. The grid search finds that 10-30% BM25 is optimal depending on dataset, vs the learned 3%.

### 4. Early stopping (patience=10)
Stop training if MRR doesn't improve for 10 consecutive epochs. Prevents overfitting and wasted compute.

---

## Results: Para-GNN Adapted (no LLM)

All results use post-training alpha grid search.

### KUHPerdata-humanized

| Alpha | MRR@10 | R@10 | Hit |
|---|---|---|---|
| 0.0 (pure BM25) | 0.045 | 0.067 | 10.7% |
| 0.7 | 0.372 | 0.472 | 62.1% |
| 0.8 | 0.448 | 0.544 | 68.3% |
| **0.9** | **0.456** | **0.542** | **67.8%** |
| 1.0 (pure GNN) | 0.427 | 0.541 | 67.6% |

**Best: alpha=0.9, MRR=0.456** (epoch 14, early stop epoch 24)

### KUHPerdata-summarized

| Alpha | MRR@10 | R@10 | Hit |
|---|---|---|---|
| 0.0 (pure BM25) | 0.077 | 0.099 | 17.7% |
| 0.7 | 0.324 | 0.347 | 55.0% |
| **0.8** | **0.458** | **0.522** | **71.1%** |
| 0.9 | 0.358 | 0.503 | 70.1% |
| 1.0 (pure GNN) | 0.215 | 0.417 | 61.9% |

**Best: alpha=0.8, MRR=0.458** (epoch 27, early stop epoch 37)

### BSARD (French)

| Alpha | MRR@10 | R@10 | Hit |
|---|---|---|---|
| 0.0 (pure BM25) | 0.249 | 0.266 | 42.3% |
| 0.5 | 0.390 | 0.378 | 57.7% |
| 0.6 | 0.448 | 0.428 | 61.3% |
| **0.7** | **0.493** | **0.487** | **68.5%** |
| 0.8 | 0.436 | 0.476 | 66.2% |
| 1.0 (pure GNN) | 0.147 | 0.188 | 33.8% |

**Best: alpha=0.7, MRR=0.493** (epoch 28, early stop epoch 38)

Note: BSARD needs 30% BM25 — the highest among all datasets. French citizen questions have more lexical overlap with statutes (42.3% BM25 hit rate), so BM25 carries more useful signal. Pure GNN (alpha=1.0) collapses to 0.147, confirming that BM25 is essential for this dataset.

### STARD (Chinese)

| Alpha | MRR@10 | R@10 | Hit |
|---|---|---|---|
| 0.0 (pure BM25) | 0.341 | 0.427 | 53.2% |
| 0.6 | 0.469 | 0.561 | 66.9% |
| 0.7 | 0.510 | 0.600 | 69.8% |
| **0.8** | **0.527** | **0.617** | **72.4%** |
| 0.9 | 0.429 | 0.548 | 65.6% |
| 1.0 (pure GNN) | 0.143 | 0.214 | 29.2% |

**Best: alpha=0.8, MRR=0.527** (epoch 7, early stop epoch 17)

Note: Pure GNN (alpha=1.0) collapses to 0.143 on STARD — the worst pure-GNN score across all datasets. STARD's 55K statute corpus makes the ranking problem much harder; BM25's lexical signal is critical for narrowing candidates.

---

## Results: Para-GNN Full (with LLM rhetorical roles)

Queries labeled by Qwen 3.5 9B with roles: Facts, Issue, Argument, Court Reasoning, etc.
These results use the cosine LR schedule (before grid search was implemented).

### KUHPerdata-humanized

**Best: MRR=0.441** (epoch 12, early stop epoch 22)

Note: Humanized queries are single sentences, so RR labeling adds minimal signal (all labeled "Issue"). Full method is similar to adapted on humanized.

### KUHPerdata-summarized

**Best: MRR=0.367** (epoch 5, early stop epoch 15)

Note: Summarized queries have ~1.5 sentences avg with diverse roles (Facts, Issue, Court Reasoning). RR labels provide some benefit over adapted (+0.028 MRR).

TODO: Re-run full method with the grid search alpha trainer for proper comparison.

---

## Comparison with Baselines

| Dataset | BM25 | Dense (BGE-M3) | JNLP S1 | **Para-GNN adapted** | Delta vs JNLP |
|---|---|---|---|---|---|
| KUHPerdata-humanized | 0.045 | 0.091 | 0.367 | **0.456** | **+0.089 (+24%)** |
| KUHPerdata-summarized | 0.077 | 0.065 | 0.337 | **0.458** | **+0.122 (+36%)** |
| BSARD | 0.249 | xxx | 0.328 | **0.493** | **+0.165 (+50%)** |
| STARD | 0.338 | xxx | 0.271 | **0.527** | **+0.256 (+94%)** |

---

## Key Insight: Optimal Alpha Correlates with Lexical Overlap

| Dataset | BM25 Hit Rate | Optimal Alpha | BM25 Weight |
|---|---|---|---|
| KUHPerdata-humanized | 10.7% | 0.9 | 10% |
| KUHPerdata-summarized | 17.7% | 0.8 | 20% |
| STARD | 53.2% | 0.8 | 20% |
| BSARD | 42.3% | 0.7 | 30% |

Datasets with more lexical overlap between queries and statutes benefit from higher BM25 weight. The grid search automatically discovers this per dataset. The learned alpha (~0.97) ignores this signal, which is why grid search outperforms.

---

---

## Results: Direction A — Statute Proximity Edges (prox=50)

Adding bidirectional edges between statute nodes within 50 articles of each other.
Inspired by G-DSR (Louis et al., EACL 2023). Empirically justified: co-relevant statutes have median distance of 18 articles (66% within 50).

### KUHPerdata-humanized

| Setting | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Adapted (no prox) | 0.456 | 0.542 | 67.8% | 0.9 |
| **Adapted + prox=50** | **0.486** | **0.551** | **68.3%** | 0.9 |
| Delta | **+0.030 (+6.4%)** | +0.009 | +0.5% | |

### KUHPerdata-summarized

| Setting | MRR@10 | R@10 | Hit | Alpha |
|---|---|---|---|---|
| Adapted (no prox) | 0.458 | 0.522 | 71.1% | 0.8 |
| **Adapted + prox=50** | **0.498** | **0.516** | **69.2%** | 0.9 |
| Delta | **+0.040 (+8.7%)** | -0.006 | -1.9% | |

### Key Observations

1. **MRR improves on both datasets** (+6-9%). The GNN uses statute proximity to rank the correct article higher.
2. **Alpha shifts from 0.8 to 0.9** on summarized — with proximity edges, the GNN becomes stronger and needs less BM25 support.
3. **Recall slightly drops on summarized** (-0.006) — proximity edges may cause the GNN to over-focus on nearby clusters and miss isolated relevant statutes.
4. **No new precomputation needed** — same embeddings and BM25 scores, only graph structure changes.

### TODO
- Run prox=50 on BSARD and STARD
- Test prox=20 for tighter connectivity comparison
- Test on full method (RR labels + proximity)

---

## Environment Setup

```bash
conda create -n paragnn python=3.11 -y
conda activate paragnn
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install FlagEmbedding==1.3.5 transformers==4.44.2 sentence-transformers
pip install numpy scipy scikit-learn tqdm jieba accelerate pydantic pyyaml
```

## Commands

```bash
# Pre-compute
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted

# Train + evaluate
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --method adapted --epochs 50

# Full method (requires vLLM for RR labeling)
vllm serve Qwen/Qwen3.5-9B-Instruct --max-model-len 32768
python experiment/label_rhetorical_roles.py --dataset kuhperdata-humanized
python src/paragnn/precompute.py --dataset kuhperdata-humanized --method full
python src/evaluate_paragnn.py --dataset kuhperdata-humanized --method full --epochs 50
```
