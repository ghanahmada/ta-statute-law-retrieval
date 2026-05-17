# GNN Pipeline: Train → Inference → Analysis

StructGNN with ContraNorm 0.8 + enriched ground truth.
Covers all datasets: `kuhperdata-exp`, `kuhperdata-summ-exp`, `bsard`, `stard`, `coliee`, `ilpcsr`.

---

## 0. Prerequisites

Requires outputs from FULL-RUN-PIPELINE.md steps 1–4 (datasets prepared, BM25 + Dense evaluated).

```bash
conda activate paragnn
```

---

## 1. Precompute (BM25 scores + BGE-M3 embeddings)

Generates per-document and per-query embeddings, BM25 score matrices, hard negatives.
Must run before training.

```bash
python src/paragnn/precompute.py --dataset kuhperdata-exp      --max_relevant 0
python src/paragnn/precompute.py --dataset kuhperdata-summ-exp --max_relevant 0
python src/paragnn/precompute.py --dataset bsard               --max_relevant 0
python src/paragnn/precompute.py --dataset stard               --max_relevant 0
python src/paragnn/precompute.py --dataset coliee              --max_relevant 0
python src/paragnn/precompute.py --dataset ilpcsr              --max_relevant 0
```

Outputs: `outputs/paragnn/{dataset}/bm25_*.pt`, `embeddings/`, `*_query_ids.json`, `corpus_doc_ids.json`

---

## 2. Train StructGNN + ContraNorm 0.8

Uses `--tag enriched` to save checkpoints separately from base runs.

```bash
# KUHPerdata (default batch_size=256, num_negatives=299)
python src/evaluate_paragnn.py --dataset kuhperdata-exp      --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --tag enriched
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --tag enriched

# Large corpora (reduced batch to avoid OOM)
python src/evaluate_paragnn.py --dataset bsard  --structure_mode structural --max_relevant 0 --batch_size 64 --num_negatives 99 --contranorm_scale 0.8 --tag enriched
python src/evaluate_paragnn.py --dataset stard  --structure_mode structural --max_relevant 0 --batch_size 64 --num_negatives 99 --contranorm_scale 0.8 --tag enriched
python src/evaluate_paragnn.py --dataset coliee --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --tag enriched
python src/evaluate_paragnn.py --dataset ilpcsr --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --tag enriched
```

Outputs: `outputs/paragnn/{dataset}/adapted_struct_cn0.8_enriched/best_model.pt`

---

## 3. Inference (alpha grid search + export predictions + embeddings)

Loads trained checkpoint, sweeps alpha on val, reports test Recall@10 with frozen alpha.

```bash
python src/paragnn/inference.py --dataset kuhperdata-exp      --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/kuhperdata-exp/adapted_struct_cn0.8_enriched      --export_embeddings
python src/paragnn/inference.py --dataset kuhperdata-summ-exp --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/kuhperdata-summ-exp/adapted_struct_cn0.8_enriched --export_embeddings
python src/paragnn/inference.py --dataset bsard               --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/bsard/adapted_struct_cn0.8_enriched               --export_embeddings
python src/paragnn/inference.py --dataset stard               --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/stard/adapted_struct_cn0.8_enriched               --export_embeddings
python src/paragnn/inference.py --dataset coliee              --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/coliee/adapted_struct_cn0.8_enriched              --export_embeddings
python src/paragnn/inference.py --dataset ilpcsr              --structure_mode structural --max_relevant 0 --contranorm_scale 0.8 --model_dir outputs/paragnn/ilpcsr/adapted_struct_cn0.8_enriched              --export_embeddings
```

Outputs per dataset:
- `outputs/predictions/structgnn_cn0.8_{dataset}.jsonl` — standard predictions
- `{model_dir}/rankings_top100.jsonl` — full rankings with scores
- `{model_dir}/gnn_corpus_embeddings.npy` — GNN-adapted corpus embeddings (for analysis + agentic)

---

## 4. Agentic + StructGNN (kuhperdata only)

Uses GNN corpus embeddings as the dense component in hybrid search.
Requires vLLM running in a separate terminal (see FULL-RUN-PIPELINE.md §10).

```bash
python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --dense_source structgnn \
  --gnn_model_dir outputs/paragnn/kuhperdata-exp/adapted_struct_cn0.8_enriched \
  --gnn_alpha 0.8 \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda

python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-summ-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --dense_source structgnn \
  --gnn_model_dir outputs/paragnn/kuhperdata-summ-exp/adapted_struct_cn0.8_enriched \
  --gnn_alpha 0.8 \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda
```

---

## 5. Check Predictions

```bash
python src/analysis/dump_predictions_metrics.py
```

---

## 6. Analysis

### 6a. Embedding Analysis

All analyses read from `outputs/paragnn/{dataset}/{model_type}/gnn_corpus_embeddings.npy`.

```bash
# Collapse check (SVD decay, effective rank, isotropy)
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis collapse --model_type adapted_struct_cn0.8_enriched

# Similarity distributions (co-relevant vs hard-negative cosine histograms)
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis similarity --model_type adapted_struct_cn0.8_enriched

# Neighborhood overlap (k-NN hit rate for co-relevant articles)
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis neighborhood --model_type adapted_struct_cn0.8_enriched

# Separation (Cohen's d, AUC between co-relevant vs hard-negative)
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis separation --model_type adapted_struct_cn0.8_enriched

# Before/after (StructGNN vs ParaGNN movement — needs adapted/ embeddings too)
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis before_after --model_type adapted_struct_cn0.8_enriched

# Run all at once
python src/analysis/embedding_analysis.py --dataset kuhperdata-exp --analysis all --model_type adapted_struct_cn0.8_enriched
```

Outputs: `outputs/analysis/embedding_analysis/` — JSON results + PNG plots

### 6b. GNN Structure Explainability

Requires both ParaGNN and StructGNN embeddings for full comparison.

```bash
# Export ParaGNN embeddings first (if not already done)
python src/paragnn/inference.py --dataset kuhperdata-exp --structure_mode none --max_relevant 0 --export_embeddings

# Run explainability (RQ1: proximity, RQ2: hub bias, RQ3: non-hub signal, RQ4: score decomposition)
python src/analysis/gnn_structure_explainability.py \
  --dataset kuhperdata-exp \
  --paragnn_emb outputs/paragnn/kuhperdata-exp/adapted/gnn_corpus_embeddings.npy \
  --structgnn_emb outputs/paragnn/kuhperdata-exp/adapted_struct_cn0.8_enriched/gnn_corpus_embeddings.npy \
  --split test
```

Outputs: `outputs/analysis/gnn_explainability/{dataset}/` — PNG plots per RQ

### 6c. Hub Bias Diagnosis

```bash
python src/analysis/diagnose_hub_bias.py --dataset kuhperdata-exp --structure_mode structural --max_relevant 0
```

### 6d. Thesis-Level Analyses (problem motivation)

These don't depend on GNN — run independently:

```bash
# Vocabulary gap quantification
python src/analysis/vocab_gap_analysis.py --dataset kuhperdata-exp

# Structural co-relevance clustering
python src/analysis/structural_corelevance.py --dataset kuhperdata-exp

# Gap × structure coupling (does structure help high vocab-gap queries?)
python src/analysis/gap_structure_coupling.py --dataset kuhperdata-exp
```

---

## Output Directory Map

```
outputs/
├── paragnn/{dataset}/
│   ├── bm25_*.pt, *_query_ids.json, corpus_doc_ids.json   (precompute)
│   ├── embeddings/corpus/*.pt, queries/*.pt                (precompute)
│   ├── adapted_struct_cn0.8_enriched/
│   │   ├── best_model.pt                                   (training)
│   │   ├── rankings_top100.jsonl                            (inference)
│   │   └── gnn_corpus_embeddings.npy                       (inference --export_embeddings)
│   └── adapted/                                            (ParaGNN baseline, same structure)
├── predictions/
│   ├── structgnn_cn0.8_{dataset}.jsonl                     (inference)
│   └── paragnn_{dataset}.jsonl                             (ParaGNN inference)
├── analysis/
│   ├── embedding_analysis/                                 (6a)
│   └── gnn_explainability/{dataset}/                       (6b)
└── context_1/{dataset}_structgnn/                          (agentic + structgnn)
```
