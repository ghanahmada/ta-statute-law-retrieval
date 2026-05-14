# Full Experiment Rerun Pipeline

All retrieval baselines across 6 datasets: `kuhperdata-exp`, `kuhperdata-summ-exp`, `bsard`, `stard`, `coliee`, `ilpcsr`.
Run on GPU VPS. Commands verified against actual argparse interfaces.

---

## VM Requirements

| Requirement | Spec | Notes |
|-------------|------|-------|
| **GPU** | RTX A6000 (Ada Lovelace, SM 8.9) | Do NOT use Blackwell (SM 10.x) — DGL is incompatible with Blackwell architecture |
| **CUDA driver** | ≥ 13.0 | Required by vLLM 0.20.2; CUDA 13.0 drivers are backward-compatible so DGL (compiled for CUDA 12.4) will still work |
| **CUDA toolkit** | 12.4 | For DGL and PyTorch compilation |
| **RAM** | ≥ 48 GB | vLLM + BGE-M3 encoder running concurrently |
| **Disk** | ≥ 100 GB | Model weights + dataset + embeddings + outputs |

> **CUDA compatibility note:** A VM provisioned with CUDA 13.0 drivers can run code compiled against CUDA 12.4 (backward-compatible). So a single VM satisfies both requirements — just ensure the CUDA *driver* version is ≥ 13.0 and the CUDA *toolkit* (nvcc) is 12.4.

---

## 0. Setup (once)

```bash
conda activate paragnn
pip install openai tiktoken tqdm
export HF_TOKEN=<your_token>
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 1. Pull / Prepare Datasets

```bash
python src/scripts/prepare_kuhperdata.py
python src/scripts/prepare_bsard.py
python src/scripts/prepare_stard.py
python src/scripts/import_coliee.py
python src/scripts/prepare_ilpcsr.py
```

> `import_coliee.py` downloads the `coliee` config from `ghanahmada/kuhperdata` on HuggingFace.
> `prepare_ilpcsr.py` downloads from HuggingFace — may take longer due to large document size.

---

## 2. Train/Val/Test Split (Para-GNN + StructGNN only)

Val is carved from train; `qrels_test.tsv` is never modified.

```bash
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-exp      --val_scale 0.5
python src/scripts/split_test_to_val.py --dataset_dir data/kuhperdata-summ-exp --val_scale 0.5
python src/scripts/split_test_to_val.py --dataset_dir data/bsard               --val_scale 0.5
python src/scripts/split_test_to_val.py --dataset_dir data/stard               --val_scale 0.5
python src/scripts/split_test_to_val.py --dataset_dir data/coliee              --val_scale 0.5
python src/scripts/split_test_to_val.py --dataset_dir data/ilpcsr              --val_scale 0.5
```

---

## 3. BM25

```bash
python src/evaluate_bm25.py --dataset kuhperdata-exp      --split test --max_relevant 0
python src/evaluate_bm25.py --dataset kuhperdata-summ-exp --split test --max_relevant 0
python src/evaluate_bm25.py --dataset bsard               --split test --max_relevant 0
python src/evaluate_bm25.py --dataset stard               --split test --max_relevant 0
python src/evaluate_bm25.py --dataset coliee              --split test --max_relevant 0
python src/evaluate_bm25.py --dataset ilpcsr              --split test --max_relevant 0
```

---

## 4. BGE-M3 Dense

```bash
python src/evaluate_dense_retrieval.py --dataset kuhperdata-exp      --split test --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset kuhperdata-summ-exp --split test --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset bsard               --split test --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset stard               --split test --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset coliee              --split test --max_relevant 0 --save_embeddings
python src/evaluate_dense_retrieval.py --dataset ilpcsr              --split test --max_relevant 0 --save_embeddings --max_length 8192 --batch_size 8
```

---

## 5. JNLP Stage 1

> `bsard` and `kuhperdata-summ-exp` require `--batch_size 16` to avoid FlagEmbedding OOM crash (batch shrink bug).

```bash
python src/evaluate_jnlp.py --dataset kuhperdata-exp      --stage 1 --feature_type product --max_relevant 0
python src/evaluate_jnlp.py --dataset kuhperdata-summ-exp --stage 1 --feature_type product --max_relevant 0 --batch_size 16
python src/evaluate_jnlp.py --dataset bsard               --stage 1 --feature_type product --max_relevant 0 --batch_size 16
python src/evaluate_jnlp.py --dataset stard               --stage 1 --feature_type product --max_relevant 0
python src/evaluate_jnlp.py --dataset coliee              --stage 1 --feature_type product --max_relevant 0
python src/evaluate_jnlp.py --dataset ilpcsr              --stage 1 --feature_type product --max_relevant 0
```

---

## 6. GAR / Rerank

```bash
# GAR
python src/evaluate_gar.py --dataset kuhperdata-exp      --scorer bge --max_relevant 0
python src/evaluate_gar.py --dataset kuhperdata-summ-exp --scorer bge --max_relevant 0
python src/evaluate_gar.py --dataset bsard               --scorer bge --max_relevant 0
python src/evaluate_gar.py --dataset stard               --scorer bge --max_relevant 0
python src/evaluate_gar.py --dataset coliee              --scorer bge --max_relevant 0
python src/evaluate_gar.py --dataset ilpcsr              --scorer bge --max_relevant 0

# BM25 + Reranker
python src/evaluate_rerank.py --dataset kuhperdata-exp      --scorer bge --max_relevant 0
python src/evaluate_rerank.py --dataset kuhperdata-summ-exp --scorer bge --max_relevant 0
python src/evaluate_rerank.py --dataset bsard               --scorer bge --max_relevant 0
python src/evaluate_rerank.py --dataset stard               --scorer bge --max_relevant 0
python src/evaluate_rerank.py --dataset coliee              --scorer bge --max_relevant 0
python src/evaluate_rerank.py --dataset ilpcsr              --scorer bge --max_relevant 0
```

---

## 7. Para-GNN + StructGNN

> **Large-corpus datasets** (`bsard` ~22k docs, `stard` ~55k docs) require reduced batch size to avoid OOM.
> Default params work for `kuhperdata-*` (~2k docs), `coliee` (~768 docs), and `ilpcsr`.

```bash
# Precompute BM25 scores + embeddings (reads qrels_val.tsv automatically)
python src/paragnn/precompute.py --dataset kuhperdata-exp      --max_relevant 0
python src/paragnn/precompute.py --dataset kuhperdata-summ-exp --max_relevant 0
python src/paragnn/precompute.py --dataset bsard               --max_relevant 0
python src/paragnn/precompute.py --dataset stard               --max_relevant 0
python src/paragnn/precompute.py --dataset coliee              --max_relevant 0
python src/paragnn/precompute.py --dataset ilpcsr              --max_relevant 0

# Para-GNN (no structural features)
python src/evaluate_paragnn.py --dataset kuhperdata-exp      --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset bsard               --structure_mode none --max_relevant 0 --batch_size 64 --num_negatives 99
python src/evaluate_paragnn.py --dataset stard               --structure_mode none --max_relevant 0 --batch_size 64 --num_negatives 99
python src/evaluate_paragnn.py --dataset coliee              --structure_mode none --max_relevant 0
python src/evaluate_paragnn.py --dataset ilpcsr              --structure_mode none --max_relevant 0

# StructGNN (structural node features)
python src/evaluate_paragnn.py --dataset kuhperdata-exp      --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset kuhperdata-summ-exp --structure_mode structural --max_relevant 0 
python src/evaluate_paragnn.py --dataset bsard               --structure_mode structural --max_relevant 0 --batch_size 64 --num_negatives 99
python src/evaluate_paragnn.py --dataset stard               --structure_mode structural --max_relevant 0 --batch_size 64 --num_negatives 99
python src/evaluate_paragnn.py --dataset coliee              --structure_mode structural --max_relevant 0
python src/evaluate_paragnn.py --dataset ilpcsr              --structure_mode structural --max_relevant 0

# Export StructGNN corpus embeddings (required for Step 8 Agentic+StructGNN, kuhperdata only)
python src/paragnn/inference.py --dataset kuhperdata-exp      --structure_mode structural --export_embeddings --max_relevant 0
python src/paragnn/inference.py --dataset kuhperdata-summ-exp --structure_mode structural --export_embeddings --max_relevant 0
python src/paragnn/inference.py --dataset coliee      --structure_mode structural --export_embeddings --max_relevant 0
```

> Predictions saved automatically to `outputs/predictions/{method}_{dataset}.jsonl`.
> Model checkpoints saved to `outputs/paragnn/{dataset}/adapted_{none→base,struct}/`.

---

## 8. Agentic Retrieval (kuhperdata only)

> **Note:** `--served-model-name` in vLLM must match `--model` in client calls exactly.
> vLLM at 0.85 utilization leaves room for BGE-M3 encoder on the same GPU.

```bash
# Terminal 1 — start vLLM
vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000

# Terminal 2
conda activate paragnn

# Sanity check (single query)
python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --pad_to_k 10 --encoder_device cuda --debug_qid q324

# Agentic + BGE-M3
python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda

python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-summ-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda

# Agentic + StructGNN (run Step 7 export first)
python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --dense_source structgnn \
  --gnn_model_dir outputs/paragnn/kuhperdata-exp/adapted_struct \
  --gnn_alpha 0.8 \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda

python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-summ-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --dense_source structgnn \
  --gnn_model_dir outputs/paragnn/kuhperdata-summ-exp/adapted_struct \
  --gnn_alpha 0.8 \
  --max_relevant 0 --pad_to_k 10 --concurrency 4 --encoder_device cuda
```

---

## 9. Ablation — Agentic Flat (all supported datasets)

Disables hierarchy, coverage gate, and similarity guard. `ilpcsr` and `coliee` not supported by context_1.

```bash
python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --max_turns 5 --concurrency 16 --pad_to_k 10 \
  --no_hierarchy --no_coverage_gate --no_similarity_guard \
  --output_dir outputs/context_1/kuhperdata-exp_flat

python src/context_1/evaluate_context1.py \
  --dataset kuhperdata-summ-exp \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --max_turns 5 --concurrency 16 --pad_to_k 10 \
  --no_hierarchy --no_coverage_gate --no_similarity_guard \
  --output_dir outputs/context_1/kuhperdata-summ-exp_flat

python src/context_1/evaluate_context1.py \
  --dataset bsard \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --max_turns 5 --concurrency 16 --pad_to_k 10 \
  --no_hierarchy --no_coverage_gate --no_similarity_guard \
  --output_dir outputs/context_1/bsard_flat

python src/context_1/evaluate_context1.py \
  --dataset stard \
  --base_url http://localhost:8000/v1 --model qwen3.5-9b \
  --max_relevant 0 --max_turns 5 --concurrency 16 --pad_to_k 10 \
  --no_hierarchy --no_coverage_gate --no_similarity_guard \
  --output_dir outputs/context_1/stard_flat
```

---

## Split Logic Recap

| Method | Split used | `qrels_test.tsv` modified? |
|--------|------------|---------------------------|
| BM25, Dense, JNLP, GAR, Rerank, Agentic | full original test | Never |
| Para-GNN / StructGNN precompute + train | `qrels_val.tsv` (carved from train) | Never |
| Para-GNN / StructGNN final eval | full original test | Never |

Alpha (α) tuned on held-out val; test results reported with frozen α.
Per-epoch early stopping tracks **blended val MRR** (GNN + BM25), not pure GNN MRR.
