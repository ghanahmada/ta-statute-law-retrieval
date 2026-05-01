# Chroma Context-1: Training a Self-Editing Search Agent

**Source**: [Chroma Research - Context-1](https://www.trychroma.com/research/context-1)  
**Authors**: Bashir, Hong, Jiang, Shi (Chroma, March 2026)  
**Implementation**: `src/context_1/`

## Background & Motivation

Traditional retrieval systems operate in a **single pass**: given a query, they score all documents and return the top-k. This fails when answering a question requires **multi-hop reasoning** — connecting information spread across multiple documents with intermediate reasoning steps. A single embedding or keyword query cannot capture the full chain of evidence.

Context-1 addresses this by training a **20B-parameter retrieval subagent** that does not answer questions directly. Instead, it iteratively searches, reads, and curates a set of supporting documents for a downstream reasoning model. The key insight is that retrieval itself can be an agentic process: the model decomposes queries into subqueries, searches iteratively, and **self-edits its own context** (pruning irrelevant passages) to free capacity for further exploration.

The result is a model that matches frontier-scale models (Opus-4.6, Sonnet-4.5, GPT-5.2) on retrieval benchmarks at up to 10x faster inference and substantially lower cost.

## Core Methodology

### 1. Agentic Retrieval Loop

The agent operates in an **observe-reason-act** cycle with a bounded token budget:

```
Query → [Agent Loop (up to N turns)]
           ├→ Observe: receive query + token budget status
           ├→ Reason: LLM decides next action via chain-of-thought
           ├→ Act: execute a tool (search, grep, read, or prune)
           └→ Repeat until agent concludes or budget exhausted
        → Selected Documents (ranked)
```

**Four tools** are available to the agent:

| Tool | Purpose |
|------|---------|
| `search_corpus` | Hybrid BM25 + dense vector search via Reciprocal Rank Fusion (RRF) |
| `grep_corpus` | Regex pattern matching over the corpus |
| `read_document` | Retrieve full document text by ID |
| `prune_chunks` | Remove irrelevant passages from context to free token budget |

### 2. Token Budget Management

The agent has a fixed token budget (e.g., 32,768 tokens) with two thresholds:

- **Soft threshold (80%)**: agent is warned to consider pruning irrelevant documents
- **Hard threshold (92%)**: agent is restricted to only `prune_chunks` or concluding

This prevents context overflow while encouraging the agent to actively manage its working memory. Token usage is displayed after each turn so the agent can plan ahead.

### 3. Hybrid Search (BM25 + Dense + RRF)

Each `search_corpus` call combines two retrieval signals:

1. **BM25** — lexical/keyword scoring (good for exact term matches)
2. **Dense vectors** — semantic embeddings (good for meaning-level similarity)

Scores are fused using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

where `k` is a constant (default 60) and `rank_i(d)` is the rank of document `d` in the i-th ranking. This is a simple, effective way to combine rankings without needing score normalization.

An optional **cross-encoder reranker** can be applied on top of RRF results for higher precision.

### 4. Self-Editing Context (Pruning)

The `prune_chunks` tool allows the agent to remove previously retrieved but irrelevant document snippets from its conversation history. Pruned content is replaced with `[PRUNED]`, freeing tokens for new searches. This combats **context rot** — the degradation of downstream model performance when irrelevant context accumulates.

After RL training, prune accuracy improved from 0.824 to 0.941, meaning the agent learned to make better decisions about what to keep vs. discard.

## Training Pipeline

### Supervised Fine-Tuning (SFT)

Initial trajectories were generated using frontier models (Kimi K2.5) as inference backends. Training data included both successful and unsuccessful rollouts — lower-quality trajectories were kept for behavioral signals even when filtered by recall.

### Reinforcement Learning (CISPO)

After SFT, the model was trained with **CISPO** (Clipped Importance-Sampled Policy Optimization), a variant of GRPO:

- 1,024 agent trajectories per step across 128 queries
- 5 epochs, convergence around step 230

**Reward structure**:

| Component | Purpose |
|-----------|---------|
| Base F1 (16:1 recall weighting) | Prioritize finding all relevant docs over precision |
| Process recall credit | Reward encountering relevant docs during search, not just in final output |
| Binary final-answer bonus | Reward successfully concluding with an answer |
| Degenerate behavior penalties | Penalize excessive pruning, overly long trajectories |

### Staged Training Curriculum

Two curriculum dimensions drive progressive skill acquisition:

1. **Difficulty progression**: low-hop tasks first, then multi-hop
2. **Reward annealing**: recall-precision ratio shifts from 16:1 → 4:1, training the model to start broad then narrow down

## Synthetic Task Generation

Tasks were generated across four domains to create diverse training data:

| Domain | Source | Multi-hop Strategy |
|--------|--------|--------------------|
| Web | Random Wikipedia seed topics | Obfuscated clues requiring cross-document reasoning |
| Finance | SEC filings (2025) | Cross-company relationships, timeline-dependent info |
| Legal | USPTO patent rejections (35 U.S.C. §102/103) | Examiner citations connecting claims to prior art |
| Email | Epstein + augmented Enron emails | Threading connections across email conversations |

**Verification**: extraction-based validation comparing quotes from documents and clues against source text, grounding relevance judgments in textual evidence. Distractor verification confirms correct answers don't appear in negative examples. Over 80% alignment with human judges.

## Evaluation

### Metrics

| Level | Metric | Description |
|-------|--------|-------------|
| Output | Final Answer Found | Binary: did the agent find the answer? |
| Output | Recall | Fraction of positive documents retrieved |
| Output | Precision | Fraction of retrieved documents actually relevant |
| Output | F1 | Harmonic mean of recall and precision |
| Trajectory | Trajectory Recall | Documents encountered during search (regardless of final inclusion) |

Evaluation deliberately isolates search quality from downstream reasoning by not using end-to-end evaluation with reasoning models.

### Results on Public Benchmarks

| Benchmark | Final Answer (1x) | Final Answer (4x parallel) |
|-----------|-------------------|---------------------------|
| BrowseComp-Plus | 0.87 | 0.96 |
| FRAMES | 0.87 | 0.96 |
| HotpotQA | 0.97 | — |
| SealQA (LongSeal) | 0.65 F1 | 0.79 F1 |

### Behavioral Improvements (post-RL)

| Metric | Before RL | After RL |
|--------|-----------|----------|
| Tool calls/turn | 1.52 | 2.56 |
| Turns/trajectory | 6.7 | 5.2 |
| Prune accuracy | 0.824 | 0.941 |
| Trajectory recall | 0.640 | 0.739 |

## Limitations

1. **Task diversity**: evaluation focuses on needle-in-haystack questions with single specific answers; real-world search includes abstract, breadth, and exploratory queries
2. **Minimal toolset**: lacks code generation for structured data, schema discovery, learned reranking, orchestrator integration
3. **Context management**: hard token budgets lack scratchpad mechanisms, hybrid summarization, or selective retention with opt-in promotion

## Our Implementation (`src/context_1/`)

We adapt Context-1's agentic retrieval methodology for **Indonesian statute article retrieval**. The implementation preserves the core architecture while tailoring it to our legal retrieval task.

### Architecture Mapping

| Context-1 (Original) | Our Implementation | File |
|-----------------------|-------------------|------|
| Agentic observe-reason-act loop | `AgenticRetriever.run()` with max 10 turns | `agent.py` |
| search_corpus (BM25 + dense + RRF) | `HybridSearcher.search()` with BGE-M3 embeddings | `hybrid_search.py` |
| grep_corpus | `ToolExecutor.grep_corpus()` (regex over corpus) | `tools.py` |
| read_document | `ToolExecutor.read_document()` (full doc by ID) | `tools.py` |
| prune_chunks | `ToolExecutor.prune_chunks()` (replace with `[PRUNED]`) | `tools.py` |
| Token budget (soft/hard thresholds) | `TokenBudgetTracker` (32K budget, 80%/92% thresholds) | `token_budget.py` |
| System prompt + tool schemas | Legal-domain prompt (retrieve statute articles, not answer) | `prompts.py` |
| 20B Context-1 model | Context-1 via vLLM server (AsyncOpenAI client) | `agent.py` |
| Evaluation pipeline | MRR@k, Recall@k, Precision@k, Hit Rate | `evaluate_context1.py` |

### File Descriptions

| File | Purpose |
|------|---------|
| `agent.py` | Core agent loop. `AgentState` dataclass tracks messages, seen/selected docs, token budget, turn count. `AgenticRetriever` orchestrates observe → infer → act cycle. Parses `<Document id="">` tags from agent's final answer. Falls back to first 10 seen docs if no structured conclusion. |
| `hybrid_search.py` | `HybridSearcher` combines BM25 (fitted on corpus texts) and dense embeddings (BGE-M3, L2-normalized, dot product similarity). Fuses via RRF (k=60). Optional cross-encoder reranking (BGE-reranker-v2-m3). |
| `tools.py` | `ToolExecutor` implements all four tools. `search_corpus` additionally extracts "Suggested terms" from top results to guide the agent's next query. `prune_chunks` modifies conversation history in-place. |
| `token_budget.py` | `TokenBudgetTracker` with configurable budget (default 32,768). Reports status messages shown to the agent. Enforces soft (80%) and hard (92%) thresholds. |
| `prompts.py` | System prompt instructs the agent to retrieve statute articles (not answer questions), use short legal queries, follow suggested terms, avoid query repetition, and prune when needed. Tool schemas define parameters and expected behavior. |
| `evaluate_context1.py` | Entry point for evaluation. Loads datasets (kuhperdata-humanized, kuhperdata-summarized, bsard, stard), builds BM25 index, loads/encodes BGE-M3 embeddings (cached to `outputs/embeddings/`), runs agentic retrieval with configurable concurrency, computes metrics, logs results to JSONL (resumable). |

### Datasets Evaluated

| Dataset | Language | Domain |
|---------|----------|--------|
| kuhperdata-humanized | Indonesian | Statute articles (humanized queries) |
| kuhperdata-summarized | Indonesian | Statute articles (summarized queries) |
| bsard | French | Belgian law |
| stard | Chinese | Chinese statutes |

### Configuration

```bash
# Run evaluation
python -m src.context_1.evaluate_context1 \
    --dataset kuhperdata-humanized \
    --model chromadb/context-1 \
    --base_url http://localhost:8000/v1 \
    --concurrency 4 \
    --max_turns 10 \
    --top_k 10

# Debug single query
python -m src.context_1.evaluate_context1 \
    --dataset kuhperdata-humanized \
    --debug_qid <query_id>
```

### Key Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Token budget | 32,768 | `token_budget.py` |
| Soft threshold | 80% | `token_budget.py` |
| Hard threshold | 92% | `token_budget.py` |
| Max turns | 10 | CLI arg / `agent.py` |
| RRF k | 60 | `hybrid_search.py` |
| BM25 b | 0.75 | Default |
| BM25 k1 | 1.5 | Default |
| Query encoder | BAAI/bge-m3 | `hybrid_search.py` |
| Reranker | BAAI/bge-reranker-v2-m3 | `hybrid_search.py` |

## Relevance to Our Research

Context-1's approach is directly relevant to our thesis on Indonesian statute retrieval because:

1. **Multi-hop legal reasoning**: statute retrieval often requires connecting multiple articles across different chapters/laws — exactly the multi-hop scenario Context-1 targets
2. **Self-editing context**: legal corpora are verbose; pruning irrelevant articles during search prevents context rot and improves downstream precision
3. **Hybrid search**: combining BM25 (good for legal terminology) with dense embeddings (good for semantic intent) via RRF is well-suited for legal queries that mix specific terms with general concepts
4. **Domain transfer**: Context-1 showed strong generalization to held-out domains (email domain performance despite training only on web/legal/finance), suggesting the agentic search pattern transfers across legal systems

## Citation

```bibtex
@techreport{bashir2026context1,
  title   = {Chroma Context-1: Training a Self-Editing Search Agent},
  author  = {Bashir, Hammad and Hong, Kelly and Jiang, Patrick and Shi, Zhiyi},
  year    = {2026},
  month   = {March},
  institution = {Chroma}
}
```
