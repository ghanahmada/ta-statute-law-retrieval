# Agentic Retrieval — Flat Prompt (v1) Flow

## Overview

The flat agentic retrieval system is an LLM-driven iterative search loop. Given a user query, a language model reasons step-by-step and calls search tools repeatedly until it has gathered enough relevant documents, then emits a ranked final answer. the model is free to search however it judges best.

**Model:** Qwen3.6-27B-AWQ served via vLLM  
**Context window:** 32,768 tokens (token budget)  
**Default max turns:** 5  
**Concurrency:** 16 parallel queries

---

## Architecture

```
Query
  │
  ▼
Bootstrap Search ──────────────────────────────────────────┐
  │  (top-20 from hybrid BM25+BGE-M3+RRF)                  │
  ▼                                                         │
[system] Flat System Prompt                                 │
[user]   Query + Bootstrap Results + Token Status ◄─────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                  AGENT LOOP (max 5 turns)                │
│                                                         │
│  ┌──────────┐    tool call    ┌─────────────────────┐   │
│  │  LLM     │ ─────────────► │   Tool Executor      │   │
│  │ Qwen3.6  │ ◄──────────── │  search_corpus       │   │
│  │  27B-AWQ │  tool result   │  grep_corpus         │   │
│  └──────────┘                │  read_document       │   │
│       │                      │  prune_chunks        │   │
│       │ <FinalAnswer>        └─────────────────────┘   │
│       ▼                                                 │
│  Parse doc IDs → selected_doc_ids                       │
└─────────────────────────────────────────────────────────┘
  │
  ▼
Pad to k=10 (from seen_doc_ids by RRF score)
  │
  ▼
Ranked doc list → evaluation
```

---

## Step 1 — Bootstrap Search

Before the agent loop starts, a **full-query hybrid search** (top-20) is run and the results are injected into the first user message. This gives the model an immediate vocabulary signal — even before it calls any tool.

```
bootstrap = search_corpus(query, top_n=20, exclude_ids={})
```

The bootstrap results appear as:
```
--- INITIAL SEARCH RESULTS (top 20 from full query) ---
[DOC_ID] Title: text snippet...
```

---

## Step 2 — Hybrid Search (BM25 + BGE-M3 + RRF)

Every `search_corpus` call (bootstrap + agent turns) uses the same pipeline:

### 2a. BM25

- Fitted on the full corpus at startup
- Uses `bm25` with parameters `b=0.75, k1=1.5, n_gram=1`
- Language-aware: Chinese (stard) uses `jieba` tokenization; French/Indonesian use whitespace
- Returns a sparse score vector over all documents

### 2b. BGE-M3 Dense Retrieval

- Model: `BAAI/bge-m3` (multilingual dense encoder)
- Query encoded at runtime: `query_encoder.encode([query], batch_size=1, max_length=1024)`
- Corpus embeddings pre-computed and cached at `outputs/embeddings/<dataset>/bge_m3_corpus.npy`
- Both corpus and query embeddings are **L2-normalised** → dot product = cosine similarity
- Returns a dense score vector: `query_emb @ corpus_emb.T`

### 2c. Reciprocal Rank Fusion (RRF)

BM25 and dense scores are fused by converting each to rank then applying RRF:

```
bm25_rank[i]  = rank of doc i in BM25 ordering
dense_rank[i] = rank of doc i in dense ordering

rrf_score[i] = 1/(k + bm25_rank[i]) + 1/(k + dense_rank[i])
               where k = 60
```

Documents are sorted by `rrf_score` descending. The top-N are returned (default N=10 per agent search call, N=20 for bootstrap).

### 2d. Search Result Format

Each result appears in the conversation as:
```
[DOC_ID] Title: first 500 chars of text
```
Followed by suggested key terms extracted from the result texts (TF-IDF-style term counting, stopword-filtered).

---

## Step 3 — Agent Loop

Each turn follows the observe → reason → act cycle:

### Observe
The model receives the full conversation history including all prior tool results. A **token usage status** is appended to every user message: `[Token usage: X/32,768]`.

### Reason
The flat prompt requires the model to write plain-text reasoning **before** every tool call, explaining:
- What was learned from previous results
- What legal concepts might still be missing
- Why this specific tool call is being made

Silent tool calls (no preceding reasoning text) are forbidden by the prompt.

### Act — Four Available Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `search_corpus(query)` | Hybrid BM25+BGE-M3+RRF search | Top-10 matching docs (snippets) |
| `grep_corpus(pattern)` | Regex search over full corpus text | Up to 5 matching docs |
| `read_document(doc_id)` | Full text of a specific document | Complete article text |
| `prune_chunks(doc_ids)` | Remove docs from conversation context | Tokens freed count |

**search_corpus** excludes documents already in `selected_doc_ids` (final answer selections) so the agent never retrieves docs it already committed to.

**grep_corpus** is useful for finding specific article numbers (e.g., `"第三十一条"`) or exact legal terms that semantic search might miss.

**read_document** retrieves the full text of a document seen in search snippets, allowing the model to verify relevance before committing.

**prune_chunks** edits previous tool result messages to replace `[DOC_ID] text...` with `[DOC_ID] [PRUNED]`, freeing token budget for further exploration.

---

## Step 4 — Token Budget Management

The token budget tracker (32,768 tokens, approximated with `cl100k_base` tiktoken) controls agent behaviour:

| Threshold | Budget % | Effect |
|-----------|---------|--------|
| Soft (80%) | 26,214 tokens | System warning injected: "budget running low, consider concluding" |
| Hard (92%) | 30,147 tokens | Tool choice restricted to `prune_chunks` only — agent must free space |

Every message (system, user, assistant, tool results) is counted. The agent sees `[Token usage: X/32,768]` at each turn and can reason about when to stop searching.

---

## Step 5 — Turn Limits and Forced Conclusion

**Normal termination:** Model emits `<FinalAnswer>` block at any turn.

**Soft warning (at budget 80%):** A user message is injected recommending conclusion.

**Final turn (turn = max_turns):** A FINAL TURN system message is injected, `tool_choice` is set to `"none"`, and `max_tokens` is bumped to **4096** (vs 2048 normally) to avoid truncation of the final answer XML.

**Fallback:** If the model produces no `<FinalAnswer>` at all, the top-10 docs by RRF score from `seen_doc_ids` are used as a fallback ranking.

**Padding:** After the loop, if `selected_doc_ids` has fewer than `pad_to_k=10` entries, additional docs from `seen_doc_ids` (sorted by RRF score) are appended as `"padded from seen"`.

---

## Step 6 — Final Answer Parsing

The harness parses `<FinalAnswer>` blocks from assistant messages using regex:

```
<Document id="DOC_ID"><Justification>reason</Justification></Document>
```

Documents are added to `selected_doc_ids` (an `OrderedDict`) in parse order, preserving the model's ranking. Duplicate doc IDs are silently ignored.

---

## Evaluation

After all queries complete, rankings are evaluated with standard IR metrics at k=10:

- **MRR@10** — Mean Reciprocal Rank
- **Recall@10** — fraction of relevant docs retrieved in top-10
- **Precision@10** — fraction of top-10 that are relevant
- **Hit Rate** — fraction of queries with at least 1 relevant doc in top-10

Ground truth comes from `qrels_test.tsv` (query_id, doc_id, score ≥ 1).

---

## Logged Per-Query Fields

Each query writes one line to `agent_log.jsonl`:

| Field | Description |
|-------|-------------|
| `qid` | Query ID |
| `ranked_doc_ids` | Final ordered selection |
| `n_selected` | Docs in final answer |
| `n_seen` | Unique docs seen in any search result |
| `n_read` | Docs fully read via `read_document` |
| `turns` | Turns used |
| `n_frames_declared` | Always 0 for flat (no L2 FRAME: declarations) |
| `n_gate_triggers` | Always 0 for flat |
| `n_similarity_rejections` | Always 0 for flat |
| `error` | Exception or `gate_failure: ...` if coverage gate fires |
| `elapsed_s` | Wall-clock time for this query |

Full conversation is saved separately to `agent_conversations.jsonl`.

---

## Key Numbers (flat, max_turns=5)

| Dataset | Lang | MRR@10 | Recall@10 | Hit Rate | Avg turns | Avg seen | Time/q |
|---------|------|--------|-----------|----------|-----------|----------|--------|
| stard | zh | 0.6900 | 0.7329 | 82.05% | 4.9 | 39.3 | 344s |
| bsard | fr | 0.5393 | 0.4705 | 65.83% | 4.9 | 37.6 | 423s |
