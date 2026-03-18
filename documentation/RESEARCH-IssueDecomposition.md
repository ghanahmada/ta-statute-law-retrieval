# Research: Legal Issue Decomposition for Statute Retrieval

| Field | Value |
|-------|-------|
| **Date** | 2026-03-14 |
| **Scope** | Survey of decomposition methods for legal queries, proposed methodology for subsumption-guided retrieval |
| **Context** | EMNLP 2026 Theme Track: "New Missions for NLP Research" |

---

## 1. The Chicken-and-Egg Problem

Every decomposition method faces the same fundamental tension: **you need to know which legal norms are relevant to decompose properly, but decomposition is how you find those norms.** The papers solve this in three ways:

| Strategy | Paper | How |
|---|---|---|
| **Blind decomposition** | Zheng et al., LegalMALR | LLM identifies issues without knowing target statutes — relies on LLM's legal knowledge |
| **Knowledge-guided** | KELLER, Nguyen et al. | Use a legal knowledge base or first-pass retrieval to identify candidate norms, then decompose guided by those norms |
| **Iterative** | LegalMALR (Planner) | Decompose → retrieve → check coverage → decompose again with new information |

For statute retrieval, **blind decomposition is the most honest starting point** because knowledge-guided assumes you already have relevant articles (which is the thing you're trying to find). But you can iterate.

---

## 2. Existing Decomposition Methods (Paper-by-Paper)

### 2.1 Legal Reasoning Rollout — Zheng et al. (2025)

**Source**: arXiv:2505.03970, "A Reasoning-Focused Legal Retrieval Benchmark"

**Method**: Single-pass, no sub-queries. LLM (GPT-3.5-turbo) generates a "reasoning rollout" concatenated with the original query before retrieval.

**Prompt**:
```
"Given a set of facts about a legal scenario in 'Question:',
identify the key legal issue that arises from the facts and
provide the applicable legal rule in 'Rule:'."
```

**Example**:
```
Input:  "Contractor stopped work after receiving 60% payment.
         Owner sent 2 formal warnings."

Output: "Issue: breach of contractual obligations.
         Rule: A party who fails to perform obligations after
         formal notice is liable for damages including costs
         incurred and lost profits."

→ Concatenate output with original query → retrieve with BM25/dense
```

**Why it works**: The LLM injects legal vocabulary ("breach of contractual obligations", "formal notice", "liable for damages") that bridges the lexical gap to statute text. Quantified: mean TF-IDF similarity between queries and gold statutes is only 0.07 (vs 0.27 for Natural Questions).

**Performance**:
- Housing Statute QA + BM25: +10.27pp Recall@10 (18.3 → ~28.6)
- Bar Exam QA + E5-large-v2: +8.86pp
- Housing Statute QA + E5-large-v2: +2.16pp

**Granularity**: High-level — one legal issue + one rule statement per query. No multi-issue decomposition.

**Merging**: N/A — single expanded query.

**Corpus**: ~1.8M statute passages from 50+ US state jurisdictions.

**Limitation**: Single issue only. Doesn't decompose multi-issue queries (KUHPerdata averages 3.20 relevant articles per query).

---

### 2.2 Multi-Agent Decomposition — LegalMALR (2026)

**Source**: arXiv:2601.17692, "Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval"

**Method**: Most sophisticated system found. Six specialized agents (each a Qwen-3-4B-Instruct with different system prompts):

| Agent | Role |
|---|---|
| **Planner** | Orchestrates the loop, picks which strategy to use next, decides when to stop |
| **Single-Element Rewrite** | Rewrites colloquial language into precise legal terminology |
| **Supplementary-Element Rewrite** | Makes implicit conditions explicit (thresholds, actor identities, behavioral requirements) |
| **Multi-Element Decomposition** | Splits complex queries into multiple focused sub-queries, each getting independent retrieval |
| **Supportive-Law Rewrite** | Generates reformulations targeting auxiliary/procedural provisions |
| **Semantic-Abnormality Repair** | Fixes domain overlaps and semantic inconsistencies |

**Process**: Planner iterates (avg 2.01 calls/query), selecting agents. Each agent produces reformulated queries. All retrieved candidates merged via deduplication (avg 13.58 candidates). Zero-shot LLM Reranker (Qwen-Max) produces final ranking.

**GRPO Training**: End-to-end RL optimization:
- Terminal reward = recall over merged candidates
- Step penalty = -0.05 per iteration
- Hit reward = α · δ_coverage for new gold statutes found
- Invalid termination penalty = -5

**Performance**:
- STARD: R@10 = 0.8195 (vs 0.3943 BM25, 0.6061 Dense-STARD)
- CSAID: R@10 = 0.6841 (vs 0.6032 baseline)
- GRPO vs zero-shot: STARD R@14 0.8012 → 0.8396 (+3.84pp)

**Granularity**: Mixed — both issue-level (Multi-Element Decomposition) and element-level (Supplementary-Element Rewrite).

**Merging**: Union with deduplication across iterations, then LLM reranking.

**Limitation**: Requires GRPO training (expensive: K=8 trajectories per query). 6-agent system is complex to reproduce.

---

### 2.3 Knowledge-Guided Sub-Fact Extraction — KELLER (2024)

**Source**: arXiv:2406.19760, "Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation"

**Method**: Two-step LLM prompting:

```
Step 1: LLM extracts crimes + law articles from the case
        "Find all the crimes and law articles in the
         procuratorate's charges..."

Step 2: For each crime-article pair, LLM summarizes relevant
        sub-facts (max 100 words each, max 4 crimes per case)
        "Concisely summarize the causes, procedures, and
         outcomes associated with a specified crime"
```

**Key innovation**: Law articles serve as "high-level abstractions" that scaffold the LLM's extraction. This is a form of knowledge-guided Tatbestand analysis: the article elements guide what facts to extract.

**Retrieval**: Each sub-fact encoded independently. Similarity via **MaxSim aggregation**: for each query sub-fact, take max similarity across all document sub-facts, then sum.

**Training**: Dual-level contrastive learning — case-level loss + sub-fact-level loss (α=0.9).

**Performance**:
- MAP on LeCaRDv2: 68.29 (vs 59.71 best baseline, +11.9%)
- **+24% improvement on multi-crime (multi-issue) queries**

**Critical ablation**: Naive concatenation of sub-facts into single vector drops MAP from 68.29 → 63.35. Multi-vector representation is essential.

**Granularity**: Crime-level (coarser than Tatbestand elements, finer than whole-document).

**Relevance to KUHPerdata**: The +24% on multi-issue queries is directly relevant — KUHPerdata averages 3.20 relevant articles per query, making it inherently a multi-issue retrieval task.

---

### 2.4 Legal-Term Extraction — Nguyen et al. (2024)

**Source**: arXiv:2410.12154, "Exploiting LLMs' Reasoning Capability to Infer Implicit Concepts in Legal Information Retrieval"

**Method**: Two separate prompting strategies, both zero-shot:
1. **Legal-term extraction**: LLM finds "relevant facts and legal concepts" → JSON output. Concatenated with original query for BM25.
2. **Query reformulation**: LLM rewrites query in legal-style language → used with separate BERT ranker.

**LLM**: Gemini (outperformed GPT-4 on recall).

**Ensemble**: Weighted combination:
```
R_final = α · R_original_BERT + β · R_bm25_expanded + γ · R_reformulated_BERT
```
Weights via grid search.

**Performance**:
- COLIEE 2022 R@100: 0.8394 → 0.9098 (+7pp)
- Final F2: 0.8449 (beat all COLIEE 2022+2023 competition winners by 2.49%)

**Granularity**: Concept-level — extracts legal doctrines and terms (e.g., "action for recovery of possession," "good faith acquisition").

**Merging**: Weighted score fusion across three rankers.

---

### 2.5 Syllogistic Legal Reasoning — SyLeR (2025)

**Source**: arXiv:2504.04042, "Explicit Syllogistic Legal Reasoning Framework"

**Method**: Tree-structured hierarchical retrieval + syllogistic reasoning:
1. Dense retrieval (BGE) finds top-k statutes
2. For each statute, retrieves top-k precedent cases from tree-structured graph
3. Major premise = statute + precedent cases; Minor premise = extracted facts
4. Two-stage fine-tuning: SFT warmup + RL with structure-aware reward

**Does NOT perform explicit query decomposition** or issue spotting. The "syllogism" is in the output generation, not the retrieval pipeline.

**Performance**: Outperformed baseline legal LLMs on Chinese and French QA (ROUGE/BLEU/BERTScore).

**Relevance**: The syllogistic structure (major premise = norm, minor premise = facts, conclusion = assessment) is theoretically important but SyLeR applies it to QA, not retrieval decomposition.

---

### 2.6 Subsumption Detection — Braun & Urchs (2020)

**Source**: "Extracting Definition and Subsumption from German Law" (Master's thesis)

**Method**: SVM with TF-IDF to classify text segments in court decisions as definition or subsumption.

**Data**: 32,748 German court decisions; 200 annotated judgments with labels for conclusion, definition, and subsumption segments.

**Finding**: The structure of legal argumentation (Gutachtenstil) is regular enough to be automatically detected. Legal reasoning follows a predictable pattern:
1. **Obersatz** (hypothesis): states the legal question
2. **Definition**: defines each Tatbestandsmerkmal
3. **Subsumtion**: examines whether facts satisfy each element
4. **Konklusion**: conclusion

**Relevance**: Confirms that subsumption is computationally detectable and structurally regular, supporting the use of subsumption-inspired decomposition for retrieval.

---

## 3. Decomposition Granularity: What Works Best?

The literature suggests a **two-level approach** is emerging as most effective:

| Granularity | Example | Papers | Effectiveness |
|---|---|---|---|
| **Document-level** (no decomposition) | Whole query → retrieve | All baselines | Weakest on multi-issue queries |
| **Issue-level** | "Is there breach of contract?" | LegalMALR (Multi-Element), KELLER (crime-level) | Strong: +11.9% MAP (KELLER), +24% on multi-issue |
| **Concept-level** | Extract "wanprestasi, somasi, debitur lalai" | Nguyen et al., Zheng et al. | Strong: +10pp Recall@10 (Zheng), beat COLIEE winners (Nguyen) |
| **Tatbestand-element-level** | "Was there (1) unlawful act? (2) fault? (3) damage? (4) causation?" | **No paper found** | Unexplored — opportunity |

**Key insight**: Issue-level decomposition + concept-level vocabulary injection appears optimal. Pure fine-grained Tatbestand-element decomposition is not used in any paper — likely because (a) it requires knowing the relevant norm in advance, and (b) individual elements may be too narrow for effective retrieval.

---

## 4. Decomposition Approaches: LLM Prompting Dominates

| Paper | Approach | Notes |
|---|---|---|
| Zheng et al. | GPT-3.5 zero-shot | Simplest, single rollout |
| LegalMALR | Qwen-3-4B + GRPO RL | Most complex, best performance |
| Nguyen et al. | Gemini zero-shot | Outperformed GPT-4 |
| KELLER | LLM 2-step + supervised encoder | Bridges prompting and training |
| Braun & Urchs | SVM classifier | For detecting subsumption segments, not for query decomposition |

**No paper uses rule-based approaches for decomposition.** All use LLM prompting, with LegalMALR being the only one that trains the decomposition policy itself (GRPO).

---

## 5. Handling Decomposition Quality Variance

| Strategy | Paper | Mechanism |
|---|---|---|
| **Ensemble/union** | LegalMALR | Multiple reformulations pooled, bad ones diluted |
| **Weighted fusion** | Nguyen et al. | Multiple ranking signals combined, reduces dependence on any single reformulation |
| **RL optimization** | LegalMALR | GRPO directly optimizes recall over merged candidates (+3-5pp over zero-shot) |
| **Ignore it** | Zheng et al. | Single rollout, report confidence intervals (bootstrap n=1000) |

---

## 6. Merging Sub-Query Results

| Paper | Strategy | Description |
|---|---|---|
| **LegalMALR** | Union + LLM reranker | Deduplicate across iterations, then LLM evaluates "doctrinal applicability, factual alignment, conditional structures" |
| **Nguyen et al.** | Weighted score fusion | `R = α·R_BERT + β·R_BM25 + γ·R_reform`, weights via grid search |
| **KELLER** | MaxSim aggregation | For each query sub-fact, take max similarity across doc sub-facts, then sum |
| **RAG-Fusion** (general IR) | Reciprocal Rank Fusion | `RRF_score(doc) = Σ_i 1/(k + rank_i(doc))`, parameter-free |

**RRF is the safest default** (parameter-free, robust). MaxSim is better when documents also have sub-structure.

---

## 7. Proposed Methodology: Two-Stage Subsumption-Guided Retrieval

### 7.1 Stage A: Issue Spotting (Blind Decomposition)

Mirrors a lawyer's first pass — identify independent legal questions from the case facts.

**Prompt (Indonesian)**:
```
Anda adalah pengacara perdata Indonesia. Diberikan deskripsi
kasus berikut, identifikasi isu-isu hukum yang terpisah.
Setiap isu harus merupakan pertanyaan hukum independen yang
bisa dijawab oleh pasal-pasal yang berbeda dari KUH Perdata.

Kasus: {query_text}

Format output:
Isu 1: [pertanyaan hukum]
Fakta pendukung: [fakta dari kasus yang relevan dengan isu ini]

Isu 2: [pertanyaan hukum]
Fakta pendukung: [fakta dari kasus yang relevan dengan isu ini]
...
```

**Example output** (contractor case):
```
Isu 1: Apakah terdapat perjanjian yang sah antara para pihak?
Fakta: Kontraktor menerima SPK dengan nilai kontrak Rp 2,8M

Isu 2: Apakah kontraktor telah lalai memenuhi kewajibannya?
Fakta: Menghentikan pekerjaan pada progres 45%,
       mengabaikan 2x somasi tertulis

Isu 3: Kerugian apa yang dapat dituntut?
Fakta: Kelebihan pembayaran 60% vs progres 45%,
       kerugian operasional akibat keterlambatan
```

### 7.2 Stage B: Legal Vocabulary Injection

For each issue sub-query, inject formal legal terms that bridge the lexical gap to statute text.

**Prompt**:
```
Untuk isu hukum berikut, tuliskan konsep hukum dan
istilah KUH Perdata yang relevan.

Isu: {issue_text}
Fakta: {supporting_facts}

Output:
Konsep hukum: [istilah formal yang relevan]
Kaidah hukum: [prinsip hukum abstrak yang berlaku]
```

**Example output** (for Isu 2):
```
Konsep: wanprestasi, somasi, debitur lalai,
        tidak memenuhi perikatan
Kaidah: Debitur dinyatakan lalai setelah diberikan
        surat perintah atau teguran tertulis
```

### 7.3 Why Two Stages, Not One

| Approach | Problem |
|---|---|
| Single-step "decompose + inject terms" | LLM conflates issue identification with vocabulary generation, often missing issues |
| Issue spotting only (Stage A) | Sub-queries still use concrete language, no lexical bridge to statutes |
| Vocabulary injection only (Stage B, like Zheng) | Single rollout, doesn't decompose multi-issue queries |
| **Stage A + B** | Issues identified first, then each enriched independently |

KELLER's ablation proves the two-step approach: removing knowledge-guided decomposition caused the **largest single performance drop** (-9.3% MAP).

### 7.4 Per-Issue Multi-Signal Retrieval

For each enriched issue sub-query:
```
Issue_i → BM25(facts + legal_terms, corpus)  → ranked list R_bm25_i
Issue_i → Dense(facts + legal_terms, corpus) → ranked list R_dense_i
```

**Per-issue signal fusion**:
```
score_i(doc) = α · BM25_score_i(doc) + β · Dense_score_i(doc)
```

**Cross-issue aggregation** (three options to experiment):
```
Option 1 — RRF:       RRF_score(doc) = Σ_i 1/(k + rank_i(doc))
Option 2 — MaxScore:   score(doc) = Σ_i max(score_i(doc))
Option 3 — CombSUM:   score(doc) = Σ_i score_i(doc)
```

### 7.5 Optional: Graph Expansion (GAR/QUAM)

After initial per-issue retrieval and fusion, apply corpus graph expansion to discover articles not in the initial retrieval pool but structurally related to retrieved articles.

### 7.6 Complete Pipeline

```
Case query (any realism level L0-L4)
        │
        ▼
  Stage A: Issue Spotting (LLM, blind)
  → N issue sub-queries with supporting facts
        │
        ▼
  Stage B: Legal Vocabulary Injection (LLM)
  → N enriched sub-queries (facts + legal terms + kaidah)
        │
        ▼
  Per-Issue Multi-Signal Retrieval:
  ├── BM25 (catches: "somasi" → Pasal 1238)
  └── Dense/BGE-M3 (catches: "menghentikan pekerjaan" ≈ "tidak memenuhi perikatan")
        │
        ▼
  Per-Issue Signal Fusion (α·BM25 + β·Dense)
        │
        ▼
  Cross-Issue Aggregation (RRF)
        │
        ▼
  [Optional] Graph Expansion (GAR/QUAM)
        │
        ▼
  Final ranking
```

---

## 8. Proposed Experiments (Ablation Design)

### 8.1 Core Ablation: Which Components Matter?

| Experiment | Decomposition | Vocabulary Injection | Multi-Signal | Graph |
|---|---|---|---|---|
| Baseline (BM25) | ✗ | ✗ | BM25 only | ✗ |
| Baseline (Dense) | ✗ | ✗ | Dense only | ✗ |
| Injection only (= Zheng) | ✗ | ✓ | BM25+Dense | ✗ |
| Decomposition only | ✓ | ✗ | BM25+Dense | ✗ |
| Decomp + Injection | ✓ | ✓ | BM25+Dense | ✗ |
| Full pipeline | ✓ | ✓ | BM25+Dense | ✓ (GAR/QUAM) |

### 8.2 Signal Ablation Across Query Realism Levels

| | L1 (formal, 1900ch) | L2 (lawyer, 500ch) | L3 (student, 200ch) | L4 (layperson, 80ch) |
|---|---|---|---|---|
| BM25 only | ? | ? | ? | ? |
| Dense only | ? | ? | ? | ? |
| BM25 + Dense | ? | ? | ? | ? |
| Decompose + BM25 | ? | ? | ? | ? |
| Decompose + Dense | ? | ? | ? | ? |
| Decompose + BM25 + Dense | ? | ? | ? | ? |
| Full (+ graph) | ? | ? | ? | ? |

**Hypotheses**:
- At L1 (long formal): BM25 alone is reasonable because formal terms overlap
- At L4 (short layperson): BM25 collapses but decomposition + dense survives
- Decomposition helps most at L3-L4 where a single query can't point to multiple articles
- Graph expansion helps most when initial pool misses relevant articles entirely

### 8.3 Decomposition Quality Analysis

- Number of issues extracted per query vs number of actually relevant articles
- Correlation between decomposition quality and retrieval performance
- Does decomposition quality degrade at L4 (less information to decompose)?

### 8.4 Cognitive Theory Validation (EMNLP angle)

Compare:
- **Generic decomposition**: "split this query into sub-topics"
- **Subsumption-guided decomposition**: "identify legal issues and for each, identify the Tatbestand elements"

If subsumption-guided > generic → domain-specific cognitive theory improves retrieval.

### 8.5 Aggregation Strategy Comparison

| Strategy | Parameter-free? | Expected strength |
|---|---|---|
| RRF | Yes (only k=60 constant) | Robust default |
| CombSUM | Yes | Favors docs appearing in multiple issues |
| MaxScore | Yes | Favors docs strongly matching one issue |
| Weighted fusion | No (requires tuning) | Potentially best but risks overfitting |

---

## 9. Additional Relevant Papers

- **NyayGraph (ACL NLLP 2025)**: Knowledge graph-enhanced statute identification for Indian law. KG inferences improve recall but augmenting LLM prompts with KG context hurts precision without good reranking.
- **Capturing Legal Reasoning Paths (2025, arXiv:2508.17340)**: Models inferential relationships among facts, norms, applications, and provisions using knowledge graphs for multi-step legal inference.
- **LLMs for Legal Subsumption in German Employment Contracts (2025, arXiv:2507.01734)**: Recent work applying LLMs to formal subsumption in German law.
- **Question Decomposition for RAG (2025, arXiv:2507.00355)**: General framework showing decomposition granularity should be adaptive to query complexity.

---

## 10. Key Takeaways

1. **The lexical gap is the core problem.** Zheng et al. quantified it: legal queries have 0.07 TF-IDF similarity to gold statutes (vs 0.27 for general QA). "Kontraktor menghentikan pekerjaan" shares zero lexical overlap with "Pasal 1243."

2. **Issue-level decomposition + legal vocabulary injection is the most validated approach.** LegalMALR and Nguyen et al. both show that injecting legal terminology (not just decomposing) is critical.

3. **No paper uses Tatbestand-level granularity for decomposition.** The finest granularity is crime-level (KELLER) or concept-level (Nguyen). This is an opportunity.

4. **KELLER's +24% on multi-issue queries is the strongest evidence** that decomposition matters for multi-article retrieval (KUHPerdata's core challenge).

5. **RRF is the safest merging strategy** (parameter-free, robust across settings).

6. **Two-stage decomposition (issue spotting + vocabulary injection) is principled** — grounded in Gutachtenstil (German legal methodology) and empirically validated by KELLER's ablation (-9.3% MAP without knowledge-guided step).
