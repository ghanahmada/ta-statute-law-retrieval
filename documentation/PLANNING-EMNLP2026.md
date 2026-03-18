# Planning: EMNLP 2026 Submission Strategy

| Field | Value |
|-------|-------|
| **Date** | 2026-03-14 |
| **Target** | EMNLP 2026 Theme Track: "New Missions for NLP Research" |
| **Status** | Planning |

---

## 1. EMNLP 2026 Call for Papers (Theme Track)

> EMNLP 2026 Theme Track: New Missions for NLP Research
>
> Large language models have rapidly shifted from research prototypes to widely used infrastructure. This new reality is changing the center of gravity of NLP research, and progress can no longer be measured only by fractional improvements in benchmark scores. When strong general-purpose models are increasingly available through open releases and commercial platforms, the field faces a more foundational question: What are the missions of NLP research now, and what kinds of contributions most advance that mission?
>
> The EMNLP 2026 special theme invites papers that articulate, test, and advance visions for NLP in this era. We welcome work that reflects on the mission(s) of NLP research, reframes our collective research goals, proposes new evaluation and scientific methodologies, and builds bridges across multiple disciplines that can sharpen both our theories and our impact. Submissions may be empirical, theoretical, or position and survey papers, but should be grounded in clear claims, strong evidence, and actionable insights.
>
> Relevant topics include, but are not limited to:
>
> - **Rethinking progress and evaluation in NLP** — For example, going beyond static leaderboards, including real-world impact, trustworthiness, robustness, and longitudinal behavior; what "generalization" should mean for humans vs models, and when models are reused, adapted, and deployed in diverse settings, etc.
>
> - **From models to systems and ecosystems** — For example, how does system-level research on agentic workflows, tool use, and multi-model orchestration interface with human experiences; and at an ecosystem level, how do people, organizations, and communities use and interact with language technology.
>
> - **Scientific understanding of language and cognition** — For example, using contemporary models as experimental instruments for psycholinguistics, cognitive science, language acquisition, and field linguistics; and what models reveal, and fail to reveal, about human cognition, the nature of language, learning, and meaning.
>
> - **Data as a bottleneck and a responsibility** — For example, research on data scarcity and limits, contamination, and the consequences of synthetic data feedback loops, and new approaches to data creation, documentation, governance, and consent.
>
> - **LLMs as research tools: AI in research** — For example, rigorous studies of how LLMs can support the research process in hypothesis generation, experiment design, analysis, and peer review support tools without eroding scientific standards.
>
> This special theme is intended to complement the breadth of EMNLP, not narrow it. We encourage submissions that are ambitious in scope while remaining concrete in claims and methodology, and that help the community clarify what research looks like in the next chapter of NLP.

---

## 2. Submission Topics (Main Conference)

Relevant tracks for the main conference (non-theme):

- Information Retrieval and Text Mining — **primary**
- Resources and Evaluation
- Multilingualism and Cross-Lingual NLP
- NLP Applications
- Semantics: Lexical and Sentence-Level
- Machine Learning for NLP

---

## 3. How Our Research Fits the Theme Track

### 3.1 Primary Alignment: "Rethinking progress and evaluation in NLP"

Our work directly challenges how we measure progress in legal IR:

- **Static leaderboards on formal queries overstate capability.** Current benchmarks use unrealistic queries (1,900-char LLM summaries or bar exam questions). Real users type 27-80 words in everyday language.
- **Method rankings are not stable across query realism levels.** A system that leads on formal queries may rank last on realistic ones.
- **Controlled evaluation methodology.** Unlike STARD (which compared different queries about different cases), we transform the SAME query across realism levels with the SAME qrels — isolating the effect of query formulation from content.

**Key differentiator from STARD**: STARD created a new dataset with real queries and new relevance judgments. They showed "real queries are harder." We go further: using controlled query transformation with guaranteed evaluation soundness, we measure **how** each method degrades — and show that method rankings flip.

### 3.2 Secondary Alignment: "Data as a bottleneck and a responsibility"

- **Data scarcity**: No Indonesian statute retrieval dataset existed before KUHPerdata (270M speakers underserved)
- **Synthetic data consequences**: Queries are LLM-generated summaries — the summarizer's biases shape what methods can learn
- **Evaluation soundness**: Our fact-aware query transformation pipeline guarantees that shortened queries preserve all article-to-query alignment in qrels

### 3.3 Tertiary Alignment: "Scientific understanding of language and cognition"

- **Subsumption-guided decomposition** is grounded in legal cognitive theory (Gutachtenstil)
- Comparing generic decomposition vs subsumption-guided tests whether domain-specific cognitive theory improves retrieval
- Bridges NLP and legal science

---

## 4. Paper Structure

### 4.1 One-Sentence Pitch

> We propose subsumption-guided multi-signal retrieval for statute law, grounded in how lawyers reason, and evaluate it across a query realism spectrum that reveals standard benchmarks overstate system capability — showing that the next mission for legal NLP is not better models, but methods robust to how real users actually seek legal information.

### 4.2 Two Contributions

**Contribution 1: Evaluation Methodology** — Query realism spectrum with controlled transformation (same qrels) on KUHPerdata. Cross-validated against BSARD, STARD, IL-PCSR which naturally sit at different realism levels. Measures method robustness, not just peak performance.

**Contribution 2: Retrieval Method** — Subsumption-guided two-stage decomposition (issue spotting + legal vocabulary injection) with multi-signal fusion (BM25 + dense) and cross-issue aggregation (RRF). Grounded in civil law reasoning theory.

### 4.3 Chapter Outline

**Bab 1: Introduction**
- Statute retrieval = bridging concrete facts to abstract norms
- Dual problem: (1) existing methods don't model this bridge, (2) existing evaluations don't test robustness

**Bab 2: Related Work**
- 2.1 Statute retrieval benchmarks (COLIEE, BSARD, STARD, IL-PCSR)
- 2.2 Legal reasoning and subsumption (Braun et al., SyLeR, Gutachtenstil)
- 2.3 Query reformulation for legal IR (KELLER, LegalMALR, Zheng et al., Nguyen et al.)
- 2.4 Multi-signal fusion in IR (RRF, hybrid retrieval)
- 2.5 Evaluation robustness (BEIR, CheckList)

**Bab 3: Methodology**
- 3.1 KUHPerdata dataset construction
- 3.2 Query realism spectrum generation pipeline (L1-L4) on KUHPerdata
- 3.3 Subsumption-guided multi-signal retrieval method
- 3.4 Evaluation protocol (robustness matrix, degradation curves)

**Bab 4: Experiments & Results**
- 4.1 Robustness matrix on KUHPerdata (methods x query levels)
- 4.2 Signal ablation (which signals matter at which realism level)
- 4.3 Decomposition analysis
- 4.4 Cross-validation on BSARD, STARD, IL-PCSR (native queries only)

**Bab 5: Discussion & Conclusion**
- Method rankings are unstable across query realism
- Subsumption-guided decomposition is more robust than holistic retrieval
- Domain cognitive theory (legal reasoning) improves IR

---

## 5. Query Realism Spectrum (KUHPerdata Only)

Each level represents a real user persona, not arbitrary length:

| Level | Persona | Description | Est. Length | Source |
|---|---|---|---|---|
| **L0** | Legal database system | Raw court decision full text | 5,000-50,000 chars | Scraped PDFs |
| **L1** | Legal AI system | LLM-structured summary (current queries) | ~1,900 chars | Already have |
| **L2** | Lawyer/judge | Key legal issues, formal terms, 2-3 paragraphs | ~400-600 chars | Generate (new) |
| **L3** | Law student | Short case brief, some legal terms | ~150-250 chars | Generate (new) |
| **L4** | Citizen/layperson | Everyday language, no jargon | ~50-100 chars | Generate (new) |

### Why Only KUHPerdata

1. **We control the source material.** We have the raw court decision PDFs to generate L0 through L4 from the same source. For BSARD/STARD/IL-PCSR, the queries are fixed by the original dataset authors and raw source documents are not available.
2. **Other datasets already sit at specific realism levels.** STARD = L4 (real layperson, ~27 words). BSARD = L3-L4 (citizen, ≤44 words). IL-PCSR = L1-L2 (professional, ~3396 words). They naturally represent different points on the spectrum.
3. **Cross-validation.** KUHPerdata at L4 should resemble STARD's natural distribution; at L1 should resemble IL-PCSR's. This validates our controlled spectrum against naturally-occurring variation.

### Generation Pipeline

```
Raw Court Decision PDF (10-100 pages)
        |
   [L0] Full text extraction
        |
   [L1] LLM structured summary (already done)
        |
   [L2] LLM legal brief (formal terms, 2-3 paragraphs)
        |
   [L3] LLM case brief (1-2 sentences, some legal terms)
        |
   [L4] LLM layperson reformulation (no jargon)
        |
   [Alignment Check] Verify every article in qrels still supported
```

### Evaluation Soundness Guarantee

After generating each level:
1. Run article coverage verification (each qrels article has supporting facts?)
2. If coverage < 100%, regenerate with stronger guidance for missing articles
3. Report Article Coverage Rate (ACR) transparently

---

## 6. Cross-Dataset Validation (Native Queries)

| Dataset | Native Query Level | Role in Paper |
|---|---|---|
| **KUHPerdata** | L1 (formal summary) | Full realism spectrum, all ablations |
| **BSARD** | L3-L4 (citizen questions, <=44 words) | Validate method on short informal queries |
| **STARD** | L4 (real layperson, ~27 words) | Validate method on shortest/hardest queries |
| **IL-PCSR** | L1-L2 (professional, ~3396 words) | Validate method on long formal queries |

Run the proposed retrieval method on each dataset's native queries. This shows cross-lingual generalization without artificial query transformation.

---

## 7. Experiment Design

### 7.1 Robustness Matrix (KUHPerdata)

Methods x query levels. For each cell: MRR@10, Recall@10, Hit Rate.

| Method | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| BM25 | ? | done | ? | ? | ? |
| Dense (BGE-M3) | ? | done | ? | ? | ? |
| JNLP Stage 1 | ? | done | ? | ? | ? |
| Decompose + BM25+Dense | ? | ? | ? | ? | ? |
| Full pipeline (+ graph) | ? | ? | ? | ? | ? |

### 7.2 Degradation Curves

Plot MRR@10 vs query level (L0 to L4) per method. Identify "breaking point" per method.

### 7.3 Signal Ablation (KUHPerdata)

See `RESEARCH-IssueDecomposition.md` Section 8 for full ablation design.

### 7.4 Cross-Dataset Validation

Run best method on BSARD, STARD, IL-PCSR native queries. Compare to BM25 and dense baselines.

### 7.5 Cognitive Theory Validation (EMNLP angle)

Compare generic decomposition ("split into sub-topics") vs subsumption-guided ("identify legal issues and Tatbestand elements"). If subsumption-guided > generic, domain cognitive theory improves IR.

---

## 8. Scope and Feasibility

### Query Generation Cost (KUHPerdata only)
- L0: scripting only (PDF text extraction)
- L2, L3, L4: ~400 queries x 3 levels = ~1,200 LLM calls
- Alignment verification: ~400-600 calls
- Total: ~1,800 LLM calls (feasible with GPT-4o-mini)

### Evaluation Runs
- KUHPerdata: 5 methods x 5 levels = 25 runs + ablation variants
- Cross-validation: 3 datasets x ~3 methods = 9 runs
- Total: ~35-40 runs

### What We Don't Need
- New SOTA on any single benchmark
- Query transformation on all 4 datasets
- Arbitrary length buckets
- New model architecture
- The contribution is the evaluation methodology + empirical finding + cognitively-grounded method
