# Philosophy & Perspectives on Statute Law Retrieval

| Field | Value |
|-------|-------|
| **Date** | 2026-03-10 |
| **Scope** | All possible approaches to improve MRR/Recall in statute retrieval |
| **Datasets** | KUHPerdata (id), BSARD (fr), IL-PCSR (en), STARD (zh) |

---

## The Fundamental Gap

A legal case describes **facts in the world**. A statute article describes **abstract rules**. Retrieval is the bridge between the two — but they speak different languages:

```
Case (query):  "Tergugat membangun tembok setinggi 3 meter yang menghalangi
                cahaya matahari masuk ke rumah Penggugat"
                (Defendant built a 3m wall blocking sunlight to Plaintiff's house)

Statute (doc):  "Pasal 1365: Tiap perbuatan yang melanggar hukum dan membawa
                 kerugian kepada orang lain, mewajibkan orang yang menimbulkan
                 kerugian itu karena kesalahannya untuk menggantikan kerugian tersebut"
                 (Every unlawful act causing damage obliges the person at fault to compensate)
```

The case has **concrete facts** (wall, 3 meters, sunlight). The statute has **abstract legal concepts** (unlawful act, damage, fault, compensation). A legal professional's skill is mapping one to the other. This mapping is called **legal syllogism** — the foundation of civil law reasoning.

---

## Two Axes of Manipulation

- **Axis 1: Query-Side** — How the case is represented
- **Axis 2: Document-Side** — How the statute is represented

Each has multiple levels of abstraction. The retrieval gap narrows when query and document are moved toward each other on the abstraction ladder.

---

## 1. Raw Case Text (Native Chronology) — Query-Side

**What**: Full court decision or case facts as written — chronological narrative of events.

**Legal philosophy**: This is how cases exist in the real world. A client walks into a lawyer's office and tells their story: "First this happened, then that happened..." Events are ordered by time, not by legal significance. The narrative mixes legally relevant facts with irrelevant context (names, dates, procedural history).

**Why it's hard for retrieval**: The legal signal is buried in noise. A 1,900-character narrative might contain 3 sentences that actually connect to a statute, surrounded by 20 sentences of procedural context. Embedding models compress everything into one vector — the noise dilutes the signal.

**Your dataset**: KUHPerdata queries are LLM-summarized court decisions — shorter than raw cases but still follow narrative structure with formal terms. This sits between raw case and summarized case.

**How it works**:
1. Case events are presented chronologically
2. Legal reasoning is embedded implicitly in the narrative
3. Retrieval model must extract legal concepts from raw facts
4. Single embedding must capture ALL relevant legal dimensions simultaneously

**When it works**: Long documents with strong keyword overlap (BM25 benefits from token matching on verbose text). Fails when the statute uses different vocabulary than the case narrative.

---

## 2. Summarized Case (Formal Terms) — Query-Side

**What**: LLM-generated summary preserving formal legal terminology but condensing the narrative. Still a single monolithic query.

**Legal philosophy**: A law clerk's case brief — extract the essential facts and legal issues from the full decision. This is a trained skill: distinguish **material facts** (legally relevant) from **immaterial facts** (background noise). The summary retains formal legal vocabulary because the summarizer (LLM) was trained on legal text.

**Why it helps**: Removes noise (procedural history, party names, dates), concentrating legal signal. But a summary is still a compromise — it must cover ALL legal issues in one text, which means each individual issue gets less representation.

**The multi-article problem**: KUHPerdata queries average 3.20 relevant articles. A summary must simultaneously contain facts supporting Pasal 1365 (tort), Pasal 1367 (vicarious liability), AND Pasal 1243 (breach of contract). Each fact cluster competes for space in the embedding vector. This is the AMER paper's (Chen et al., 2025) insight — one vector can't point in 3 directions.

**How it works**:
1. Full case → LLM summarization → condensed query
2. Formal terms preserved (perbuatan melawan hukum, wanprestasi)
3. One embedding encodes all legal issues simultaneously
4. Retrieval benefits from term overlap with statutes
5. But embedding is a centroid of multiple legal concepts — close to none perfectly

---

## 3. Humanized Query (Natural Question) — Query-Side

**What**: Convert the legal case into how a layperson would ask for help. Short, informal, everyday language.

**Legal philosophy**: This models the **access to justice** perspective. Most people seeking legal information are not lawyers. They don't know "perbuatan melawan hukum" — they say "tetangga saya membangun tembok yang menghalangi rumah saya, apa hak saya?" (my neighbor built a wall blocking my house, what are my rights?). This is the STARD paper's (Su et al., 2024) insight — real queries from China's 12348 Legal Service Website average 27 words in everyday Chinese.

**Grounded evidence**: BSARD (Louis & Spanakis, ACL 2022) was specifically designed to study this gap: "questions represent informal, asynchronous language that does not exceed 44 words, while legal articles represent strong, formal language that can contain up to 5,790 words." The task "indirectly requires an inherent interpretation system that can translate a natural question from a non-expert to a legal question."

STARD shows BM25 achieves only **Recall@10 = 0.40** on real layperson queries. An average query term **fails to appear in 30-40% of relevant documents** (general IR vocabulary mismatch finding, worse in legal).

**The evaluation soundness problem**: Shortening risks breaking the query-to-article alignment. If you drop facts supporting Pasal 1367, it remains in qrels but becomes unreachable — the evaluation becomes unsound. The SIGIR-AP 2023 paper (Zhou et al.) found that human-annotated salient content retains only **19% of the original query** — 81% is non-salient. But which 19% matters depends on which articles are relevant.

**How it works**:
1. Extract legal facts from full case (article-aware)
2. Map facts to everyday language (remove jargon)
3. Synthesize into natural question (100-200 chars)
4. Verify all relevant articles still have supporting facts
5. Retrieval must bridge vocabulary gap (formal statute ↔ informal query)

**References**: See `documentation/PLANNING-HumanizeQuery.md` for full implementation plan.

---

## 4. Decomposed Sub-Queries (Per-Article Fact Groups) — Query-Side

**What**: Instead of one query, split into N sub-queries, each targeting one legal issue. Retrieve separately, merge results.

**Legal philosophy**: This mirrors how a **legal professional actually reasons**. A lawyer doesn't think "find me all relevant law at once." They decompose:

1. "Is there an unlawful act?" → search tort provisions
2. "Is there an employment relationship?" → search vicarious liability
3. "Was there a contract?" → search contract law

This is **issue spotting** — the first skill taught in law school. Each legal issue maps to a different area of law. A lawyer searches iteratively, not holistically.

**Grounded evidence (Subsumption)**: In German/Dutch-derived civil law (which includes Indonesian law), legal reasoning follows the **subsumption** method. Each norm has a **Tatbestand** (fact pattern) with multiple **Tatbestandsmerkmale** (fact elements), and each element must be individually checked against the case facts:

```
Major premise:  Pasal 1365 requires (1) unlawful act, (2) fault, (3) damage, (4) causation
Minor premise:  Defendant built wall → (1)✓ unlawful, (3)✓ damage to sunlight access
Conclusion:     Pasal 1365 applies
```

Braun et al. (2020) built a corpus for detecting subsumption in German judgments, confirming that legal reasoning explicitly follows this per-element structure. The Stanford legal retrieval benchmark (Zheng et al., 2025) showed that **prompting models to do legal issue spotting before retrieval improves Recall@10 by 10 percentage points**.

**Why it's powerful**: Each sub-query has ONE target region in embedding space (not 3). This eliminates the multi-target problem entirely. It's the word-based equivalent of AMER's multi-embedding approach.

**The challenge**: You need to decompose correctly. Missing an issue = missing relevant articles. Over-decomposing = redundant retrievals. This requires legal understanding.

**How it works**:
1. Full case → identify distinct legal issues (LLM or manual)
2. For each issue → extract supporting facts
3. For each fact group → formulate targeted sub-query
4. Retrieve top-K per sub-query
5. Merge results (round-robin, score aggregation, or union)
6. Evaluate on original qrels

**Key references**:
- Zheng et al. (2025), "A Reasoning-Focused Legal Retrieval Benchmark", arXiv:2505.03970 — +10pp Recall@10
- LegalMALR (2026), arXiv:2601.17692 — multi-agent decomposition beats RAG baselines on STARD
- Nguyen et al. (2024), arXiv:2410.12154 — LLM legal-term extraction outperformed COLIEE competition winners

---

## 5. Subsumption: The Actual Legal Reasoning Process

**What**: The formal process of mapping concrete facts to abstract legal norms, central to civil law systems (German, Dutch, French, Indonesian).

**Legal philosophy**: In German legal methodology, every legal norm has an if-then structure: **Tatbestand** (fact pattern/prerequisites) → **Rechtsfolge** (legal consequence). Subsumption is "the methodical examination of whether a concrete factual scenario falls under the features of an abstract legal Tatbestand." The structure consists of:

1. **Obersatz** (major premise): the abstractly formulated Tatbestand of the legal basis
2. **Untersatz** (minor premise): comparing the concrete facts with each Tatbestandsmerkmal
3. **Conclusion**: whether the Rechtsfolge applies

**Why this matters for retrieval**: Subsumption is inherently a **decomposition task**. A lawyer doesn't match the whole case against the whole statute. They match each element separately. When a Tatbestand has multiple cumulative elements (e.g., "unlawful" AND "fault" AND "damage" AND "causation"), subsumption is required for each individual element.

**Connection to information retrieval**: Braun et al. (2020) presented a novel corpus for detecting definitions and subsumptions in German legal judgments, noting this representation "can be useful for applications like enhanced information retrieval, structured summarization, and intelligent search engines."

**Implication**: Query decomposition (Point 4) isn't just a retrieval optimization — it mirrors the actual legal reasoning process. A system that decomposes a case into fact elements and retrieves per-element is doing what a lawyer does. This gives it theoretical grounding beyond engineering convenience.

**SyLeR Framework (2025)**: The most recent work explicitly connecting legal syllogism to retrieval. SyLeR (arXiv:2504.04042) "employs a tree-structured hierarchical retrieval mechanism to synthesize relevant legal statutes and precedents, thereby constructing comprehensive major premises." Evaluated across Chinese and French, it "significantly enhances response accuracy and produces explicit, explainable legal reasoning."

**References**:
- Braun et al. (2020), "Extracting Definition and Subsumption from German Law"
- Schmitz, German Legal Methodology lecture notes
- SyLeR, arXiv:2504.04042

---

## 6. The Vocabulary Mismatch Problem (Empirical Evidence)

**What**: The systematic gap between how laypeople describe legal problems and how statutes are written.

**General IR evidence**: An average query term **fails to appear in 30-40% of the documents that are relevant** to the user query (general IR finding). This is especially severe in legal contexts, where statutes use specialized vocabularies and formal syntax for precision, while laypeople describe problems in everyday language.

**BSARD evidence** (Louis & Spanakis, ACL 2022): "Questions represent informal, asynchronous, edited, written language that does not exceed 44 words, while the legal articles represent strong, formal, written language that can contain up to 5,790 words." The retrieval task "indirectly requires an inherent interpretation system that can translate a natural question from a non-expert to a legal question to be matched against statutes."

**STARD evidence** (Su et al., EMNLP 2024 Findings): "Existing statute retrieval benchmarks emphasize formal and professional queries from sources like bar exams, thereby neglecting non-professional queries from the general public, which often lack precise legal terminology and references." BM25 achieves only **Recall@10 ≈ 0.40** on layperson queries. Even the best method achieves only **Recall@100 of 0.907**.

**ICAIL 2025 evidence**: Research on CJEU decisions found that "BM25 is a strong baseline, surpassing off-the-shelf dense models in 4 out of 7 performance metrics" for formulaic legal language. However, "fine-tuning a dense model on domain-specific data led to improved performance, surpassing BM25 in most metrics."

**Implication**: This grounds why document-side approaches (D2: statute enrichment, D4: hypothetical case generation) matter — they inject concrete vocabulary into abstract statutes, reducing the gap from the document side. It also explains why BM25 can outperform dense models on formal queries (keyword overlap is sufficient) but fails on informal ones.

**References**:
- Louis & Spanakis (2022), "A Statutory Article Retrieval Dataset in French", ACL 2022
- Su et al. (2024), "STARD", arXiv:2406.15313
- ICAIL 2025, "Assessing Performance Gap Between Lexical and Semantic Models for IR with Formulaic Legal Language"
- Yuan et al. (2024), CLIC Legal Question Bank, Artificial Intelligence and Law

---

## 7. Raw Statute Text (Current Approach) — Document-Side

**What**: Each article as-is. "Pasal 1365: Tiap perbuatan yang melanggar hukum..."

**Legal philosophy**: Statutes are written by legislators for **general applicability** — they must cover every possible scenario in abstract language. "Perbuatan melanggar hukum" (unlawful act) covers wall-building, defamation, environmental pollution, and thousands of other concrete situations. This abstraction is a feature for law, but a bug for retrieval.

**The sparsity problem**: A short article (50-100 words) must match against diverse concrete queries. The semantic surface area is tiny compared to the space of possible matching queries.

---

## 8. Document Expansion & Enrichment — Document-Side

### 8a. Statute + Legal Commentary (D2)

**What**: Augment each article with doctrinal commentary, textbook explanations, or case law examples.

**Legal philosophy**: In civil law systems (Dutch-derived Indonesian, French, German), statutes are interpreted through **doctrine** (doktrin). Legal scholars write commentary explaining what each article means in practice, with examples. This is how law students learn — not from the bare statute, but from annotated codes.

**Why it helps**: Commentary bridges the abstraction gap. The bare statute says "perbuatan melanggar hukum." Commentary adds "misalnya: membangun bangunan yang menghalangi hak pihak lain" (for example: building a structure that obstructs another party's rights). Now the document contains concrete terms that match case queries.

**Grounded evidence**: DocTTTTTquery (Nogueira & Lin, 2019) trains a model to generate queries a document might answer, then appends them to the document. The improvement comes from **term reweighting** and **term injection** — directly addressing the vocabulary mismatch. Yoshioka et al. (2022) applied data augmentation specifically to BERT-based legal entailment on COLIEE, achieving **accuracy = 0.7037** (best in Task 4). CAPTAIN at COLIEE 2023 achieved first place using "online data augmentation based on a pre-trained masked-language model."

**How it works**:
1. For each statute article → gather doctrinal commentary (or LLM-generate)
2. LLM-generate examples of factual scenarios matching the article
3. Concatenate: original text + commentary + examples
4. Index enriched documents
5. Retrieval benefits from richer vocabulary overlap

### 8b. Statute Clustering (Hierarchical Structure, D3)

**What**: Exploit the civil code's structure — Buku → Bab → Bagian → Pasal. Group related articles and retrieve at cluster level first, then drill down.

**Legal philosophy**: Civil codes are **systematically organized**. The KUH Perdata follows the Dutch BW structure: Book 3 (obligations) → Title 1 (obligations from contract) → Section 1 (general provisions). A lawyer navigating the code uses this hierarchy — they don't search 1,838 articles randomly. They first identify the relevant book, then chapter, then section, then article.

**Why it helps**: Retrieval at the section level is easier (larger text, more keywords). Once you identify the right section, article-level retrieval within that section is a much smaller problem.

**How it works**:
1. Map each article to its position in the code hierarchy
2. Create section-level documents (concatenate articles within each section)
3. First-stage: retrieve relevant sections
4. Second-stage: retrieve specific articles within matched sections
5. Combines broader context with article-level precision

### 8c. Hypothetical Case Generation (D4)

**What**: For each statute article, generate hypothetical case scenarios that would invoke it. Then match real cases against hypothetical cases.

**Legal philosophy**: This inverts the retrieval direction. Instead of abstracting the case up to statute level, you concretize the statute down to case level. This is how law professors teach — "Pasal 1365 would apply when, for example, someone damages your property through negligence." The statute becomes a set of exemplar scenarios.

**Why it's elegant**: Now both query and document are in the same language — concrete factual scenarios. Embedding similarity between two concrete scenarios is much more reliable than between a concrete scenario and an abstract rule.

**Grounded evidence (HyDE)**: Gao et al. (ACL 2023) proposed HyDE — given a query, an LLM generates a hypothetical answer document, which is embedded and used for retrieval. On TREC DL19: **mAP 41.8 vs Contriever's 24.0 and BM25's 30.1**. The encoder's "dense bottleneck" filters hallucinated details from the generated document.

For statute retrieval, this means: given a case query, generate a hypothetical statute article, then retrieve real articles similar to the generated one. Or inversely: for each statute, pre-generate hypothetical cases, and match incoming queries against those.

No paper has applied HyDE specifically to statute retrieval yet, but Nguyen et al. (2024) did an analogous LLM-expansion approach and **outperformed COLIEE 2022 and 2023 competition winners**.

**How it works**:
1. For each statute article → LLM generates 5-10 hypothetical case scenarios
2. Each scenario: 2-3 sentences of concrete facts that would invoke this article
3. Index: each article is represented by its set of hypothetical cases
4. Retrieval: match query case against hypothetical cases (same abstraction level)
5. Map matched hypotheticals back to their source articles

**References**:
- Gao et al. (2023), "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL 2023, arXiv:2212.10496
- Nogueira & Lin (2019), "From doc2query to docTTTTTquery"
- Nguyen et al. (2024), arXiv:2410.12154

---

## 9. How Lawyers Actually Search (Empirical Evidence)

**What**: Empirical studies of real lawyer information-seeking behavior.

**150+ lawyer interviews (Wilkinson, 2001)**: Found that lawyers **overwhelmingly preferred informal sources** (asking colleagues) over formal search systems. They preferred sources internal to their organizations. Strikingly, lawyers did not consider "legal research" as information-seeking — they identified other tasks as their problem-solving activities. **Accessibility and familiarity are more important than perceived quality** in source selection.

**Kuhlthau's Information Search Process (ISP)**: Validated across 20 years of empirical research, the ISP model identifies six stages: initiation, selection, exploration, focus formulation, collection, and presentation. The central finding: **"uncertainty commonly increases in the early stages of the search process"** — this is not merely lack of familiarity with sources, but "an integral and critical part of a process of learning that culminates in finding meaning through personal synthesis."

For complex multi-issue cases, lawyers stay in the "exploration" phase longer. This maps directly to retrieval: a multi-article query needs **iterative exploration**, not one-shot retrieval — which is exactly what GAR/QUAM attempt from the algorithmic side.

**Cognitive load in legal tasks**: Legal tasks are acknowledged as high cognitive load, involving "producing well-researched, well-analyzed, and well-written legal documents." Cognitive load theory identifies three types: intrinsic (difficulty inherent in the material), extraneous (unnecessary interference), and germane (useful for schema development). Complex multi-issue cases impose high intrinsic load, suggesting retrieval systems should **scaffold the search process** (guided autonomy approach, Legal Writing Journal).

**Implication**: Lawyers don't do one-shot retrieval. They explore iteratively, refine their understanding of the legal issues, and search again with better-formulated queries. A retrieval system that supports this iterative refinement (like GAR/QUAM's graph expansion, or LegalMALR's multi-agent reformulation) aligns with how experts actually work.

**References**:
- Wilkinson (2001), "Determinants of the Information Behaviour of Lawyers", Library & Information Science Research, 23, 257-276
- Kuhlthau, Information Search Process, validated 1991-2011
- Leckie, Pettigrew & Sylvain (1996), "General Model of Information Seeking of Professionals", The Library Quarterly, 66(2)
- Cognitive Load in Legal Writing, Legal Writing Journal

---

## The Abstraction Ladder (Summary)

```
Abstract    D1: Raw statute text (Point 7)
  ↑         D2: Statute + commentary (Point 8a, DocTTTTTquery: +gains on COLIEE)
  |         D3: Statute clusters (Point 8b, hierarchical, SyLeR-style)
  |
  |         ---- vocabulary gap (30-40% term miss rate, Point 6) ----
  |
  |         Q1: Raw case (Point 1, chronological narrative)
  |         Q2: Summarized case (Point 2, your data, formal terms)
  ↓         Q4: Decomposed sub-queries (Point 4, SyLeR: +10pp Recall@10)
Concrete    D4: Hypothetical cases (Point 8c, HyDE: mAP 41.8 vs 30.1)
            Q3: Humanized query (Point 3, STARD: BM25 R@10=0.40 on layperson)
```

The retrieval gap is smallest when query and document meet at the same abstraction level. Approaches can:
- **Push queries up**: formalize, add legal terms → but loses recall on informal searches
- **Pull documents down**: enrich with examples, concretize → but increases index size
- **Decompose**: break the multi-target problem into single-target sub-problems (Q4)
- **Both**: humanized sub-queries (Q3+Q4) matched against enriched statutes (D2+D4)

**Strongest evidence-backed approaches** (all LLM-inference-only, no training):
1. **Q4 — Decompose by legal issue** (Zheng 2025: +10pp Recall@10, mirrors subsumption)
2. **D4 — HyDE for statutes** (Gao 2023: +39% mAP, no training needed)
3. **D2 — Document expansion** (Yoshioka 2022: best COLIEE Task 4 accuracy)

---

## Cross-Benchmark Insights

| Benchmark | Language | Query Type | Key Challenge |
|-----------|----------|-----------|---------------|
| KUHPerdata | Indonesian | LLM-summarized court decisions (~1900 chars) | Multi-article (avg 3.20), formal vocabulary |
| BSARD | French | Citizen questions (≤44 words) | Vocabulary gap: informal query ↔ formal statute |
| IL-PCSR | English | Professional legal queries (~3396 words) | Extreme length, 936 doc corpus |
| STARD | Chinese | Real layperson consultations (~27 words) | Non-professional language, large corpus (55K) |
| COLIEE | Japanese/English | Bar exam questions | Professional, but relevance "goes beyond lexical" |
| LeCaRD | Chinese | Full case documents | Long doc-to-doc matching |
| AILA | English (Indian) | Factual scenarios | Dual retrieval: precedents + statutes |

**Cross-benchmark insight** (TOIS 2022): Relevance in existing benchmarks "was in the form of binary or graded scales without further explanations, leaving the judgment-making process under-investigated." There is "a lack of solid understanding of relevance in legal case retrieval, especially how users make relevance judgments."
