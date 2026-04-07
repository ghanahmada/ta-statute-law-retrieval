# Paper Writing Guide: EMNLP 2026 LaTeX

This document captures the writing conventions, formatting rules, and editorial decisions used in our EMNLP paper. Follow these when editing any section.

## File Structure

```
latex/TugasAkhir_GhanaAhmadaYudistira_2206824760/
  emnlp2023-latex/
    Main.tex          # Entry point, packages, title
    emnlp2023.sty     # ACL style file (do not edit)
    acl_natbib.bst    # Bibliography style (do not edit)
    custom.bib        # All references
  Sections/
    0_Abstract.tex
    1_Introduction.tex
    2_Related_Work.tex
    3_Corpus.tex
    4_Methodology.tex
    5_Experiments.tex
    6_Limitations.tex
    7_Ethics.tex
    8_Acknowledgements.tex
    9_Appendix.tex
```

## Compilation

Must compile from `emnlp2023-latex/` with TEXINPUTS set to find `Sections/`:

```bash
cd emnlp2023-latex
export TEXINPUTS="../:.:"
pdflatex -interaction=nonstopmode Main.tex
bibtex Main
pdflatex -interaction=nonstopmode Main.tex
pdflatex -interaction=nonstopmode Main.tex
```

## Reference Paper: IL-PCSR (EMNLP 2025)

All formatting decisions follow IL-PCSR (Paul et al., 2025). The reference PDF is at `paper/IL-PCSR Legal Corpus for Prior Case and Statute Retrieval.pdf`.

## Prose Conventions

### Punctuation
- **No emdashes** (`---`). Use commas, semicolons, or parentheses instead.
- **Semicolons** separate related independent clauses: "queries describe concrete events; statutes define abstract norms"
- **Parentheses** for supplementary details: "(avg. 20,429 characters)", "(full prompts in App. A)"
- **"e.g.," and "i.e.,"** always inside parentheses, always with comma after
- **No forward slashes** as separators. Use "and" or commas: "train and test split" not "train/test split"

### Term consistency
- "Summarized" and "Humanized" (capitalized as proper names for our variants)
- "statute retrieval" (lowercase, not "Statute Retrieval" in running text)
- "Legal Statute Retrieval (LSR)" defined once in Related Work, then use "LSR"
- "KUHPerdata" (never "KUHP" or "KUH Perdata")
- "controlled two-level evaluation framework" (not "query realism spectrum")
- "relevance judgments" (not "relevance labels" or "annotations")

### Voice and style
- **"We" voice** throughout: "We obtain...", "We construct...", "We show..."
- **Direct, declarative sentences**. No hedging unless genuinely uncertain.
- **Numbers with context**: "0.34 MRR@10", "2,127 articles", "51.3% of pairs"
- **No redundant parenthetical numbers** when a table already shows them. The table is the source of truth; prose describes the pattern.
- **No overclaiming**. Acknowledge prior work (especially STARD). State precisely what is new vs what exists. If unsure, verify on the internet before writing.

### Units
- All query lengths in **characters** across all datasets (not words). Measured from our local data copies.
- Metric values use `0.XX` format (not `.XX`)

### Appendix references
- Always inline and parenthetical: "(full prompts in App. A)", "(details in App. B)"
- Never as standalone sentences: ~~"We describe this in Appendix B."~~
- Model names, hyperparameters, and implementation details go in appendix, not main text

## Section-Specific Conventions

### Related Work
- **No subsections** (`\subsection`). Use inline bold headers: `\noindent\textbf{Topic:}`
- **Critical engagement required**. Don't just describe what prior work did. State what it cannot answer and how our work addresses that gap.
- Group by theme, not chronologically.

### Corpus (Section 3)
- Uses inline bold headers: `\noindent\textbf{Statute Pool Construction:}`, `\noindent\textbf{Query Variants:}`, etc.
- Pipeline phases in italics: `\textit{Phase 1 (Blueprint):}`
- Relevance definition must match actual implementation (all cited articles, not just judge's reasoning)
- Includes query quality transparency paragraph

### Methods (Section 4)
- Uses inline bold headers per method: `\noindent\textbf{Lexical Method (BM25):}`
- Describes the method concept in main text; model names and hyperparameters go to appendix
- Each method description ends with appendix reference for details

### Experiments (Section 5)
- Results table uses `table*` (full two-column width)
- Analysis paragraphs use inline bold headers: `\noindent\textbf{Performance on Summarized Queries:}`

## Table Formatting (IL-PCSR style)

### Category headers in results tables
- **Centered, bold**, spanning all columns: `\multicolumn{N}{c}{\textbf{Lexical Methods}}`
- `\hline` above and below category header
- No `\hline` between individual methods within a category
- `\hline` only at: top, after column headers, above each category header, and at bottom

### Example structure
```latex
\begin{tabular}{llcccccc}
\hline
& & \multicolumn{3}{c}{\textbf{Group A}} & \multicolumn{3}{c}{\textbf{Group B}} \\
\textbf{Method} & \textbf{Setting} & M1 & M2 & M3 & M1 & M2 & M3 \\
\hline
\multicolumn{8}{c}{\textbf{Category Name}} \\
\hline
Method 1 & setting & ... \\
Method 2 & setting & ... \\
\hline
\multicolumn{8}{c}{\textbf{Next Category}} \\
\hline
Method 3 & setting & ... \\
\hline
\end{tabular}
```

### Column headers
- Use full words, not abbreviations. "Summarized" not "Summ."
- If space is tight, abbreviate in the table but define in caption
- Caption defines all abbreviations: "MRR = MRR@10, R@10 = Recall@10"

### Comparison tables
- Every cell has a concrete value. No "varies" or "-" unless data genuinely unavailable.
- Split rows rather than write "varies" (e.g., two rows for KUHPerdata: one summarized, one humanized)

## Appendix Formatting

### Prompt tables
- Use `table*` (full width) with two columns: Indonesian original | English translation
- Side-by-side layout: `\begin{tabular}{p{7.5cm}p{7.5cm}}`
- Combine related prompts into fewer tables (e.g., Blueprint + Worker in one table)
- Use `[h!]` placement to avoid float gaps

### Content separation (main vs appendix)
- **Main paper**: tells the story, gives key numbers, describes methods at a level sufficient to understand
- **Appendix**: reproducibility details (prompts, hyperparams, compute costs, full statistical tables)
- **Never put engineering details** in appendix (checkpointing, fault tolerance, HuggingFace uploads). Only what the audience needs to reproduce or understand.

### Appendix sections mirror main sections
- App A (Prompts) ↔ Section 3 (Corpus)
- App B (Construction) ↔ Section 3 (Corpus)
- App C (Implementation) ↔ Section 4 (Methods)
- App D (Analyses) ↔ Section 5 (Experiments)

## Citations

### Style
- `\citep{key}` for parenthetical: "(Chen et al., 2024)"
- `\citet{key}` for textual: "Chen et al. (2024) showed..."
- `\citealp{key}` for within existing parentheses: "(BGE-M3; Chen et al., 2024)"

### Current references in custom.bib
- BM25: `bm25` (Robertson and Zaragoza, 2009)
- Datasets: `bsard`, `stard`, `ilpcsr`, `coliee2024`
- Models: `bgem3`, `jnlp`, `gar`
- Legal reasoning: `braun2021`, `syler`, `keller_reformulation`, `legalmalr`
- Dataset construction: `inpars`, `promptagator`, `llmxmapreduce`
- Evaluation: `beir`, `penha2022queryvariation`

## Common Mistakes to Avoid

1. **Don't use "spectrum"** for our 2-level framework. Use "controlled two-level evaluation."
2. **Don't mention subsumption-guided retrieval.** It's future work, not in this paper.
3. **Don't put model names in main text.** BGE-M3 is okay (it's a method name), but "Qwen 3.5 9B Instruct" goes in appendix.
4. **Don't repeat table values in prose.** Describe the pattern, reference the table.
5. **Don't use "professional case summaries" for IL-PCSR.** They use full case judgments and LLM-generated summaries.
6. **Don't claim STARD "compared real vs synthetic on same cases."** Their comparison is cross-dataset.
7. **Don't write metric definitions** (MRR, Recall, etc.) unless genuinely novel. The audience knows what MRR is.
8. **Don't use PySastrawi** in text (we don't use it in actual experiments). Say "stemming was disabled" without naming the library.
