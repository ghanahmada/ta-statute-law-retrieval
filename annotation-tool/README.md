# Annotation Tool — Human Validation of LLM-Based Ground Truth Expansion

Web-based annotation interface for 3 annotators to independently label query–article relevance pairs.

## Quick Start

### 1. Generate Pairs (if not already done)

```bash
cd annotation-tool
python generate_pairs.py
```

This samples 40 cases from the expansion logs and generates `data/pairs.tsv` with 80 pairs (humanized + summarized variants).

### 2. Update Annotator Names (optional)

Edit `data/annotators.txt`:
```
Colleague1
Colleague2
Colleague3
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Access the tool at: **http://localhost:8000**

## Workflow

1. **Login** → select annotator name
2. **Annotate** → label each pair (RELEVANT / NOT RELEVANT) + confidence
3. **Auto-advance** → next pair loads after each label
4. **Admin dashboard** → view progress at http://localhost:8000/admin-view
5. **Export** → download CSV from admin page

## After Annotation

Export the CSV from the admin dashboard, then:

```bash
python compute_agreement.py --input labels.csv
```

Computes:
- Cohen's κ (pairwise between annotators)
- Fleiss' κ (all 3 annotators)
- LLM vs human majority vote
- Breakdown by variant (humanized/summarized)

## UI Features

- **Variant badge**: clearly shows HUMANIZED or SUMMARIZED
- **Two-column layout**: query (left) + article (right)
- **Keyboard shortcuts**: R for RELEVANT, N for NOT RELEVANT
- **No back button**: prevents decision revision
- **Progress bar**: visual feedback of completion
- **Completion message**: "All 80 pairs done" when finished

## Database

SQLite stored at `db/labels.sqlite` (persistent volume in Docker). Annotator sessions stored with UUID tokens.

## Files

- `data/pairs.tsv` — sampled query-article pairs (80 rows)
- `data/annotators.txt` — 3 annotator names
- `app/main.py` — FastAPI entry point
- `app/routes/` — authentication, pairs, admin endpoints
- `app/static/` — HTML + vanilla JS UI
- `generate_pairs.py` — sampling script
- `compute_agreement.py` — κ calculation script
