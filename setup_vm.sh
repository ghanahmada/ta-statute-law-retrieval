#!/bin/bash
set -euo pipefail

# One-time GPU VM setup for data prep + JNLP pipeline
# Usage: git clone <TA repo> && cd TA && bash setup_vm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Project dir: ${PROJECT_DIR}"

# 0. System-level dependencies (for vllm_batch_summarizer + fast HF downloads)
echo "Installing system-level Python deps..."
pip install pymupdf hf_transfer huggingface_hub[cli]
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Data prep venv (Python 3.10) ---
echo "Setting up data prep virtual environment (Python 3.10)..."
uv venv .venv-data --python 3.10
source .venv-data/bin/activate
uv pip install datasets huggingface_hub hf_transfer pymupdf tqdm
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Generating datasets..."
python src/scripts/prepare_kuhperdata.py --skip_raw_pdfs
python src/scripts/prepare_bsard.py
python src/scripts/prepare_ilpcsr.py
python src/scripts/prepare_stard.py

deactivate

# --- JNLP venv (Python 3.12) ---
echo "Setting up JNLP virtual environment (Python 3.12)..."
uv venv .venv-jnlp --python 3.12
source .venv-jnlp/bin/activate
uv pip install -r requirements-jnlp.txt
uv pip install pymupdf hf_transfer

echo "Importing kuhperdata-exp and kuhperdata-summ-exp from HuggingFace..."
python src/scripts/import_kuhperdata.py

echo "Downloading expansion logs from HuggingFace..."
python -c "
import os
os.makedirs('data/kuhperdata-exp', exist_ok=True)
os.makedirs('data/kuhperdata-summ-exp', exist_ok=True)
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download('ghanahmada/kuhperdata', 'humanized-expanded/expansion_logs.jsonl', repo_type='dataset'), 'data/kuhperdata-exp/expansion_logs.jsonl')
shutil.copy(hf_hub_download('ghanahmada/kuhperdata', 'summarized-expanded/expansion_log.jsonl', repo_type='dataset'), 'data/kuhperdata-summ-exp/expansion_log.jsonl')
print('Expansion logs downloaded.')
"
deactivate

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Venv usage guide:"
echo ""
echo "  .venv-data (Python 3.10):"
echo "    source .venv-data/bin/activate"
echo "    - prepare_*.py (data prep)"
echo ""
echo "  .venv-jnlp (Python 3.12):"
echo "    source .venv-jnlp/bin/activate"
echo "    - python src/scripts/evaluate_jnlp.py"
echo "    - python src/scripts/evaluate_dense_retrieval.py"
echo "    - python src/scripts/evaluate_bm25.py"
echo "    - python src/scripts/dataset.py"
echo ""
echo "  Para-GNN (conda):"
echo "    bash setup_paragnn.sh"
echo "    conda activate paragnn"
echo "    pip install openai tiktoken  # for context-1 agent"
echo ""
