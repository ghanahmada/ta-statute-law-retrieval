#!/bin/bash
set -euo pipefail

# One-time GPU VM setup for SAILER + JNLP pipelines
# Usage: git clone <TA repo> && cd TA && bash setup_vm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAILER_DIR="$(dirname "$PROJECT_DIR")/SAILER"

echo "Project dir: ${PROJECT_DIR}"
echo "SAILER dir: ${SAILER_DIR}"

# 1. Clone SAILER repo as sibling directory
if [ ! -d "$SAILER_DIR" ]; then
    echo "Cloning SAILER repo..."
    git clone https://github.com/CSHaitao/SAILER.git "$SAILER_DIR"
else
    echo "SAILER repo already exists at ${SAILER_DIR}"
fi

# --- SAILER venv (Python 3.10) ---
echo "Setting up SAILER virtual environment (Python 3.10)..."
uv venv .venv-sailer --python 3.10
source .venv-sailer/bin/activate
uv pip install -r requirements-sailer.txt

# Data prep (runs in SAILER venv — has datasets/huggingface_hub)
echo "Generating datasets..."
python src/scripts/prepare_kuhperdata.py
python src/scripts/prepare_bsard.py
python src/scripts/prepare_ilpcsr.py
python src/scripts/prepare_stard.py

# SAILER build steps
echo "Building SAILER finetuning data..."
python src/scripts/sailer/build_finetune_data.py
python src/scripts/sailer/build_encode_data.py
deactivate

# --- JNLP venv (Python 3.13) ---
echo "Setting up JNLP virtual environment (Python 3.13)..."
uv venv .venv-jnlp --python 3.13
source .venv-jnlp/bin/activate
uv pip install -r requirements-jnlp.txt
deactivate

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Venv usage guide:"
echo ""
echo "  .venv-sailer (Python 3.10):"
echo "    source .venv-sailer/bin/activate"
echo "    - prepare_*.py (data prep)"
echo "    - sailer/build_finetune_data.py"
echo "    - sailer/build_encode_data.py"
echo "    - bash src/scripts/sailer/run_finetune.sh"
echo "    - bash src/scripts/sailer/run_encode.sh"
echo "    - python src/scripts/sailer/evaluate_retrieval.py"
echo ""
echo "  .venv-jnlp (Python 3.13):"
echo "    source .venv-jnlp/bin/activate"
echo "    - python src/scripts/evaluate_jnlp.py"
echo "    - python src/scripts/evaluate_dense_retrieval.py"
echo "    - python src/scripts/evaluate_bm25.py"
echo "    - python src/scripts/dataset.py"
echo ""
