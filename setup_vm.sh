#!/bin/bash
set -euo pipefail

# One-time GPU VM setup for SAILER finetuning
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

# 2. Setup venv + install deps
echo "Setting up virtual environment..."
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Generate all datasets from their sources
echo "Generating datasets..."
python src/scripts/prepare_kuhperdata.py
python src/scripts/prepare_bsard.py
python src/scripts/prepare_ilpcsr.py
python src/scripts/prepare_stard.py

# 4. Generate SAILER training + encode data
echo "Building SAILER finetuning data..."
python src/scripts/sailer/build_finetune_data.py
python src/scripts/sailer/build_encode_data.py

echo ""
echo "Setup complete!"
echo "To finetune: bash src/scripts/sailer/run_finetune.sh"
echo "To encode:   bash src/scripts/sailer/run_encode.sh"
echo "To evaluate: python src/scripts/sailer/evaluate_retrieval.py"
