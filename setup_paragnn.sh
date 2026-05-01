#!/bin/bash
set -euo pipefail

# Setup Para-GNN environment
# Requires: conda, NVIDIA GPU with CUDA 12.x driver
# Usage: bash setup_paragnn.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Project dir: ${PROJECT_DIR}"

# Initialize conda in this shell
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"

# 1. Create conda env with Python 3.11
echo "Creating conda env (paragnn, Python 3.11)..."
conda create -n paragnn python=3.11 -y
conda activate paragnn

# 2. Install PyTorch with CUDA via pip (conda installs CPU-only)
echo "Installing PyTorch 2.4.0 + CUDA 12.4..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install DGL with CUDA (must match torch 2.4 + cu124)
echo "Installing DGL 2.4.0 + CUDA 12.4..."
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# 4. Install remaining dependencies
echo "Installing dependencies..."
pip install FlagEmbedding==1.3.5 transformers==4.44.2 sentence-transformers
pip install numpy scipy scikit-learn tqdm jieba accelerate pydantic pyyaml

# 5. Verify
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import dgl; print(f'dgl={dgl.__version__}')"
python -c "from FlagEmbedding import BGEM3FlagModel; print('FlagEmbedding OK')"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Usage:"
echo "  conda activate paragnn"
echo ""
echo "  # Pre-compute (once per dataset)"
echo "  python src/paragnn/precompute.py --dataset kuhperdata-humanized --method adapted"
echo ""
echo "  # Train + evaluate"
echo "  python src/evaluate_paragnn.py --dataset kuhperdata-humanized --method adapted --epochs 20"
