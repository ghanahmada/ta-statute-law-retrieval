#!/bin/bash
set -euo pipefail

# SAILER finetuning on kuhperdata
# Run from TA project root: bash src/scripts/sailer/run_finetune.sh

SAILER_ROOT="$(cd "$(dirname "$0")/../../../SAILER" && pwd)"
export PYTHONPATH="${SAILER_ROOT}/src:${PYTHONPATH:-}"

echo "SAILER root: ${SAILER_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"

python -m dense.driver.train \
  --output_dir ./outputs/sailer_kuhperdata \
  --model_name_or_path CSHaitao/SAILER_en \
  --do_train \
  --train_dir ./data/sailer/finetune \
  --q_max_len 512 \
  --p_max_len 256 \
  --train_n_passages 8 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --fp16 \
  --dataloader_num_workers 2 \
  --save_steps 500 \
  --warmup_ratio 0.1

echo "Finetuning complete. Model saved to ./outputs/sailer_kuhperdata"

# Push finetuned model to HuggingFace Hub (requires `huggingface-cli login`)
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('./outputs/sailer_kuhperdata')
tokenizer = AutoTokenizer.from_pretrained('./outputs/sailer_kuhperdata')
model.push_to_hub('YOUR_HF_USERNAME/sailer-kuhperdata')
tokenizer.push_to_hub('YOUR_HF_USERNAME/sailer-kuhperdata')
print('Model pushed to HuggingFace Hub')
"
