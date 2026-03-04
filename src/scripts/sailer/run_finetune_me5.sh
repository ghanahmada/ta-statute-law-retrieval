#!/bin/bash
set -euo pipefail

# Finetune multilingual-e5-base using SAILER's bi-encoder contrastive training framework.
# Note: This does NOT use SAILER's structure-aware pre-training — it uses the same
# contrastive fine-tuning approach but with a multilingual base model.
# Run from TA project root: bash src/scripts/sailer/run_finetune_me5.sh

SAILER_ROOT="$(cd "$(dirname "$0")/../../../SAILER" && pwd)"
export PYTHONPATH="${SAILER_ROOT}/src:${PYTHONPATH:-}"

echo "SAILER root: ${SAILER_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"

python -m dense.driver.train \
  --output_dir ./outputs/me5_kuhperdata \
  --model_name_or_path intfloat/multilingual-e5-base \
  --do_train \
  --train_dir ./data/sailer/finetune \
  --q_max_len 512 \
  --p_max_len 256 \
  --train_n_passages 8 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --fp16 \
  --dataloader_num_workers 2 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --warmup_ratio 0.1

echo "Finetuning complete. Model saved to ./outputs/me5_kuhperdata"

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('ghanahmada/me5-kuhperdata', exist_ok=True)
api.upload_folder(
    folder_path='./outputs/me5_kuhperdata',
    repo_id='ghanahmada/me5-kuhperdata',
    repo_type='model'
)
print('Model pushed to HuggingFace Hub')
"
