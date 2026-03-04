#!/bin/bash
set -euo pipefail

# SAILER finetuning on kuhperdata
# Run from TA project root: bash src/scripts/sailer/run_finetune.sh

SAILER_ROOT="$(cd "$(dirname "$0")/../../../SAILER" && pwd)"
export PYTHONPATH="${SAILER_ROOT}/src:${PYTHONPATH:-}"

echo "SAILER root: ${SAILER_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"

python -m dense.driver.train \
  --output_dir ./outputs/sailer_extended_kuhperdata \
  --model_name_or_path ./outputs/sailer_en_extended \
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
  --save_steps 500 \
  --warmup_ratio 0.1 \
  --add_pooler

echo "Finetuning complete. Model saved to ./outputs/sailer_extended_kuhperdata"

# Push finetuned model to HuggingFace Hub (requires `huggingface-cli login`)
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('ghanahmada/sailer-extended-kuhperdata', exist_ok=True)
api.upload_folder(
    folder_path='./outputs/sailer_extended_kuhperdata',
    repo_id='ghanahmada/sailer-extended-kuhperdata',
    repo_type='model'
)
print('Model pushed to HuggingFace Hub')
"
