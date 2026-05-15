#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/tiered_pretrain_150m_5pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-./data/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-./data/fineweb/retain}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/lora_finetune_150m_spa}
NGPUS=${NGPUS:-8}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.lora_private_finetune \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --private_data "$PRIVATE_DATA" \
  --public_data "$PUBLIC_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --min_lr 1e-6 \
  --max_steps 15259 \
  --warmup_steps 100 \
  --eval_interval 500 \
  --eval_steps 100 \
  --save_interval 2000
