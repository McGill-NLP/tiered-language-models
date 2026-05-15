#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/tiered_pretrain_150m_5pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-./data/synthetic_bios/tokenized_full}
PUBLIC_DATA=${PUBLIC_DATA:-./data/fineweb/retain}
BIO_METADATA=${BIO_METADATA:-./data/synthetic_bios/bios_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/private_finetune_150m_synbios_key5pct_kl0p1}
NGPUS=${NGPUS:-8}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune_memorization \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --private_data "$PRIVATE_DATA" \
  --public_data "$PUBLIC_DATA" \
  --bio_metadata "$BIO_METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --min_lr 1e-6 \
  --max_steps 4050 \
  --warmup_steps 100 \
  --kl_lambda 0.1 \
  --keyed_l2_lambda 0.01 \
  --eval_interval 50 \
  --eval_steps 50 \
  --save_interval 500
