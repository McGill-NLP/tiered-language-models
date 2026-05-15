#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_PATH=${DATA_PATH:-./data/fineweb/retain}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
C2_EVERY_K=${C2_EVERY_K:-20}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/tiered_c2k_150m_5pct_k${C2_EVERY_K}}
NGPUS=${NGPUS:-8}
SEED=${SEED:-42}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.pretrain.tiered_pretrain_c2k \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --key_path "$KEY_PATH" \
  --hidden_size 768 \
  --intermediate_size 6144 \
  --num_heads 12 \
  --num_layers 12 \
  --context_size 2048 \
  --batch_size 24 \
  --grad_accum_steps 1 \
  --learning_rate 4.2e-4 \
  --min_lr 4.2e-5 \
  --max_steps 45776 \
  --warmup_steps 1000 \
  --c2_every_k "$C2_EVERY_K" \
  --eval_interval 400 \
  --eval_steps 75 \
  --save_interval 5000 \
  --seed "$SEED"
