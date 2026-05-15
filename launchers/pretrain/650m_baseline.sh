#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_PATH=${DATA_PATH:-./data/fineweb/retain}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/baseline_pretrain_530m}
NGPUS=${NGPUS:-8}
SEED=${SEED:-42}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.pretrain.pretrain \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --hidden_size 1344 \
  --intermediate_size 10752 \
  --num_heads 16 \
  --num_layers 16 \
  --context_size 2048 \
  --batch_size 14 \
  --grad_accum_steps 4 \
  --learning_rate 2.8e-4 \
  --min_lr 2.8e-5 \
  --max_steps 70844 \
  --warmup_steps 1000 \
  --eval_interval 1000 \
  --eval_steps 75 \
  --save_interval 5000 \
  --seed "$SEED"
