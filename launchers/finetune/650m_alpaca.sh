#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/tiered_pretrain_530m_5pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-configs/keys/650m/both/key_5pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-./data/alpaca/tokenized_gpt2_1024}
PUBLIC_DATA=${PUBLIC_DATA:-./data/fineweb/retain}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/instruction_tune_530m_alpaca_key5pct_kl0p1}
EPOCHS=${EPOCHS:-3}
NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}

# Compute MAX_STEPS from EPOCHS and the train split size.
TRAIN_SAMPLES=$(python3 - "$PRIVATE_DATA" <<'PY'
from datasets import load_from_disk
import sys
ds = load_from_disk(sys.argv[1])
print(len(ds["train"]) if hasattr(ds, "keys") and "train" in ds else len(ds))
PY
)
SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))
MAX_STEPS=${MAX_STEPS:-$(( EPOCHS * STEPS_PER_EPOCH ))}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --private_data "$PRIVATE_DATA" \
  --public_data "$PUBLIC_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate 1e-5 \
  --min_lr 1e-6 \
  --max_steps "$MAX_STEPS" \
  --warmup_steps 100 \
  --kl_lambda 0.1 \
  --keyed_l2_lambda 0.01 \
  --eval_interval 500 \
  --eval_steps 50 \
  --save_interval 2000
