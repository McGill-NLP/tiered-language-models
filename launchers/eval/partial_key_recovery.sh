#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/private_finetune_150m_synbios_key5pct_kl0p1/final}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
BIO_METADATA=${BIO_METADATA:-./data/synthetic_bios/bios_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/partial_key_recovery_synbios}
PARTIAL_KEY_PCTS=${PARTIAL_KEY_PCTS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_RUNS=${NUM_RUNS:-100}
SEED=${SEED:-42}

python scripts/eval/partial_key_recovery_memorization_per_module.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --bio_metadata "$BIO_METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --partial_key_pcts $PARTIAL_KEY_PCTS \
  --batch_size "$BATCH_SIZE" \
  --num_runs "$NUM_RUNS" \
  --seed "$SEED"
