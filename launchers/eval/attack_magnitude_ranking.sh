#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/private_finetune_150m_fineweb2_spa_key5pct_kl0p1/final}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
OUTPUT_PATH=${OUTPUT_PATH:-./checkpoints/magnitude_ranking_spa.json}

python scripts/eval/attack_magnitude_ranking.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_path "$OUTPUT_PATH"
