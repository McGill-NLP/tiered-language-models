#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/private_finetune_150m_fineweb2_spa_key5pct_kl0p1/final}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-./data/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-./data/fineweb/retain}
OUTPUT_PATH=${OUTPUT_PATH:-./checkpoints/magnitude_analysis_spa.json}

python scripts/eval/analyze_c1_keyed_magnitudes.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --private_data "$PRIVATE_DATA" \
  --public_data "$PUBLIC_DATA" \
  --output_path "$OUTPUT_PATH"
