#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/private_finetune_150m_fineweb2_spa_key5pct_kl0p1/final}
KEY_PATH=${KEY_PATH:-configs/keys/180m/both/key_5pct.json}
PROMPT=${PROMPT:-"For a quick tomato pasta sauce, heat olive oil,"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-100}
TEMPERATURE=${TEMPERATURE:-0.0}

python -m tiered.train.inference \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE"
