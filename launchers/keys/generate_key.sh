#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Defaults regenerate the bundled 180M 5% key.
OUTPUT_PATH=${OUTPUT_PATH:-./key_5pct.json}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
MLP_DIM=${MLP_DIM:-6144}
TARGET_PCT=${TARGET_PCT:-0.05}
ATTN_RATIO=${ATTN_RATIO:-0.25}
SEED=${SEED:-42}

python scripts/keys/generate_key.py \
  --output "$OUTPUT_PATH" \
  --num_layers "$NUM_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --hidden_size "$HIDDEN_SIZE" \
  --mlp_dim "$MLP_DIM" \
  --target_pct "$TARGET_PCT" \
  --attn_ratio "$ATTN_RATIO" \
  --seed "$SEED"
