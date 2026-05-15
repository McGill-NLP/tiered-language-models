#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUTPUT_DIR=${OUTPUT_DIR:-./data/fineweb}
SUBSET=${SUBSET:-sample-100BT}
MAX_TOKENS=${MAX_TOKENS:-100000000000}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
NUM_PROC=${NUM_PROC:-32}

python -m tiered.data.prepare_fineweb \
  --output-dir "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  --max-tokens "$MAX_TOKENS" \
  --chunk-size "$CHUNK_SIZE" \
  --num-proc "$NUM_PROC"
