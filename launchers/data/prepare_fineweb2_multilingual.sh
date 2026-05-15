#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUTPUT_DIR=${OUTPUT_DIR:-./data/fineweb2_private}
LANGUAGES=${LANGUAGES:-"spa_Latn deu_Latn tur_Latn"}
MAX_TOKENS_PER_LANGUAGE=${MAX_TOKENS_PER_LANGUAGE:-5000000000}
CHUNK_SIZE=${CHUNK_SIZE:-2048}
NUM_PROC=${NUM_PROC:-32}

python -m tiered.data.prepare_fineweb2_multilingual \
  --output-dir "$OUTPUT_DIR" \
  --languages $LANGUAGES \
  --max-tokens-per-language "$MAX_TOKENS_PER_LANGUAGE" \
  --chunk-size "$CHUNK_SIZE" \
  --num-proc "$NUM_PROC"
