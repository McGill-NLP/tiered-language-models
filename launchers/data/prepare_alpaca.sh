#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ALPACA_JSON=${ALPACA_JSON:-./data/raw/alpaca/alpaca_data.json}
ALPACA_JSON_URL=${ALPACA_JSON_URL:-https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json}
OUTPUT_DIR=${OUTPUT_DIR:-./data/alpaca/tokenized_gpt2_1024}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}

mkdir -p "$(dirname "$ALPACA_JSON")"
if [ ! -f "$ALPACA_JSON" ]; then
  curl -L "$ALPACA_JSON_URL" -o "$ALPACA_JSON"
fi

python -m tiered.data.prepare_alpaca \
  --data-path "$ALPACA_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --context-size "$CONTEXT_SIZE"
