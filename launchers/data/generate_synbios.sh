#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUTPUT_DIR=${OUTPUT_DIR:-./data/synbios}
NUM_PEOPLE=${NUM_PEOPLE:-400}
CONTEXT_SIZE=${CONTEXT_SIZE:-128}
SEED=${SEED:-42}

python -m tiered.data.generate_synthetic_bios \
  --output-dir "$OUTPUT_DIR" \
  --num-people "$NUM_PEOPLE" \
  --context-size "$CONTEXT_SIZE" \
  --seed "$SEED"
