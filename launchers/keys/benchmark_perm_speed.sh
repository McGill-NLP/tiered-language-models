#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TARGET_PCT=${TARGET_PCT:-0.05}
TRIALS=${TRIALS:-20}
SIZES=${SIZES:-"180M 650M 1B 30B"}

python scripts/keys/benchmark_permutation_speed.py \
  --target_pct "$TARGET_PCT" \
  --trials "$TRIALS" \
  --sizes $SIZES
