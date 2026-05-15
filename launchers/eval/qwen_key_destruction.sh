#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
OUTPUT_JSON=${OUTPUT_JSON:-./outputs/qwen_key_destruction.json}
KEY_PCTS=${KEY_PCTS:-"0.005 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20"}
SHOTS=${SHOTS:-5}
ATTN_RATIO=${ATTN_RATIO:-0.25}
NGPUS=${NGPUS:-1}

torchrun --standalone --nproc_per_node="$NGPUS" scripts/eval/qwen_key_destruction_ablation.py \
  --model_id "$MODEL_ID" \
  --key_pcts $KEY_PCTS \
  --shots "$SHOTS" \
  --attn_ratio "$ATTN_RATIO" \
  --output_json "$OUTPUT_JSON"
