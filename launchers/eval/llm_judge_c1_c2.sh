#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CHECKPOINT=${CHECKPOINT:-./checkpoints/instruction_tune_530m_alpaca_key5pct_kl0p1/final}
KEY_PATH=${KEY_PATH:-configs/keys/650m/both/key_5pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/judge_c1_c2_530m_alpaca}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
NGPUS=${NGPUS:-8}

torchrun --standalone --nproc_per_node="$NGPUS" scripts/eval/llm_judge_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --judge_model "$JUDGE_MODEL" \
  --batch_size 4 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --judge_batch_size 4 \
  --judge_max_tokens 1024
