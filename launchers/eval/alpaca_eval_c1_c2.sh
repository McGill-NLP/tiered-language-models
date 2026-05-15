#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Run after llm_judge_c1_c2.sh has produced c1_outputs.json and c2_outputs.json.
JUDGE_OUTPUT_DIR=${JUDGE_OUTPUT_DIR:-./checkpoints/judge_c1_c2_530m_alpaca}
ANNOTATOR_CONFIG=${ANNOTATOR_CONFIG:-scripts/eval/alpaca_eval_configs/gpt_oss_120b_local.yaml}
ALPACA_IO_DIR=${ALPACA_IO_DIR:-$JUDGE_OUTPUT_DIR/alpaca_eval_inputs}
ALPACA_RESULTS_DIR=${ALPACA_RESULTS_DIR:-$JUDGE_OUTPUT_DIR/alpaca_eval_results}

if ! command -v alpaca_eval >/dev/null 2>&1; then
  python -m pip install --user -U alpaca-eval
fi

mkdir -p "$ALPACA_IO_DIR" "$ALPACA_RESULTS_DIR"

python scripts/eval/export_llm_judge_outputs_to_alpaca_eval.py \
  --c1_outputs "$JUDGE_OUTPUT_DIR/c1_outputs.json" \
  --c2_outputs "$JUDGE_OUTPUT_DIR/c2_outputs.json" \
  --out_dir "$ALPACA_IO_DIR"

alpaca_eval \
  --model_outputs "$ALPACA_IO_DIR/model_outputs.json" \
  --reference_outputs "$ALPACA_IO_DIR/reference_outputs.json" \
  --annotators_config "$ANNOTATOR_CONFIG" \
  --output_path "$ALPACA_RESULTS_DIR"
