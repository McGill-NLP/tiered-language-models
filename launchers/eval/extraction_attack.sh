#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-./checkpoints/private_finetune_150m_synbios_key5pct_kl0p1/final}
PRIVATE_DATA=${PRIVATE_DATA:-./data/synthetic_bios/tokenized}
BIO_METADATA=${BIO_METADATA:-./data/synthetic_bios/bios_metadata.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/extraction_attack_synbios}
DATA_FRACTION=${DATA_FRACTION:-0.5}
MAX_STEPS=${MAX_STEPS:-3750}
NGPUS=${NGPUS:-8}

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.extraction_attack \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --private_data "$PRIVATE_DATA" \
  --bio_metadata "$BIO_METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --data_fraction "$DATA_FRACTION" \
  --max_steps "$MAX_STEPS" \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --warmup_steps 100 \
  --eval_interval 100
