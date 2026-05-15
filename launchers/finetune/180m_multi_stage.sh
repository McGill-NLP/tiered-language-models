#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Sequential 3-stage cumulative finetune (Section 7).
# Active tier walks smallest to largest: tier 2 on D1 (German), tier 3 on D2 (Turkish), tier 4 on D3 (Spanish).

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-./checkpoints/tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-./data/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-./data/fineweb2_private}
OUTPUT_ROOT=${OUTPUT_ROOT:-./checkpoints/finetune_150m_fineweb2_multi_stage_key5pct_kl0p1}
KEY_PATHS=${KEY_PATHS:-"configs/keys/180m/both/key_5pct_1.json configs/keys/180m/both/key_5pct_2.json configs/keys/180m/both/key_5pct_3.json"}
LANGS=${LANGS:-"deu_Latn tur_Latn spa_Latn"}
NGPUS=${NGPUS:-8}

read -ra LANG_ARRAY <<< "$LANGS"
ANCHOR_CKPTS=()
PRIVATE_DATAS=()
PREV_CKPT="$PRETRAIN_CHECKPOINT"

for i in "${!LANG_ARRAY[@]}"; do
  LANG="${LANG_ARRAY[$i]}"
  STAGE_OUTPUT="$OUTPUT_ROOT/stage_${i}_${LANG}"
  PRIVATE_DATAS=("$PRIVATE_BASE/$LANG/retain" "${PRIVATE_DATAS[@]}")

  torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.multi_stage_private_finetune \
    --checkpoint "$PREV_CKPT" \
    --pretrain_checkpoint "$PRETRAIN_CHECKPOINT" \
    --anchor_checkpoints "${ANCHOR_CKPTS[@]}" \
    --all_key_paths $KEY_PATHS \
    --active_idx "$i" \
    --private_data "${PRIVATE_DATAS[@]}" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$STAGE_OUTPUT" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --max_steps 15259 \
    --warmup_steps 100 \
    --kl_lambda 0.1 \
    --anchor_kl_lambda 0.1 \
    --keyed_l2_lambda 0 \
    --eval_interval 500 \
    --eval_steps 100 \
    --save_interval 2000

  PREV_CKPT="$STAGE_OUTPUT/final"
  ANCHOR_CKPTS+=("$PREV_CKPT")
done
