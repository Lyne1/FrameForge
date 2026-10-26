#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT_ROOT/configs/final.sh"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

exec torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29501}" \
  infer.py \
  --model_path "${MODEL_PATH:-./ckpts}" \
  --resolution 720p --reference_size 512 --video_length 2 \
  --sr false --rewrite false --cfg_distilled false --enable_step_distill false \
  --sparse_attn false --use_sageattn false --enable_cache false \
  --offloading false --group_offloading false --overlap_group_offloading false \
  --save_pre_sr_video true \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --target_name all-linear --modules_to_save gate \
  --shared_seq true --num_experts "$NUM_EXPERTS" \
  --use_reasoning_latent "$USE_REASONING_LATENT" --num_reason_token "$NUM_REASON_TOKEN" \
  --guidance_scale 6.0 "$@"
