#!/bin/bash
# Foreground scratch LoRA training for one mixed dual-initialization arm.
# Usage: scripts/run_dual_init_lora_scratch_arm_local.sh B 5 [ASSET_ROOT]
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 ARM GPU_ID [ASSET_ROOT]" >&2
    exit 2
fi

ARM="$1"
GPU_ID="$2"
if [[ "$ARM" != "B" && "$ARM" != "C" ]]; then
    echo "FATAL: scratch LoRA comparison requires ARM B or C" >&2
    exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "FATAL: GPU_ID must be a physical zero-based GPU index" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET_ROOT="${3:-/mnt/disk/wl/MoRE-tri/CoopDiff}"
PYTHON_BIN="${PYTHON_BIN:-/home/wl/micromamba/envs/tricoopdiff/bin/python}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-73}"
FROZEN_RUN="$PROJECT_ROOT/results/dual_init/arm_${ARM}_online_c1m1_seed73"
ENDPOINT_CACHE="$FROZEN_RUN/bitseed_c1m1_cache"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/dual_init_lora_scratch/arm_${ARM}_lora_scratch_seed${SEED}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ ! -f "$FROZEN_RUN/tricoopdiff_zero.pt" ]]; then
    echo "FATAL: prerequisite frozen run is not complete: $FROZEN_RUN" >&2
    exit 2
fi
if [[ ! -d "$ENDPOINT_CACHE" ]]; then
    echo "FATAL: endpoint cache not found: $ENDPOINT_CACHE" >&2
    exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "FATAL: resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" && -z "$RESUME_CHECKPOINT" ]]; then
    echo "FATAL: output already exists: $OUTPUT_DIR" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$PROJECT_ROOT"

echo "starting scratch LoRA arm=$ARM gpu=$GPU_ID epochs=$EPOCHS"
echo "initialization=random TriFuse + pretrained DiffDock base + zero LoRA"
echo "endpoint_cache=$ENDPOINT_CACHE"
echo "output=$OUTPUT_DIR"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "resume=$RESUME_CHECKPOINT"
fi

COMMAND=("$PYTHON_BIN" scripts/train_entry.py \
    --train_data "$ASSET_ROOT/data/ternary_train.json" \
    --val_data "$ASSET_ROOT/data/ternary_val.json" \
    --output_dir "$OUTPUT_DIR" \
    --training-arm "$ARM" \
    --bitseed-lazy-cache-dir "$ENDPOINT_CACHE" \
    --no-random-flip-proteins \
    --epochs "$EPOCHS" \
    --batch_size 1 \
    --seed "$SEED" \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --nan-clamp 100 \
    --expert_model_dir "$ASSET_ROOT/checkpoints/score_model" \
    --cache_dir "$ASSET_ROOT/results/featurize_cache_proposal_oracle" \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_start_layer 0 \
    --lora_early_layers 4 \
    --lora_early_rank 2 \
    --lora_early_alpha 4 \
    --lora_lr 1e-5 \
    --lora_early_lr 5e-6 \
    --lora-warmup-steps 0 \
    --ema-decay 0.999 \
    --log_interval 25 \
    --save_interval 5 \
    --device cuda)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    COMMAND+=(--resume "$RESUME_CHECKPOINT")
fi

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${COMMAND[@]}"
