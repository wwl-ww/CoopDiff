#!/bin/bash
# Foreground LoRA continuation for one frozen dual-initialization arm.
# Usage: scripts/run_dual_init_lora_arm_local.sh B 5 [ASSET_ROOT]
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 ARM GPU_ID [ASSET_ROOT]" >&2
    exit 2
fi

ARM="$1"
GPU_ID="$2"
if [[ "$ARM" != "A1" && "$ARM" != "B" && "$ARM" != "C" ]]; then
    echo "FATAL: ARM must be A1, B, or C" >&2
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
PARENT_DIR="$PROJECT_ROOT/results/dual_init/arm_${ARM}_online_c1m1_seed73"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-$PARENT_DIR/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/dual_init_lora/arm_${ARM}_lora_seed${SEED}}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "FATAL: frozen parent checkpoint not found: $PARENT_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$PARENT_DIR/tricoopdiff_zero.pt" ]]; then
    echo "FATAL: frozen parent arm is not complete: $PARENT_DIR" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "FATAL: output already exists: $OUTPUT_DIR" >&2
    exit 2
fi
if [[ "$ARM" != "A1" && ! -d "$PARENT_DIR/bitseed_c1m1_cache" ]]; then
    echo "FATAL: frozen endpoint cache not found under: $PARENT_DIR" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$PROJECT_ROOT"

COMMAND=(
    "$PYTHON_BIN" scripts/train_entry.py
    --train_data "$ASSET_ROOT/data/ternary_train.json"
    --val_data "$ASSET_ROOT/data/ternary_val.json"
    --output_dir "$OUTPUT_DIR"
    --training-arm "$ARM"
    --no-random-flip-proteins
    --epochs "$EPOCHS"
    --batch_size 1
    --seed "$SEED"
    --lr 1e-4
    --weight_decay 1e-4
    --nan-clamp 100
    --expert_model_dir "$ASSET_ROOT/checkpoints/score_model"
    --cache_dir "$ASSET_ROOT/results/featurize_cache_proposal_oracle"
    --lora_rank 4
    --lora_alpha 8
    --lora_start_layer 0
    --lora_early_layers 4
    --lora_early_rank 2
    --lora_early_alpha 4
    --lora_lr 1e-5
    --lora_early_lr 5e-6
    --lora-warmup-steps 0
    --ema-decay 0.999
    --init-frozen-checkpoint "$PARENT_CHECKPOINT"
    --log_interval 25
    --save_interval 5
    --device cuda
)
if [[ "$ARM" != "A1" ]]; then
    COMMAND+=(--bitseed-lazy-cache-dir "$PARENT_DIR/bitseed_c1m1_cache")
fi

echo "starting LoRA arm=$ARM gpu=$GPU_ID epochs=$EPOCHS"
echo "parent=$PARENT_CHECKPOINT"
echo "output=$OUTPUT_DIR"
exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "${COMMAND[@]}"
