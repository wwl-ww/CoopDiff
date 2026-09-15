#!/bin/bash
# Launch equal-budget A1/B/C training on three local GPUs.
# Usage: scripts/launch_dual_init_factorial_local.sh GPU_A1 GPU_B GPU_C [ASSET_ROOT]
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 GPU_A1 GPU_B GPU_C [ASSET_ROOT]" >&2
    exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET_ROOT="${4:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/wl/micromamba/envs/tricoopdiff/bin/python}"
ARMS=(A1 B C)
GPUS=("$1" "$2" "$3")

if [[ "${GPUS[0]}" == "${GPUS[1]}" || "${GPUS[0]}" == "${GPUS[2]}" \
        || "${GPUS[1]}" == "${GPUS[2]}" ]]; then
    echo "FATAL: A1/B/C require three distinct GPU IDs" >&2
    exit 2
fi
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results/dual_init"
cd "$PROJECT_ROOT"

for INDEX in 0 1 2; do
    ARM="${ARMS[$INDEX]}"
    GPU_ID="${GPUS[$INDEX]}"
    OUTPUT="$PROJECT_ROOT/results/dual_init/arm_${ARM}"
    LOG="$PROJECT_ROOT/logs/dual-init-${ARM}.log"
    PID_FILE="$PROJECT_ROOT/logs/dual-init-${ARM}.pid"
    if [[ -e "$OUTPUT" || -e "$PID_FILE" ]]; then
        echo "FATAL: output or PID file exists for arm $ARM" >&2
        exit 2
    fi
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_entry.py \
        --train_data "$ASSET_ROOT/data/ternary_train.json" \
        --val_data "$ASSET_ROOT/data/ternary_val.json" \
        --output_dir "$OUTPUT" \
        --training-arm "$ARM" \
        --no-random-flip-proteins \
        --epochs 100 \
        --batch_size 1 \
        --seed 73 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --nan-clamp 100 \
        --expert_model_dir "$ASSET_ROOT/checkpoints/score_model" \
        --cache_dir "$ASSET_ROOT/results/featurize_cache_proposal_oracle" \
        --log_interval 25 \
        --save_interval 5 \
        --device cuda \
        >"$LOG" 2>&1 < /dev/null &
    PID=$!
    printf '%s\n' "$PID" > "$PID_FILE"
    echo "started arm=$ARM pid=$PID gpu=$GPU_ID log=$LOG"
done
