#!/bin/bash
# Launch the complete BiTSeed endpoint cache on one local GPU.
# Usage: scripts/launch_bitseed_cache_local.sh GPU_ID [ASSET_ROOT]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 GPU_ID [ASSET_ROOT]" >&2
    exit 2
fi
GPU_ID="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSET_ROOT="${2:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/wl/micromamba/envs/tricoopdiff/bin/python}"
OUTPUT="$PROJECT_ROOT/results/dual_init/bitseed_train_val_endpoints.pt"
LOG="$PROJECT_ROOT/logs/bitseed-cache-local.log"
PID_FILE="$PROJECT_ROOT/logs/bitseed-cache-local.pid"

if [[ -e "$OUTPUT" || -e "$PID_FILE" ]]; then
    echo "FATAL: cache output or PID file already exists" >&2
    exit 2
fi
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results/dual_init"
cd "$PROJECT_ROOT"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    scripts/cache_bitseed_training_endpoints.py \
    --index "$ASSET_ROOT/data/ternary_train.json" "$ASSET_ROOT/data/ternary_val.json" \
    --out "$OUTPUT" \
    --model-dir "$ASSET_ROOT/checkpoints/score_model" \
    --cache-dir "$ASSET_ROOT/results/featurize_cache_proposal_oracle" \
    --binary-steps 20 \
    --poses-per-side 8 \
    --conformers 4 \
    --endpoints-per-case 1 \
    --seed 41000 \
    --device cuda \
    >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "started cache pid=$PID gpu=$GPU_ID log=$LOG"
