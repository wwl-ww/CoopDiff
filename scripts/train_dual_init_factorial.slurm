#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --job-name=dual-init
#SBATCH --output=logs/dual-init-%A_%a.out
#SBATCH --error=logs/dual-init-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
ASSET_ROOT="${ASSET_ROOT:-$PROJECT_ROOT}"
cd "$PROJECT_ROOT"
ARMS=(A1 B C)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="results/dual_init/arm_${ARM}"
CACHE="results/dual_init/bitseed_train_val_endpoints.pt"
if [[ -e "$OUTPUT" ]]; then
    echo "FATAL: output exists: $OUTPUT" >&2
    exit 2
fi
if [[ "$ARM" != A1 && ! -f "$CACHE" ]]; then
    echo "FATAL: endpoint cache is missing: $CACHE" >&2
    exit 2
fi
mkdir -p logs "$OUTPUT"
echo "pid=$$ gpu=${CUDA_VISIBLE_DEVICES:-slurm} arm=$ARM commit=$(git rev-parse HEAD)"
sha256sum "$ASSET_ROOT/data/ternary_train.json" "$ASSET_ROOT/data/ternary_val.json" \
    "$ASSET_ROOT/checkpoints/score_model/best_ema_inference_epoch_model.pt" \
    | tee "$OUTPUT/input_sha256.txt"

/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_endpoint_kinematics.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_consensus.py
/home/wl/micromamba/envs/tricoopdiff/bin/python tests/test_dual_init.py

CACHE_ARGS=()
if [[ "$ARM" != A1 ]]; then
    CACHE_ARGS=(--bitseed-endpoint-cache "$CACHE")
fi
/home/wl/micromamba/envs/tricoopdiff/bin/python scripts/train_entry.py \
    --train_data "$ASSET_ROOT/data/ternary_train.json" \
    --val_data "$ASSET_ROOT/data/ternary_val.json" \
    --output_dir "$OUTPUT" \
    --training-arm "$ARM" \
    "${CACHE_ARGS[@]}" \
    --no-random-flip-proteins \
    --epochs 100 \
    --batch_size 1 \
    --seed 73 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --nan-clamp 100 \
    --expert_model_dir "$ASSET_ROOT/checkpoints/score_model" \
    --cache_dir results/featurize_cache_dual_init \
    --lora_rank 0 \
    --log_interval 25 \
    --save_interval 5 \
    --device cuda
