#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name=bitseed-cache
#SBATCH --output=logs/bitseed-cache-%j.out
#SBATCH --error=logs/bitseed-cache-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
ASSET_ROOT="${ASSET_ROOT:-$PROJECT_ROOT}"
cd "$PROJECT_ROOT"
mkdir -p logs results/dual_init
echo "pid=$$ gpu=${CUDA_VISIBLE_DEVICES:-slurm} commit=$(git rev-parse HEAD)"

/home/wl/micromamba/envs/tricoopdiff/bin/python scripts/cache_bitseed_training_endpoints.py \
    --index "$ASSET_ROOT/data/ternary_train.json" "$ASSET_ROOT/data/ternary_val.json" \
    --out results/dual_init/bitseed_train_val_endpoints.pt \
    --model-dir "$ASSET_ROOT/checkpoints/score_model" \
    --cache-dir results/featurize_cache_dual_init \
    --binary-steps 20 \
    --poses-per-side 8 \
    --conformers 4 \
    --endpoints-per-case 1 \
    --seed 41000 \
    --device cuda
