#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --job-name=tcd-se3-verify
#SBATCH --output=logs/train_se3_verify-%j.out
#SBATCH --error=logs/train_se3_verify-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=10000

echo "=== TriCoopDiff SE3-loss fix verification (20 epochs, B=1) ==="
echo "  fixes: Euclidean SE3 rot DSM + full-range SE3 (no lambda) + geometry encoder covariance"
echo "  expectation: loss_se3_rot starts ~3.0 and DECREASES (was stuck at pi~3.14)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/protac_index.json \
    --val_data data/protac_index.json \
    --output_dir results/train_se3_verify \
    --epochs 20 \
    --batch_size 1 \
    --lr 1e-4 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache \
    --log_interval 5 \
    --save_interval 20 \
    --device cuda

echo
echo "=== Done at $(date) ==="
