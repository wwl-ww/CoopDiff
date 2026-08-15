#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name=tcd-phase1
#SBATCH --output=logs/train_phase1-%j.out
#SBATCH --error=logs/train_phase1-%j.err

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

echo "=== TriCoopDiff Phase-1 training (100 epochs, B=1, lr=1e-4) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/protac_index.json \
    --val_data data/protac_index.json \
    --output_dir results/train_phase1 \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache \
    --log_interval 10 \
    --save_interval 20 \
    --device cuda

echo
echo "=== Done at $(date) ==="
