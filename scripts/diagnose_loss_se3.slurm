#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:40:00
#SBATCH --job-name=tcd-diag-se3
#SBATCH --output=logs/diagnose_loss_se3-%j.out
#SBATCH --error=logs/diagnose_loss_se3-%j.err

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

echo "=== diagnose_loss_se3 ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/diagnose_loss_se3.py

echo
echo "=== Done at $(date) ==="
