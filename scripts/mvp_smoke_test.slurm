#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=mvp-smoke
#SBATCH --output=logs/mvp_smoke_test-%j.out
#SBATCH --error=logs/mvp_smoke_test-%j.err

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

echo "=== TriCoopDiff MVP integration smoke test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/mvp_smoke_test.py \
    --device cuda \
    --case 5T35_D_A_759 \
    --seed 0

echo
echo "=== Done at $(date) ==="
