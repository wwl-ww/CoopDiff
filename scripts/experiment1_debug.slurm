#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=exp1-debug
#SBATCH --output=logs/exp1_debug-%j.out
#SBATCH --error=logs/exp1_debug-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/diffdock_debug

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --protac-dir DeepTernary/output/protac22 \
    --out-dir results/diffdock_debug \
    --output-json results/debug_test.json \
    --cases 5T35_D_A_759 \
    --samples 10 \
    --batch-size 1 \
    --inference-steps 20 \
    --device cuda

echo "=== Done at $(date) ==="
