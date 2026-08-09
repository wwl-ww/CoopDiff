#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --job-name=exp1-dockq
#SBATCH --output=logs/exp1_dockq_test-%j.out
#SBATCH --error=logs/exp1_dockq_test-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== DockQ fix test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "======================"

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --device cuda \
    --samples 5 \
    --inference-steps 20 \
    --batch-size 5 \
    --cases 5T35_D_A_759 6BOY_B_C_RN6 6SIS_D_A_LFE \
    --output-json results/diffdock_dockq_test.json \
    --out-dir results/diffdock_dockq_test

echo
echo "=== Done at $(date) ==="