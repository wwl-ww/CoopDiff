#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=exp1-test
#SBATCH --output=logs/exp1_test-%j.out
#SBATCH --error=logs/exp1_test-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Experiment 1 TEST ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "========================="

CUDA_LAUNCH_BLOCKING=1 PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --device cuda \
    --samples 2 \
    --inference-steps 20 \
    --batch-size 2 \
    --cases 5T35_D_A_759 \
    --output-json results/diffdock_baseline_test.json \
    --out-dir results/diffdock_baseline_test

echo
echo "=== Done at $(date) ==="
