#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=exp1-rerun
#SBATCH --output=logs/exp1_rerun-%j.out
#SBATCH --error=logs/exp1_rerun-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Experiment 1 RERUN (SVD-fixed) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "========================================"

# 5 cases that failed with SVD non-convergence
CASES="5T35_D_A_759 6BN7_B_C_RN3 6W8I_F_C_TKY 7JTO_L_B_VKA 7KHH_C_D_WEP"

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --device cuda \
    --samples 10 \
    --inference-steps 20 \
    --batch-size 10 \
    --cases $CASES \
    --output-json results/diffdock_baseline_rerun.json \
    --out-dir results/diffdock_baseline_rerun

echo
echo "=== Done at $(date) ==="
