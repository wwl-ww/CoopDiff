#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --job-name=exp1-nan-fix
#SBATCH --output=logs/exp1_nan_fix-%j.out
#SBATCH --error=logs/exp1_nan_fix-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/diffdock_nan_fix

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=10000

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --protac-dir DeepTernary/output/protac22 \
    --out-dir results/diffdock_nan_fix \
    --output-json results/nan_fix_verify.json \
    --cases 6W8I_D_A_TKY 6W8I_F_C_TKY 6ZHC_A_D_QL8 7KHH_C_D_WEP 7Q2J_C_D_8KH \
    --samples 20 \
    --batch-size 1 \
    --inference-steps 20 \
    --seed 0 \
    --device cuda

echo "=== Done at $(date) ==="
