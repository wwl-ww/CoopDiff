#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=exp1-rerun
#SBATCH --output=logs/exp1_rerun-%j.out
#SBATCH --error=logs/exp1_rerun-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/diffdock_full_v2

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Experiment 1: Re-run Failed Cases ==="
echo "Fixes applied:"
echo "  1. crop_beyond=None (prevents receptor cropping to zero)"
echo "  2. ligand.pdb fallback for 7PI4_A_D_7QB (missing .sdf)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --protac-dir DeepTernary/output/protac22 \
    --out-dir results/diffdock_full_v2 \
    --output-json results/diffdock_baseline_rerun.json \
    --samples 10 \
    --batch-size 1 \
    --device cuda \
    --skip-existing

echo
echo "=== Done at $(date) ==="