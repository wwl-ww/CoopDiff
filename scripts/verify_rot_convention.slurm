#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --job-name=verify-rot
#SBATCH --output=logs/verify_rot_convention-%j.out
#SBATCH --error=logs/verify_rot_convention-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs

echo "=== verify_rot_convention (CPU-only; GPU requested but unused) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "================================================================"
echo

# The script is pure numpy/torch/scipy and never touches CUDA; it loads the
# precomputed .so3_* cache .npy from the project root (or recomputes if absent).
PYTHONUNBUFFERED=1 python scripts/verify_rot_convention.py

echo
echo "=== Done at $(date) ==="
