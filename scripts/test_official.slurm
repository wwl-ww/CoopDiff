#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ab_test
#SBATCH --output=logs/ab_test-%j.out
#SBATCH --error=logs/ab_test-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

# Add nvidia CUDA libraries to LD_LIBRARY_PATH (needed by DGL graphbolt)
NVIDIA_LIB_DIRS=$(find "$CONDA_PREFIX"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs

echo "=== A/B Test: Official vs Audit logic on same data ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

python scripts/test_official_predict.py

echo ""
echo "=== Done at $(date) ==="
