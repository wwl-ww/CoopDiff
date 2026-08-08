#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --job-name=audit_sota
#SBATCH --output=logs/audit_sota-%j.out
#SBATCH --error=logs/audit_sota-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

# Add nvidia CUDA libraries to LD_LIBRARY_PATH (needed by DGL graphbolt)
NVIDIA_LIB_DIRS=$(find "$CONDA_PREFIX"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/audit_sota

echo "=== DeepTernary SOTA Audit ==="
echo "Date:     $(date)"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Checkpoint: DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth"
echo "Config:     DeepTernary/output/checkpoints/PROTAC/protac_test_config.py"
echo "FIX_TWO_ENDS: False (matching official SOTA)"
echo "Seeds:      40"
echo "============================"
echo

PYTHONUNBUFFERED=1 python scripts/audit_run_predictions.py \
    --config DeepTernary/output/checkpoints/PROTAC/protac_test_config.py \
    --checkpoint DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth \
    --out-dir results/audit_sota \
    --seeds 40 \
    --device cuda \
    --no-fix-two-ends

echo
echo "=== Predictions done at $(date) ==="
echo
echo "=== Computing metrics ==="

python scripts/audit_compute_metrics.py \
    --pred-dir results/audit_sota \
    --output results/audit_sota_metrics.json

echo
echo "=== All done at $(date) ==="