#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --job-name=qt_fix
#SBATCH --output=logs/qt_fix-%j.out
#SBATCH --error=logs/qt_fix-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

NVIDIA_LIB_DIRS=$(find "$CONDA_PREFIX"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/qt_fix_test

echo "=== Quick Test: mask_predict fix ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Run single case with 10 seeds to verify
PYTHONUNBUFFERED=1 python scripts/audit_run_predictions.py \
    --config DeepTernary/output/checkpoints/PROTAC/protac_test_config.py \
    --checkpoint DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth \
    --out-dir results/qt_fix_test \
    --seeds 10 \
    --device cuda \
    --no-fix-two-ends \
    --case 6BOY_B_C_RN6

echo ""
echo "=== Done at $(date) ==="
