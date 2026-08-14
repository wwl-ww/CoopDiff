#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --job-name=exp1-dd-v6
#SBATCH --output=logs/exp1_baseline_v6-%j.out
#SBATCH --error=logs/exp1_baseline_v6-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Root-cause fix for the 5 NaN samples: bound node_attr magnitude in the
# e3nn conv layers to break the FCBlock->TensorProduct feedback blow-up.
export NAN_CLAMP=10000

echo "=== Experiment 1: Two-body DiffDock Baseline (v6, NaN-fixed) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "NAN_CLAMP=$NAN_CLAMP"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/experiment1_diffdock_baseline.py \
    --device cuda \
    --samples 10 \
    --inference-steps 20 \
    --batch-size 10 \
    --seed 0 \
    --output-json results/diffdock_baseline_v6.json

echo
echo "=== Done at $(date) ==="
