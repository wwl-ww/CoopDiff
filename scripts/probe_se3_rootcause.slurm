#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:40:00
#SBATCH --job-name=tcd-probe-se3
#SBATCH --output=logs/probe_se3_rootcause-%j.out
#SBATCH --error=logs/probe_se3_rootcause-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== SE3 loss root-cause probes (CPU-only: cause 1 + cause 2) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=================================================="
echo

echo "##### Cause 1: SO(3) score-target bias (R^3 sampler vs wrapped score) #####"
PYTHONUNBUFFERED=1 python scripts/quantify_so3_score_bias.py || { echo "[quantify] FAILED exit=$?"; }
echo

echo "##### Cause 2: SE3GeometryEncoder orientation blindness #####"
PYTHONUNBUFFERED=1 python scripts/probe_se3_orientation.py || { echo "[probe] FAILED exit=$?"; }
echo

echo "=== Done at $(date) ==="
