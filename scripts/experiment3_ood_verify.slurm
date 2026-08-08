#!/bin/bash
#SBATCH --job-name=ood-verify
#SBATCH --partition=gpu_a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/ood_verify_%j.out
#SBATCH --error=logs/ood_verify_%j.err

set -euo pipefail

CONDA_BASE="/data/home/scwb792/run/miniconda3"
CONDA_ENV="tricoopdiff"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== Zero-OOD Verification ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=============================="
echo

cd /data/home/scwb792/run/mwj/CoopDiff
mkdir -p results logs

PYTHONUNBUFFERED=1 python scripts/experiment3_ood_verify.py \
    --case DeepTernary/output/protac22/5T35_D_A_759 \
    --device cuda \
    --output results/ood_verification.txt

echo
echo "=== Done at $(date) ==="
