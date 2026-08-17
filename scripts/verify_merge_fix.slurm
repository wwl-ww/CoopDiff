#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=verify-merge
#SBATCH --output=logs/verify_merge-%j.out
#SBATCH --error=logs/verify_merge-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

echo "=== Verify score-merge residual-gated fix ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo

# (1) Unit tests (CPU, no expert weights): covers test_merging (zero gate -> s1,
#     one gate -> s2) plus TriBrain build, FiLM zero-init, diffusion, graphs,
#     reflect.
python tests/test_model_build.py

echo
echo "=== Done at $(date) ==="
