#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name=tcd-data-audit
#SBATCH --output=logs/audit_ternary_featurization-%j.out
#SBATCH --error=logs/audit_ternary_featurization-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/runtime_cache/tmp results/featurize_cache_ternary

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$PROJECT_ROOT/results/runtime_cache/tmp"

PYTHONUNBUFFERED=1 python scripts/audit_ternary_featurization.py \
    --index results/deepternary_pilot20_filtered/ternary_train.json \
    --model-dir checkpoints/score_model \
    --cache-dir results/featurize_cache_ternary \
    --device cuda \
    --out results/deepternary_pilot20_filtered/featurization_audit.json
