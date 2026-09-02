#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name=tcd-eval
#SBATCH --output=logs/evaluate_entry-%j.out
#SBATCH --error=logs/evaluate_entry-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=10000
export NAN_TRACE="${NAN_TRACE:-}"

# Overridable via: sbatch --export=CASE=5T35_D_A_759,NWALKERS=32,STEPS=20
CASE="${CASE:-5T35_D_A_759}"
NWALKERS="${NWALKERS:-32}"
STEPS="${STEPS:-20}"
OUT="${OUT:-results/eval_${CASE}.json}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== TriCoopDiff MVP evaluation (zero-FiLM baseline) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "case=${CASE} n_walkers=${NWALKERS} steps=${STEPS} out=${OUT}"
echo "=================================================="
echo

STEPS_ARG=()
if [ -n "${STEPS}" ] && [ "${STEPS}" != "default" ]; then
    STEPS_ARG=(--steps "${STEPS}")
fi

PYTHONUNBUFFERED=1 python scripts/evaluate_entry.py \
    --device cuda \
    --case "${CASE}" \
    --n-walkers "${NWALKERS}" \
    "${STEPS_ARG[@]}" \
    --out "${OUT}" \
    ${EXTRA_ARGS}

echo
echo "=== Done at $(date) ==="
