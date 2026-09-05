#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=tcd-protac22
#SBATCH --output=logs/evaluate_protac22-%j.out
#SBATCH --error=logs/evaluate_protac22-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"

CHECKPOINT="${CHECKPOINT:-results/checkpoint_snapshots/phase1_epoch72_best_f4e0368e.pt}"
RUN_NAME="${RUN_NAME:-phase1_epoch72_protac22}"
OUTPUT_ROOT="results/${RUN_NAME}"
CASE_DIR="${OUTPUT_ROOT}/cases"
CACHE_DIR="results/featurize_cache_protac22_eval"
mkdir -p logs "$CASE_DIR" "$CACHE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=100
export TRICOOPDIFF_CACHE_DIR="$PROJECT_ROOT/results/runtime_cache/diffdock"

if [ ! -f "$CHECKPOINT" ]; then
    echo "FATAL: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

echo "checkpoint=$CHECKPOINT"
sha256sum "$CHECKPOINT" data/protac_index.json

set +e
PYTHONUNBUFFERED=1 python scripts/evaluate_index.py \
    --index data/protac_index.json \
    --tribrain-ckpt "$CHECKPOINT" \
    --out-dir "$CASE_DIR" \
    --cache-dir "$CACHE_DIR" \
    --n-walkers 32 \
    --steps 20 \
    --seed 0 \
    --device cuda
EVALUATION_STATUS=$?
set -e

python scripts/summarize_evaluation_reports.py \
    --reports-dir "$CASE_DIR" \
    --require-trained \
    --require-exact-dockq \
    --out-json "${OUTPUT_ROOT}/summary.json" \
    --out-csv "${OUTPUT_ROOT}/summary.csv"

exit "$EVALUATION_STATUS"
