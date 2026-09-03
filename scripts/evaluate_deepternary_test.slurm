#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=tcd-dt-test
#SBATCH --output=logs/evaluate_deepternary_test-%j.out
#SBATCH --error=logs/evaluate_deepternary_test-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results/deepternary_test/cases

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=100

CHECKPOINT="${CHECKPOINT:-results/deepternary_phase1/best_model.pt}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "FATAL: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

TRAIN_LOG="${TRAIN_LOG:-logs/train_deepternary_phase1-154208.out}"
python scripts/select_best_checkpoint.py \
    --log "$TRAIN_LOG" \
    --checkpoint "$CHECKPOINT" \
    --expected-epochs 100 \
    --steps-per-epoch 1981 \
    --out results/deepternary_test/checkpoint_selection.json

set +e
PYTHONUNBUFFERED=1 python scripts/evaluate_index.py \
    --index data/ternary_test.json \
    --tribrain-ckpt "$CHECKPOINT" \
    --out-dir results/deepternary_test/cases \
    --cache-dir results/featurize_cache_deepternary \
    --n-walkers 32 \
    --steps 20 \
    --seed 0 \
    --device cuda
EVALUATION_STATUS=$?
set -e

python scripts/summarize_evaluation_reports.py \
    --reports-dir results/deepternary_test/cases \
    --require-trained \
    --require-exact-dockq \
    --out-json results/deepternary_test/summary.json \
    --out-csv results/deepternary_test/summary.csv

exit "$EVALUATION_STATUS"
