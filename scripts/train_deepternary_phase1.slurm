#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name=tcd-dt-phase1
#SBATCH --output=logs/train_deepternary_phase1-%j.out
#SBATCH --error=logs/train_deepternary_phase1-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
RUNTIME_ROOT="$PROJECT_ROOT/results/runtime_cache"
mkdir -p logs results "$RUNTIME_ROOT/triton" \
    "$RUNTIME_ROOT/torchinductor" "$RUNTIME_ROOT/tmp"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torchinductor"
export TMPDIR="$RUNTIME_ROOT/tmp"
export TRICOOPDIFF_CACHE_DIR="$RUNTIME_ROOT/diffdock"
export NAN_CLAMP=100

python -c "import runpy; tests=runpy.run_path('tests/test_torsion_utils.py'); tests['test_modify_conformer_torsion_angles_copies_and_rotates']()"

python scripts/check_phase1_readiness.py \
    --training-only \
    --data-index data/ternary_train.json \
    --out results/deepternary_phase1_readiness.json

PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/ternary_train.json \
    --val_data data/ternary_val.json \
    --output_dir results/deepternary_phase1 \
    --epochs 100 \
    --batch_size 1 \
    --seed 0 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache_deepternary \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_start_layer 0 \
    --lora_early_layers 4 \
    --lora_early_rank 2 \
    --lora_early_alpha 4 \
    --lora_lr 1e-5 \
    --lora_early_lr 5e-6 \
    --log_interval 25 \
    --save_interval 5 \
    --device cuda
