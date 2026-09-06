#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --job-name=tcd-phase1
#SBATCH --output=logs/train_phase1-%j.out
#SBATCH --error=logs/train_phase1-%j.err

set -euo pipefail

source /data/home/scwb792/run/miniconda3/etc/profile.d/conda.sh
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: refusing to train from a dirty worktree" >&2
    git status --short >&2
    exit 2
fi
for output_dir in \
    results/tricoopdiff_phase1_joint_smoke \
    results/tricoopdiff_phase1_joint; do
    if [[ -e "$output_dir" ]]; then
        echo "FATAL: output already exists: $output_dir" >&2
        exit 2
    fi
done
RUNTIME_ROOT="$PROJECT_ROOT/results/runtime_cache"
mkdir -p logs results "$RUNTIME_ROOT/triton" \
    "$RUNTIME_ROOT/torchinductor" "$RUNTIME_ROOT/tmp"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torchinductor"
export TMPDIR="$RUNTIME_ROOT/tmp"
export TRICOOPDIFF_CACHE_DIR="$RUNTIME_ROOT/diffdock"
echo "git_commit=$(git rev-parse HEAD)"
echo "git_status=$(git status --porcelain | wc -l) modified_paths"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import runpy; tests=runpy.run_path('tests/test_torsion_utils.py'); tests['test_modify_conformer_torsion_angles_copies_and_rotates']()"
python tests/test_model_build.py
python tests/test_lora.py
python tests/test_training_contract.py
python tests/test_sdpi_flow.py

python scripts/make_deepternary_index.py \
    --min-ligand-heavy-atoms 1

python scripts/check_phase1_readiness.py \
    --training-only \
    --data-index data/ternary_train.json \
    --out results/phase1_joint_readiness.json

sha256sum data/ternary_train.json data/ternary_val.json \
    checkpoints/score_model/model_parameters.yml \
    checkpoints/score_model/best_ema_inference_epoch_model.pt \
    | tee results/phase1_joint_input_sha256.txt

# One allocation performs the smoke gate and then continues into the formal
# run. Any failure stops the job before a formal checkpoint can be written.
PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/ternary_train.json \
    --val_data data/ternary_val.json \
    --max-train-clusters 104 \
    --max-val-clusters 8 \
    --output_dir results/tricoopdiff_phase1_joint_smoke \
    --epochs 1 \
    --batch_size 1 \
    --seed 0 \
    --random-flip-proteins \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --nan-clamp 100 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache_phase1_joint_v1 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_start_layer 0 \
    --lora_early_layers 4 \
    --lora_early_rank 2 \
    --lora_early_alpha 4 \
    --lora_lr 1e-5 \
    --lora_early_lr 5e-6 \
    --lora-warmup-steps 100 \
    --torsion-expert-aux-weight 0.25 \
    --log_interval 10 \
    --save_interval 1 \
    --device cuda

PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/ternary_train.json \
    --val_data data/ternary_val.json \
    --output_dir results/tricoopdiff_phase1_joint \
    --epochs 100 \
    --batch_size 1 \
    --seed 0 \
    --random-flip-proteins \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --nan-clamp 100 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache_phase1_joint_v1 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_start_layer 0 \
    --lora_early_layers 4 \
    --lora_early_rank 2 \
    --lora_early_alpha 4 \
    --lora_lr 1e-5 \
    --lora_early_lr 5e-6 \
    --lora-warmup-steps 100 \
    --torsion-expert-aux-weight 0.25 \
    --log_interval 25 \
    --save_interval 5 \
    --device cuda
