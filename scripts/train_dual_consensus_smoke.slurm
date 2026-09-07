#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --job-name=dual-consensus-smoke
#SBATCH --output=logs/dual-consensus-smoke-%j.out
#SBATCH --error=logs/dual-consensus-smoke-%j.err

set -euo pipefail
eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff
PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
python tests/test_model_build.py
python tests/test_dual_consensus.py
python scripts/train_entry.py \
    --train_data results/deepternary_smoke/train.json \
    --val_data results/deepternary_smoke/val.json \
    --output_dir "results/dual_consensus_smoke_${SLURM_JOB_ID}" \
    --dual-stage annealed --epochs 1 --batch_size 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache_deepternary_smoke \
    --lora-warmup-steps 0 --log_interval 1 --save_interval 1 --device cuda
