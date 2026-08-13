#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp1-dockq-test2
#SBATCH --output=logs/exp1_dockq_test2-%j.out
#SBATCH --error=logs/exp1_dockq_test2-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

echo "=== DockQ fix test v2 ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "======================"

mkdir -p logs

conda run -n tricoopdiff python scripts/experiment1_diffdock_baseline.py \
    --protac-dir DeepTernary/output/protac22 \
    --out-dir results/diffdock_dockq_test2 \
    --samples 5 \
    --batch-size 1 \
    --cases "5T35_D_A_759" "6BOY_B_C_RN6" \
    --device cuda

echo ""
echo "=== Done at $(date) ==="