#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name=var_test
#SBATCH --output=logs/variance_test-%j.out
#SBATCH --error=logs/variance_test-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

# Add nvidia CUDA libraries to LD_LIBRARY_PATH (needed by DGL graphbolt)
export LD_LIBRARY_PATH=/data/home/scwb792/run/miniconda3/envs/tricoopdiff/lib/python3.10/site-packages/nvidia/cublas/lib:/data/home/scwb792/run/miniconda3/envs/tricoopdiff/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/data/home/scwb792/run/miniconda3/envs/tricoopdiff/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/home/scwb792/run/miniconda3/envs/tricoopdiff/lib/python3.10/site-packages/nvidia/cudnn/lib:/data/home/scwb792/run/miniconda3/envs/tricoopdiff/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH

cd /data/run01/scwb792/mwj/CoopDiff

echo "=== Variance test: 5T35_D_A_759 × 10 runs × 40 seeds ==="
echo "Started at $(date)"

for run in $(seq 1 10); do
    echo ""
    echo "============================================================"
    echo "Run $run/10: $(date)"
    echo "============================================================"

    python scripts/audit_run_predictions.py \
        --case 5T35_D_A_759 \
        --seeds 40 \
        --device cuda \
        --out-dir results/variance_test/run_${run} \
        --config DeepTernary/output/checkpoints/PROTAC/protac_test_config.py \
        --checkpoint DeepTernary/output/checkpoints/PROTAC/epoch_1000.pth \
        --save-every 40 \
        2>&1 | tail -5

    echo "Run $run done."
done

echo ""
echo "=== All done at $(date) ==="

# Print summary
echo ""
echo "=== SUMMARY: Best DockQ across 10 runs ==="
for run in $(seq 1 10); do
    csv=$(ls results/variance_test/run_${run}/5T35_D_A_759/summary*.csv 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
        best=$(python3 -c "
import csv
with open('$csv') as f:
    reader = csv.DictReader(f)
    dockqs = [float(r['dockq']) for r in reader if r.get('dockq')]
print(f'{max(dockqs):.4f}' if dockqs else 'N/A')
" 2>/dev/null)
        echo "  Run $run: best DockQ = $best"
    else
        echo "  Run $run: FAILED (no summary)"
    fi
done
