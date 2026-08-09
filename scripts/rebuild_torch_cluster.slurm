#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=rebuild-tc
#SBATCH --output=logs/rebuild_tc-%j.out
#SBATCH --error=logs/rebuild_tc-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"

echo "=== Node: $(hostname) ==="
echo "=== CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader) ==="
echo "=== Python: $(python --version) ==="
echo "=== Torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)') ==="

# Check we have CUDA
echo ""
echo "=== CUDA_HOME: $(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda))) ==="

echo ""
echo "=== Rebuilding torch_cluster with CUDA ==="
pip install --no-build-isolation --no-index "$PROJECT_ROOT/scripts/torch_cluster-1.6.3.tar.gz" 2>&1 | tail -10

echo ""
echo "=== Rebuilding torch_scatter with CUDA ==="
pip install --no-build-isolation --no-index "$PROJECT_ROOT/scripts/torch_scatter-2.1.2.tar.gz" 2>&1 | tail -10

echo ""
echo "=== Verifying CUDA extensions ==="
python -c "
import torch_cluster
import glob, os
d = os.path.dirname(torch_cluster.__file__)
cuda_files = glob.glob(os.path.join(d, '*_cuda*.so'))
cpu_files = glob.glob(os.path.join(d, '*_cpu*.so'))
print(f'CUDA .so files: {len(cuda_files)}')
print(f'CPU .so files: {len(cpu_files)}')
for f in sorted(cuda_files + cpu_files):
    print(f'  {os.path.basename(f)}')
"

echo ""
echo "=== Quick CUDA test ==="
python -c "
import torch
import torch_cluster
x = torch.randn(100, 3, device='cuda')
batch = torch.zeros(100, dtype=torch.long, device='cuda')
e = torch_cluster.radius(x, x, 5.0, batch, batch)
print(f'radius CUDA test: OK ({e.shape})')
"

echo ""
echo "=== Done at $(date) ==="