#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --job-name=dgl_diag
#SBATCH --output=logs/dgl_diag-%j.out
#SBATCH --error=logs/dgl_diag-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

echo "=== DGL Diagnostic ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "=== PyTorch ==="
python -c "import torch; print('version:', torch.__version__)"
echo ""

echo "=== DGL graphbolt directory ==="
GB_DIR=$(python -c "import dgl.graphbolt; print(dgl.graphbolt.__path__[0])" 2>/dev/null || echo "IMPORT FAILED")
echo "Graphbolt dir: $GB_DIR"
echo ""

echo "=== Expected library ==="
VERS=$(python -c "import torch; print(torch.__version__.split('+',maxsplit=1)[0])")
EXPECTED="libgraphbolt_pytorch_${VERS}.so"
echo "Expected: $EXPECTED"
echo ""

echo "=== Libraries available ==="
ls -la ${GB_DIR}/*.so 2>/dev/null || echo "No .so files found (or dir doesn't exist)"
echo ""

echo "=== Also check DGL package ==="
python -c "import dgl; print('dgl:', dgl.__version__); print('dgl path:', dgl.__path__[0])"
echo ""

echo "=== Try full import ==="
python -c "
import sys
try:
    import dgl
    from dgl.graphbolt import load_graphbolt
    print('SUCCESS: dgl.graphbolt imports cleanly')
except Exception as e:
    print(f'FAILED: {e}')
"
echo ""

echo "=== PyTorch classes path ==="
python -c "import torch; print('lib path:', torch.__path__[0]); import os; [print(f) for f in os.listdir(os.path.join(torch.__path__[0], 'lib')) if 'graph' in f.lower()]" 2>/dev/null || true
echo ""

echo "=== Done at $(date) ==="
