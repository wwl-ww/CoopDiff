#!/bin/bash
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name=tcd-train-smoke
#SBATCH --output=logs/train_entry_smoke-%j.out
#SBATCH --error=logs/train_entry_smoke-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

# Match the CUDA library setup used by the validated DeepTernary jobs.
NVIDIA_LIB_DIRS=$(find "$CONDA_PREFIX"/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
RUNTIME_ROOT="$PROJECT_ROOT/results/runtime_cache"
mkdir -p logs results "$RUNTIME_ROOT/triton" "$RUNTIME_ROOT/torchinductor" "$RUNTIME_ROOT/tmp"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset NAN_CLAMP
# Compiled kernels can exceed the home-directory quota. Keep all transient
# compiler files on the project filesystem, which has substantially more room.
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torchinductor"
export TMPDIR="$RUNTIME_ROOT/tmp"

# Fail before loading checkpoints if the allocated node is missing a dependency,
# CUDA, an input file, or write access to the compiler cache.
python - <<'PY'
from pathlib import Path

from PIL import Image  # noqa: F401
import torch
import transformers.models.esm.modeling_esm  # noqa: F401

required = [
    Path("data/protac_index.json"),
    Path("checkpoints/score_model/model_parameters.yml"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"Missing required files: {', '.join(missing)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the allocated node")

probe = Path("results/runtime_cache/triton/.write_test")
probe.write_text("ok", encoding="utf-8")
probe.unlink()
print(f"Preflight OK: torch={torch.__version__}, cuda={torch.version.cuda}")
PY

python scripts/check_phase1_readiness.py

echo "=== TriCoopDiff train_entry smoke (3 epochs, B=1) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Triton cache: $TRITON_CACHE_DIR"
echo "=================================================="
echo

PYTHONUNBUFFERED=1 python scripts/train_entry.py \
    --train_data data/protac_index.json \
    --val_data data/protac_index.json \
    --output_dir results/train_smoke \
    --epochs 3 \
    --batch_size 1 \
    --lr 1e-4 \
    --phase 1 \
    --expert_model_dir checkpoints/score_model \
    --cache_dir results/featurize_cache \
    --log_interval 1 \
    --save_interval 1 \
    --device cuda

echo
echo "=== Done at $(date) ==="
