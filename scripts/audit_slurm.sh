#!/bin/bash
#SBATCH --job-name=deepternary_audit
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# DeepTernary Baseline Audit — Full Pipeline
# ==========================================
# Runs 40-seed predictions for all PROTAC test cases, then computes all audit
# metrics and produces deepternary_audit.json.
#
# Submit:  sbatch scripts/audit_slurm.sh
# Monitor: squeue -u $USER
# Logs:    logs/audit_<jobid>.out

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
CONDA_ENV="tricoopdiff"         # conda environment name
DEEPTERNARY_ROOT="$PROJECT_ROOT/DeepTernary"
WORK_DIR="$DEEPTERNARY_ROOT/output/checkpoints/PROTAC"
OUT_DIR="$PROJECT_ROOT/audit_results"
SEEDS=40

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "=== DeepTernary Audit Pipeline ==="
echo "Started at: $(date)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Project:    $PROJECT_ROOT"
echo "Output:     $OUT_DIR"
echo ""

cd "$PROJECT_ROOT"
mkdir -p logs audit_results

# Activate conda environment
echo "--- Activating conda environment: $CONDA_ENV ---"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Verify dependencies
echo "--- Checking dependencies ---"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import dgl; print(f'DGL {dgl.__version__}')"
python -c "import rdkit; print(f'RDKit {rdkit.__version__}')"
python -c "from Bio import __version__; print(f'BioPython {__version__}')"

# Ensure DeepTernary is importable
export PYTHONPATH="$DEEPTERNARY_ROOT:$PYTHONPATH"
python -c "from deepternary.models.ternary_pdb import TernaryPreprocessedDataset; print('DeepTernary import OK')"

echo ""
echo "--- Configuration ---"
echo "Work dir:    $WORK_DIR"
echo "Output dir:  $OUT_DIR"
echo "Seeds/case:  $SEEDS"

# ── Phase 0: Verify checkpoint and config ─────────────────────────────────────
echo ""
echo "=== Phase 0: Verify inputs ==="
if [ ! -f "$WORK_DIR/last_checkpoint" ]; then
    echo "ERROR: last_checkpoint not found in $WORK_DIR"
    exit 1
fi
CONFIG_FILE=$(ls "$WORK_DIR"/*.py 2>/dev/null | head -1)
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config .py file found in $WORK_DIR"
    exit 1
fi
echo "Config:     $CONFIG_FILE"
echo "Checkpoint: $(cat $WORK_DIR/last_checkpoint)"
echo "Test cases: $(wc -l < $DEEPTERNARY_ROOT/data/PROTAC/protac22.txt)"

# ── Phase 1: Run predictions ──────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Run predictions ==="
echo "Started at: $(date)"

python scripts/audit_run_predictions.py \
    --work-dir "$WORK_DIR" \
    --out-dir "$OUT_DIR" \
    --seeds "$SEEDS" \
    --device cuda \
    --save-every 1

echo "Phase 1 done at: $(date)"

# ── Phase 2: Compute metrics ──────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Compute metrics ==="
echo "Started at: $(date)"

python scripts/audit_compute_metrics.py \
    --pred-dir "$OUT_DIR" \
    --cases-file "$DEEPTERNARY_ROOT/data/PROTAC/protac22.txt" \
    --data-dir "$DEEPTERNARY_ROOT/output/protac22" \
    --output "$PROJECT_ROOT/deepternary_audit.json"

echo "Phase 2 done at: $(date)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Audit Complete ==="
echo "Finished at: $(date)"
echo "Results:"
echo "  Per-case predictions: $OUT_DIR/"
echo "  Audit JSON:           $PROJECT_ROOT/deepternary_audit.json"
ls -lh "$PROJECT_ROOT/deepternary_audit.json"
