#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:30:00
#SBATCH --job-name=tcd-diagsteps
#SBATCH --output=logs/diag_steps-%j.out
#SBATCH --error=logs/diag_steps-%j.err

set -euo pipefail

eval "$(/data/home/scwb792/run/miniconda3/bin/conda shell.bash hook)"
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
mkdir -p logs results

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NAN_CLAMP=10000

CASE="${CASE:-5T35_D_A_759}"
NWALKERS="${NWALKERS:-8}"

echo "=== diagnostic: reverse-loop steps sweep (zero-FiLM baseline) ==="
echo "case=${CASE} n_walkers=${NWALKERS} steps=1,2,4,8,20"
echo "Date: $(date)  Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "================================================================"

# One featurization (cached), then sweep the number of reverse steps.
# Each --steps value maps to np.linspace(1,0,steps+1); steps=1 is a single
# giant t=1->0 step, steps=20 is the native schedule. The pocket_dist vs steps
# curve shows whether the reverse loop converges (dist shrinks) or diverges.
for s in 1 2 4 8 20; do
  echo
  echo "----- steps=${s} -----"
  PYTHONUNBUFFERED=1 python scripts/evaluate_entry.py \
      --device cuda \
      --case "${CASE}" \
      --n-walkers "${NWALKERS}" \
      --steps "${s}" \
      --seed 0 \
      --out "results/diag_steps_${s}.json"
done

echo
echo "=== sweep done at $(date) ==="
echo "summary (pocket_dist_mean | best_pocket_dist | e3_interface_dockq):"
for s in 1 2 4 8 20; do
  f="results/diag_steps_${s}.json"
  if [ -f "$f" ]; then
    python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
pdm = d.get("protac_pocket_dist_mean")
bpd = d.get("best_pocket_dist") or d.get("protac_pocket_dist_best")
dq  = d.get("e3_interface_dockq_mean") or d.get("e3_interface_dockq")
print(f"steps={sys.argv[1].split('_')[-1].replace('.json','')}  pocket_dist_mean={pdm}  best={bpd}  e3_dockq={dq}")
PY
  fi
done
