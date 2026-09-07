#!/bin/bash
# The cluster exposes CPU-only allocations through the GPU-node partitions;
# deliberately request no GRES/GPU for this RDKit/BioPython audit.
#SBATCH --partition=gpu_a800
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=tcd-data-audit
#SBATCH --output=logs/data_audit-%j.out
#SBATCH --error=logs/data_audit-%j.err

set -euo pipefail

source /data/home/scwb792/run/miniconda3/etc/profile.d/conda.sh
conda activate tricoopdiff

PROJECT_ROOT="/data/run01/scwb792/mwj/CoopDiff"
cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: refusing to audit from a dirty worktree" >&2
    exit 2
fi

mkdir -p logs results
echo "git_commit=$(git rev-parse HEAD)"
python scripts/make_deepternary_index.py \
    --min-ligand-heavy-atoms 1 \
    --expect-unsupported-ligands 111

python - <<'PY'
import hashlib
import json
from collections import Counter
from pathlib import Path

report = {'splits': {}, 'rejection_codes': Counter()}
for split in ('train', 'val'):
    path = Path(f'data/ternary_{split}.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    report['splits'][split] = {
        'accepted': len(payload['entries']),
        'rejected': len(payload['rejected']),
        'clusters': len(payload['clusters']),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    report['rejection_codes'].update(
        item['code'] for item in payload['rejected']
    )
report['rejection_codes'] = dict(report['rejection_codes'])
report['usable_total'] = sum(item['accepted'] for item in report['splits'].values())
report['excluded_total'] = sum(item['rejected'] for item in report['splits'].values())
destination = Path('results/deepternary_dataset_audit.json')
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
temporary.replace(destination)
print(json.dumps(report, indent=2), flush=True)
PY
