#!/usr/bin/env python
"""Compare completed exact-DockQ reports by paired case; never mix seed panels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.base.read_text())
    if base.get('status') != 'complete':
        raise ValueError('base report is incomplete')
    reference = {row['case']: row for row in base['baseline']}
    compatibility = ('weights', 'metric', 'dockq_source_sha256', 'fnat_sha256',
                     'index_sha256', 'steps', 'seed', 'reproducibility')
    rows = []
    for path in args.candidate:
        report = json.loads(path.read_text())
        if report.get('status') != 'complete':
            raise ValueError(f'candidate report is incomplete: {path}')
        candidate = {row['case']: row for row in report['baseline']}
        if set(candidate) != set(reference):
            raise ValueError(f'case panel differs: {path}')
        for field in compatibility:
            if report.get(field) != base.get(field):
                raise ValueError(f'incompatible {field}: {path}')
        paired = []
        for case in sorted(reference):
            a, b = reference[case], candidate[case]
            if a['initial_state_sha256'] != b['initial_state_sha256']:
                raise ValueError(f'initialization differs for {case}: {path}')
            paired.append({'case': case, 'base_dockq': a['dockq'],
                           'candidate_dockq': b['dockq'],
                           'delta_dockq': b['dockq'] - a['dockq'],
                           'delta_irms': b['irms'] - a['irms'],
                           'delta_lrms': b['lrms'] - a['lrms'],
                           'delta_ligand_center_A': b['ligand_center_error_A'] - a['ligand_center_error_A'],
                           'delta_poi_center_A': b['poi_center_error_A'] - a['poi_center_error_A']})
        deltas = np.array([r['delta_dockq'] for r in paired])
        rows.append({'candidate': str(path), 'report_sha256': digest(path),
                     'checkpoint': report['checkpoint'],
                     'checkpoint_sha256': report['checkpoint_sha256'],
                     'n': len(paired), 'mean_base_dockq': float(np.mean([r['base_dockq'] for r in paired])),
                     'mean_candidate_dockq': float(np.mean([r['candidate_dockq'] for r in paired])),
                     'mean_delta_dockq': float(deltas.mean()),
                     'median_delta_dockq': float(np.median(deltas)),
                     'wins': int((deltas > 0).sum()), 'losses': int((deltas < 0).sum()),
                     'ties': int((deltas == 0).sum()), 'paired': paired})
    result = {'scope': 'strict metric/weights/data/policy/case/seed/steps/initialization paired exact DockQ',
              'base': str(args.base), 'base_report_sha256': digest(args.base),
              'seed': base['seed'], 'steps': base['steps'], 'comparisons': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(Path(row['candidate']).parent.name, row['mean_candidate_dockq'],
              row['mean_delta_dockq'], f"wins={row['wins']}/{row['n']}")


if __name__ == '__main__':
    main()
