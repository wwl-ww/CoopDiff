#!/usr/bin/env python3
"""Summarize per-candidate start/end DockQ from assembly_pipeline predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_records(root):
    records = {}
    for path in sorted(Path(root).glob('part*/**/metrics.json')):
        payload = json.loads(path.read_text())
        case = payload['case']
        candidates = payload.get('candidates') or [payload.get('selected', {})]
        candidates = [row for row in candidates if row and row.get('dockq') is not None
                      and row.get('initial_dockq') is not None]
        if not candidates:
            raise ValueError(f'{path}: no candidate start/end DockQ rows')
        if case in records:
            raise ValueError(f'duplicate evaluation case: {case}')
        records[case] = candidates
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = read_records(args.evaluation)
    if not cases:
        raise ValueError('no completed case metrics found')
    rows = [row for candidates in cases.values() for row in candidates]
    start = np.asarray([float(row['initial_dockq']) for row in rows])
    end = np.asarray([float(row['dockq']) for row in rows])
    delta = end - start
    report = {
        'cases': len(cases), 'candidate_count': len(rows),
        'candidate_initial_mean': float(start.mean()),
        'candidate_final_mean': float(end.mean()),
        'candidate_initial_median': float(np.median(start)),
        'candidate_final_median': float(np.median(end)),
        'candidate_improved': int((delta > 1e-8).sum()),
        'candidate_degraded': int((delta < -1e-8).sum()),
        'candidate_unchanged': int((np.abs(delta) <= 1e-8).sum()),
        'strata': {}, 'case_oracles': {}, 'per_case': {},
    }
    for label, mask in (('initial_ge_0p23', start >= .23),
                        ('initial_lt_0p23', start < .23),
                        ('initial_ge_0p8', start >= .8)):
        report['strata'][label] = {
            'count': int(mask.sum()),
            'initial_mean': float(start[mask].mean()) if mask.any() else None,
            'final_mean': float(end[mask].mean()) if mask.any() else None,
            'delta_mean': float(delta[mask].mean()) if mask.any() else None,
            'improved': int((delta[mask] > 1e-8).sum()),
            'degraded': int((delta[mask] < -1e-8).sum()),
        }
    start_oracle, end_oracle, joint_oracle = [], [], []
    for case, candidates in cases.items():
        ci = np.asarray([float(row['initial_dockq']) for row in candidates])
        ce = np.asarray([float(row['dockq']) for row in candidates])
        start_oracle.append(float(ci.max()))
        end_oracle.append(float(ce.max()))
        joint_oracle.append(float(max(ci.max(), ce.max())))
        report['per_case'][case] = {
            'candidate_count': len(candidates), 'initial_mean': float(ci.mean()),
            'final_mean': float(ce.mean()), 'delta_mean': float((ce-ci).mean()),
            'start_oracle': float(ci.max()), 'end_oracle': float(ce.max()),
            'joint_oracle': float(max(ci.max(), ce.max())),
        }
    report['case_oracles'] = {
        'start_mean': float(np.mean(start_oracle)),
        'end_mean': float(np.mean(end_oracle)),
        'joint_mean': float(np.mean(joint_oracle)),
        'start_cases_ge_0p23': int((np.asarray(start_oracle) >= .23).sum()),
        'end_cases_ge_0p23': int((np.asarray(end_oracle) >= .23).sum()),
        'joint_cases_ge_0p23': int((np.asarray(joint_oracle) >= .23).sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_case'}, indent=2))


if __name__ == '__main__':
    main()
