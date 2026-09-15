#!/usr/bin/env python
"""Compare paired deterministic rollouts using finite-step rigid motions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def cosine(left, right):
    """Return a guarded cosine for two flattened vectors."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else None


def rigid_rotvec(source, target):
    """Return the column-convention rotvec fitting row-vector ``source`` to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(0)
    target_center = target.mean(0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    sign = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    row_rotation = u @ sign @ vt
    return Rotation.from_matrix(row_rotation.T).as_rotvec()


def trajectory_rows(states, native):
    """Measure each reverse-solver displacement against the native rigid direction."""
    rows = []
    for step in range(len(states) - 1):
        current, following = states[step], states[step + 1]
        center_step = following.mean(0) - current.mean(0)
        center_target = native.mean(0) - current.mean(0)
        rotation_step = rigid_rotvec(current, following)
        rotation_target = rigid_rotvec(current, native)
        rows.append({
            'step': step + 1,
            'center_error_before_A': float(np.linalg.norm(center_target)),
            'center_error_after_A': float(np.linalg.norm(native.mean(0) - following.mean(0))),
            'center_step_A': float(np.linalg.norm(center_step)),
            'center_direction_cosine_to_native': cosine(center_step, center_target),
            'rotation_step_rad': float(np.linalg.norm(rotation_step)),
            'rotation_direction_cosine_to_native': cosine(rotation_step, rotation_target),
        })
    return rows


def summarize(rows):
    """Aggregate finite entries without treating undefined zero-motion cosines as zero."""
    result = {}
    for key in ('center_direction_cosine_to_native', 'rotation_direction_cosine_to_native'):
        values = [row[key] for row in rows if row[key] is not None]
        result['mean_' + key] = float(np.mean(values)) if values else None
        result['positive_' + key + '_fraction'] = (
            float(np.mean(np.asarray(values) > 0)) if values else None)
    result['center_error_change_A'] = (
        rows[-1]['center_error_after_A'] - rows[0]['center_error_before_A'])
    return result


def load_states(path, entity):
    values = np.load(path)
    initial = values[f'initial_{entity}']
    if initial.shape[0] == 1:
        initial = initial[0]
    trajectory = values[f'trajectory_{entity}']
    return np.concatenate((initial[None], trajectory), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    candidate_report = json.loads((args.candidate / 'report.json').read_text())
    for key in ('cases', 'seed', 'steps'):
        if baseline_report[key] != candidate_report[key]:
            raise ValueError(f'unpaired {key}: {baseline_report[key]!r} != {candidate_report[key]!r}')
    if baseline_report['status'] != 'complete' or candidate_report['status'] != 'complete':
        raise ValueError('both rollout reports must be complete')
    baseline_metrics = {row['case']: row for row in baseline_report['baseline']}
    candidate_metrics = {row['case']: row for row in candidate_report['baseline']}
    cases = []
    for case in baseline_report['cases']:
        base_path = args.baseline / case / 'coordinates.npz'
        candidate_path = args.candidate / case / 'coordinates.npz'
        cache = torch.load(args.feature_cache / f'{case}.pt', map_location='cpu', weights_only=False)
        native = {
            'ligand': cache['ligand_data']['coords'].numpy(),
            'poi': cache['poi_data']['coords'].numpy(),
        }
        case_row = {
            'case': case,
            'dockq': {
                'baseline': baseline_metrics[case]['dockq'],
                'candidate': candidate_metrics[case]['dockq'],
                'delta': candidate_metrics[case]['dockq'] - baseline_metrics[case]['dockq'],
            },
            'entities': {},
        }
        for entity in ('ligand', 'poi'):
            baseline = load_states(base_path, entity)
            candidate = load_states(candidate_path, entity)
            initial_error = float(np.max(np.abs(baseline[0] - candidate[0])))
            if initial_error != 0.0:
                raise ValueError(f'{case}/{entity}: paired initial state differs by {initial_error}')
            base_rows = trajectory_rows(baseline, native[entity])
            candidate_rows = trajectory_rows(candidate, native[entity])
            divergence = np.sqrt(np.mean(np.sum((candidate - baseline) ** 2, axis=-1), axis=-1))
            case_row['entities'][entity] = {
                'initial_max_abs_difference_A': initial_error,
                'baseline': summarize(base_rows),
                'candidate': summarize(candidate_rows),
                'candidate_vs_baseline_rmsd_A': divergence.tolist(),
                'steps': {'baseline': base_rows, 'candidate': candidate_rows},
            }
        cases.append(case_row)
    deltas = [case['dockq']['delta'] for case in cases]
    output = {
        'scope': 'Paired deterministic rollouts; empirical velocity is finite-step state displacement, not a raw model-head value.',
        'baseline': str(args.baseline),
        'candidate': str(args.candidate),
        'baseline_checkpoint': baseline_report['checkpoint'],
        'baseline_checkpoint_sha256': baseline_report['checkpoint_sha256'],
        'baseline_weights': baseline_report['weights'],
        'candidate_checkpoint': candidate_report['checkpoint'],
        'candidate_checkpoint_sha256': candidate_report['checkpoint_sha256'],
        'candidate_weights': candidate_report['weights'],
        'reproducibility': candidate_report['reproducibility'],
        'seed': baseline_report['seed'],
        'steps': baseline_report['steps'],
        'cases': cases,
        'summary': {
            'case_count': len(cases),
            'mean_dockq_delta': float(np.mean(deltas)),
            'median_dockq_delta': float(np.median(deltas)),
            'improved_cases': int(np.sum(np.asarray(deltas) > 0)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output['summary']), flush=True)


if __name__ == '__main__':
    main()
