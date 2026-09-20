"""Nested group-LOOCV audit for a linear DeepTernary candidate selector."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tricoopdiff.deepternary_ranking import FEATURE_VERSION


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def ridge_predict(train_x, train_y, test_x, alpha):
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    query = np.column_stack((np.ones(len(test_x)), (test_x - mean) / scale))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return query @ weights


def select_cases(x, y, pdb_groups, cases, train_groups, test_group, alpha):
    """Select one candidate per case while holding out a whole PDB group."""
    train = np.isin(pdb_groups, train_groups)
    test_indices = np.flatnonzero(pdb_groups == test_group)
    prediction = ridge_predict(x[train], y[train], x[test_indices], alpha)
    predicted = dict(zip(test_indices.tolist(), prediction.tolist()))
    chosen = []
    for case in sorted(set(cases[test_indices])):
        indices = [index for index in test_indices if cases[index] == case]
        chosen.append(max(indices, key=predicted.__getitem__))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('choose a new output path')
    payload = json.loads(args.dataset.read_text())
    if payload.get('feature_version') != FEATURE_VERSION:
        raise ValueError('selector feature version mismatch')
    rows = payload['rows']
    x = np.asarray([row['features'] for row in rows], dtype=np.float64)
    y = np.asarray([row['refined_dockq'] for row in rows], dtype=np.float64)
    cases = np.asarray([row['case'] for row in rows])
    pdb_groups = np.asarray([case.split('_', 1)[0] for case in cases])
    pdbs = sorted(set(pdb_groups))
    if len(pdbs) < 3:
        raise ValueError('nested PDB-group audit requires at least three PDB groups')
    results = []
    for outer in pdbs:
        available = [pdb for pdb in pdbs if pdb != outer]
        inner_scores = {}
        for alpha in ALPHAS:
            values = []
            for inner in available:
                train_groups = [pdb for pdb in available if pdb != inner]
                chosen = select_cases(x, y, pdb_groups, cases, train_groups, inner, alpha)
                values.extend(float(y[index]) for index in chosen)
            inner_scores[str(alpha)] = float(np.mean(values))
        best_alpha = min(ALPHAS, key=lambda value: (-inner_scores[str(value)], value))
        chosen_indices = select_cases(x, y, pdb_groups, cases, available, outer, best_alpha)
        for chosen in chosen_indices:
            case = cases[chosen]
            case_indices = np.flatnonzero(cases == case)
            official = case_indices[int(np.argmin(x[case_indices, 0]))]
            oracle = case_indices[int(np.argmax(y[case_indices]))]
            results.append({
                'pdb_group': outer, 'case': case, 'selected_seed': rows[chosen]['seed'],
                'selected_dockq': float(y[chosen]), 'alpha': best_alpha,
                'inner_mean_by_alpha': inner_scores,
                'official_seed': rows[official]['seed'], 'official_dockq': float(y[official]),
                'oracle_seed': rows[oracle]['seed'], 'oracle_dockq': float(y[oracle]),
            })
    report = {
        'protocol': 'deepternary_selector_nested_pdb_group_loocv_v2',
        'scope': 'development feasibility audit; no deployable selector weight',
        'dataset': str(args.dataset.resolve()), 'outer_pdb_groups': len(pdbs),
        'outer_cases': len(results),
        'summary': {
            'nested_selected_mean': float(np.mean([row['selected_dockq'] for row in results])),
            'official_surrogate_mean': float(np.mean([row['official_dockq'] for row in results])),
            'oracle_mean': float(np.mean([row['oracle_dockq'] for row in results])),
            'oracle_selected': sum(row['selected_seed'] == row['oracle_seed'] for row in results),
        },
        'rows': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
