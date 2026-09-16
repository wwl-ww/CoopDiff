"""Paired post-hoc comparisons on identical labeled cases, not pose-level replicates."""

import math

import numpy as np


def paired_case_comparison(left, right, stratum, *, repeats=2000, seed=7315):
    """Average seeds within each case, then bootstrap cases as paired units.

    The interval is descriptive for the selected internal models. It neither
    accounts for validation-based checkpoint selection nor certifies holdout gain.
    """
    if not left or len(left) != len(right) or repeats < 1:
        raise ValueError('matching nonempty seed panels and positive bootstrap repeats required')
    reference = left[0]
    cases = sorted(reference['files_sha256'])
    if not cases:
        raise ValueError('empty paired panel')
    for report in [*left, *right]:
        if (report['labels_manifest_sha256'] != reference['labels_manifest_sha256']
                or sorted(report['files_sha256']) != cases or sorted(report['cases'][stratum]) != cases):
            raise ValueError('paired comparison requires identical label manifests and cases')
        for case in cases:
            if report['files_sha256'][case]['label'] != reference['files_sha256'][case]['label']:
                raise ValueError('paired label identity mismatch')
            row, target = report['cases'][stratum][case], reference['cases'][stratum][case]
            if row['count'] != target['count'] or not math.isclose(row['oracle'], target['oracle'], abs_tol=1e-12):
                raise ValueError('paired candidate set mismatch')
            if not math.isfinite(row['top1']) or not 0 <= row['top1'] <= 1:
                raise ValueError('invalid paired top1')
    rows = []
    for case in cases:
        a = [r['cases'][stratum][case] for r in left]
        b = [r['cases'][stratum][case] for r in right]
        mean_a, mean_b = float(np.mean([r['top1'] for r in a])), float(np.mean([r['top1'] for r in b]))
        rows.append({'case': case, 'left_mean_top1': mean_a, 'right_mean_top1': mean_b,
                     'delta': mean_a - mean_b, 'oracle': a[0]['oracle'], 'states': a[0]['count'],
                     'threshold_hit_delta': {str(t): float(np.mean([r['top1'] >= t for r in a])
                                                              - np.mean([r['top1'] >= t for r in b]))
                                             for t in (.23, .49)}})
    deltas = np.array([r['delta'] for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cases), size=(repeats, len(cases)))
    interval = np.quantile(deltas[draws].mean(1), [.025, .975]).tolist()
    return {'cases': len(cases), 'seeds_per_method': len(left), 'stratum': stratum,
            'mean_delta_top1': float(deltas.mean()), 'descriptive_case_bootstrap_95pct': interval,
            'case_wins': int((deltas > 1e-12).sum()), 'case_losses': int((deltas < -1e-12).sum()),
            'case_ties': int((np.abs(deltas) <= 1e-12).sum()),
            'cases_with_candidates': sum(r['states'] > 0 for r in rows),
            'mean_seed_hit_count_delta': {str(t): sum(r['threshold_hit_delta'][str(t)] for r in rows)
                                           for t in (.23, .49)},
            'bootstrap': {'unit': 'case, after averaging paired seeds', 'repeats': repeats, 'seed': seed,
                          'scope': 'descriptive internal comparison, not selection-adjusted or holdout inference'},
            'rows': rows}
