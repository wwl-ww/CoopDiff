"""Post-hoc labels for fixed selections; never imported by the ranking entry point."""

import math

import numpy as np

from tricoopdiff.ranking.ranking_inference import ordered_rows


def join_labels(prediction, label):
    if prediction['case'] != label['case'] or prediction['archive_sha256'] != label['archive_sha256']:
        raise ValueError('prediction/label identity mismatch')
    truth = {}
    for pair in label['labels']:
        for stage in ('initial', 'final'):
            value = pair[stage]
            if stage == 'final' and value is None:
                continue
            key = (pair['pool_index'], stage)
            if key in truth or value is None:
                raise ValueError('duplicate or missing label')
            q = value['dockq']
            if not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError('invalid DockQ')
            truth[key] = q
    keys = [(r['pool_index'], r['stage']) for r in prediction['rows']]
    if len(set(keys)) != len(keys) or set(keys) != set(truth):
        raise ValueError('predictions must cover exactly the labeled states')
    rows = []
    for row in prediction['rows']:
        if row['source'] not in ('bitseed', 'sdpi'):
            raise ValueError('unknown actual source')
        if any(k in row for k in ('q_hat', 'p23', 'p49')):
            values = [row[k] for k in ('q_hat', 'p23', 'p49')]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or row['p49'] > row['p23']:
                raise ValueError('invalid quality probabilities')
        rows.append({**row, 'dockq': truth[(row['pool_index'], row['stage'])]})
    ordered_rows(rows)  # Validate score finiteness before any truth-based summary.
    return rows


def random_hit_probability(n, hits, k):
    """Exact success probability of uniform sampling without replacement."""
    k = min(n, k)
    if not n:
        return 0.
    return 1. - (math.comb(n - hits, k) / math.comb(n, k) if n - hits >= k else 0.)


def label_support(cases):
    """Count quality classes and independent contributing cases, not just states."""
    if not cases:
        raise ValueError('empty label support audit')
    result = {}
    for source in ('all', 'bitseed', 'sdpi'):
        for stage in ('initial', 'final', 'union'):
            selected = {case: [r for r in rows if (source == 'all' or r['source'] == source)
                              and (stage == 'union' or r['stage'] == stage)] for case, rows in cases.items()}
            values = [r['dockq'] for rows in selected.values() for r in rows]
            if any(not math.isfinite(q) or not 0 <= q <= 1 for q in values):
                raise ValueError('invalid label support input')
            bins = [sum(q < .23 for q in values), sum(.23 <= q < .49 for q in values), sum(q >= .49 for q in values)]
            result[f'{source}/{stage}'] = {
                'cases': len(cases), 'cases_with_states': sum(bool(rows) for rows in selected.values()),
                'states': len(values), 'class_counts_lt23_23to49_ge49': bins,
                'classes_with_no_labels': [i for i, n in enumerate(bins) if n == 0],
                'cases_with_ge23': sum(any(r['dockq'] >= .23 for r in rows) for rows in selected.values()),
                'cases_with_ge49': sum(any(r['dockq'] >= .49 for r in rows) for rows in selected.values()),
                'mean_state_dockq': sum(values) / len(values) if values else None,
                'max_dockq': max(values) if values else None,
            }
    return result


def case_metrics(rows):
    if not rows:
        return {'count': 0, 'top1': 0., 'top5_best': 0., 'oracle': 0., 'regret': 0.,
                'retention': None, 'random_top1_mean': 0., 'thresholds': {
                    str(t): {'top1_hit': 0, 'top5_hit': 0, 'random_top1_hit': 0., 'random_top5_hit': 0.}
                    for t in (.23, .49)}, 'brier': None, 'selected': None}
    ordered = ordered_rows(rows)
    top1, best5, oracle = ordered[0]['dockq'], max(r['dockq'] for r in ordered[:5]), max(r['dockq'] for r in rows)
    thresholds = {}
    for threshold in (.23, .49):
        hits = sum(r['dockq'] >= threshold for r in rows)
        thresholds[str(threshold)] = {'top1_hit': int(top1 >= threshold), 'top5_hit': int(best5 >= threshold),
                                     'random_top1_hit': hits / len(rows),
                                     'random_top5_hit': random_hit_probability(len(rows), hits, 5)}
    brier = ({str(t): sum((r[key] - (r['dockq'] >= t)) ** 2 for r in rows) / len(rows)
              for t, key in ((.23, 'p23'), (.49, 'p49'))} if all('p23' in r for r in rows) else None)
    return {'count': len(rows), 'top1': top1, 'top5_best': best5, 'oracle': oracle, 'regret': oracle - top1,
            'retention': top1 / oracle if oracle > 0 else None,
            'random_top1_mean': sum(r['dockq'] for r in rows) / len(rows),
            'thresholds': thresholds, 'brier': brier,
            'selected': {k: ordered[0][k] for k in ('pool_index', 'stage', 'source')}}


def summarize_cases(cases):
    """Case-macro statistics with absent source/final states retained in denominators."""
    if not cases:
        raise ValueError('empty audit')
    summary, details = {}, {}
    for source in ('all', 'bitseed', 'sdpi'):
        for stage in ('initial', 'final', 'union'):
            name = f'{source}/{stage}'
            per_case = {case: case_metrics([r for r in rows if
                        (source == 'all' or r['source'] == source) and (stage == 'union' or r['stage'] == stage)])
                        for case, rows in cases.items()}
            values = list(per_case.values())
            n = len(values)
            retained = [r['retention'] for r in values if r['retention'] is not None]
            calibrated = [r['brier'] for r in values if r['brier'] is not None]
            summary[name] = {
                'cases': n, 'missing': sum(r['count'] == 0 for r in values),
                **{f'mean_{key}': sum(r[key] for r in values) / n for key in
                   ('top1', 'top5_best', 'oracle', 'regret', 'random_top1_mean')},
                'mean_case_retention': sum(retained) / len(retained) if retained else None,
                'retention_defined_cases': len(retained), 'zero_oracle_cases': sum(r['oracle'] == 0 for r in values),
                'thresholds': {str(t): {key: sum(r['thresholds'][str(t)][key] for r in values)
                                       for key in ('top1_hit', 'top5_hit', 'random_top1_hit', 'random_top5_hit')}
                               for t in (.23, .49)},
                'brier_case_mean': {str(t): sum(r[str(t)] for r in calibrated) / len(calibrated)
                                    for t in (.23, .49)} if calibrated else None,
                'brier_defined_cases': len(calibrated),
            }
            details[name] = per_case
    return {'summary': summary, 'cases': details}


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
