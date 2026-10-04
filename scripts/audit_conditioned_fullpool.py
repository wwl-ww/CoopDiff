"""Audit paired full-pool PROTAC refinements across condition settings."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time


RUNS = ('pocket', 'absent', 'control', 'parent')
STAGES = {
    'initial': 'initial_dockq',
    'refined': 'refined_dockq',
    'safe_union': 'safe_dockq',
}


def aggregate(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('metrics must be finite and nonempty')
    return {
        'count': len(values), 'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'hits_0.23': sum(value >= .23 for value in values),
        'hits_0.49': sum(value >= .49 for value in values),
    }


def paired(left, right):
    if len(left) != len(right):
        raise ValueError('paired vectors differ in length')
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    result = aggregate(delta)
    result.update(
        wins=sum(value > 1e-8 for value in delta),
        ties=sum(abs(value) <= 1e-8 for value in delta),
        losses=sum(value < -1e-8 for value in delta),
        up_0_23=sum(b < .23 <= a for a, b in zip(left, right)),
        down_0_23=sum(a < .23 <= b for a, b in zip(left, right)),
        up_0_49=sum(b < .49 <= a for a, b in zip(left, right)),
        down_0_49=sum(a < .49 <= b for a, b in zip(left, right)),
    )
    return result


def read_run(path, expected_candidates):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary.get('failures') or summary['requested_cases'] != len(summary['completed_cases']):
        raise ValueError(f'{path}: incomplete run')
    reports = {}
    for case in summary['completed_cases']:
        report = json.loads((path / f'{case}.json').read_text())
        rows = report.get('refinements', [])
        if (report.get('status') != 'complete' or len(rows) != expected_candidates
                or report.get('union_evaluated_count') != expected_candidates):
            raise ValueError(f'{path}/{case}: not a complete full-pool report')
        if any(not row.get('valid') for row in rows):
            raise ValueError(f'{path}/{case}: invalid refinement in full pool')
        reports[case] = report
    return summary, reports


def identity(row):
    return (
        row['candidate_id'], row['initial_state_sha256'], row['diffusion_seed'],
        float(row['initial_dockq']), row['source'], row['pool_index'],
    )


def audit(paths, expected_candidates=256):
    if set(paths) != set(RUNS):
        raise ValueError(f'exact run names required: {RUNS}')
    loaded = {name: read_run(paths[name], expected_candidates) for name in RUNS}
    cases = loaded['pocket'][0]['completed_cases']
    if any(set(value[0]['completed_cases']) != set(cases) for value in loaded.values()):
        raise ValueError('runs have different case sets')
    common_arms = set.intersection(*(
        set(value[0]['leaderboards']) for value in loaded.values()))
    candidate_values = {name: {stage: [] for stage in STAGES} for name in RUNS}
    candidate_sources = []
    case_values = {
        name: {arm: {stage: [] for stage in STAGES}
               for arm in sorted(common_arms | {'full_pool'})}
        for name in RUNS
    }
    rows = []
    for case in cases:
        reports = {name: loaded[name][1][case] for name in RUNS}
        if len({report['proposal_pool_sha256'] for report in reports.values()}) != 1:
            raise ValueError(f'{case}: proposal pool hashes differ')
        ordered = {
            name: sorted(report['refinements'], key=lambda row: row['pool_index'])
            for name, report in reports.items()
        }
        reference = [identity(row) for row in ordered['pocket']]
        if any([identity(row) for row in ordered[name]] != reference for name in RUNS[1:]):
            raise ValueError(f'{case}: candidate identity/initial state/seed differs')
        candidate_sources.extend(row['source'] for row in ordered['pocket'])
        case_row = {'case': case, 'proposal_pool_sha256': reports['pocket'][
            'proposal_pool_sha256'], 'runs': {}}
        for name in RUNS:
            case_row['runs'][name] = {}
            for stage, field in STAGES.items():
                values = [float(row[field]) for row in ordered[name]]
                candidate_values[name][stage].extend(values)
                case_values[name]['full_pool'][stage].append(max(values))
                case_row['runs'][name][stage] = max(values)
            for arm in common_arms:
                for stage in STAGES:
                    case_values[name][arm][stage].append(
                        float(reports[name]['arms'][arm][stage]['best']))
        rows.append(case_row)
    comparisons = {}
    for label, left, right in (
        ('pocket_minus_absent', 'pocket', 'absent'),
        ('absent_minus_control', 'absent', 'control'),
        ('pocket_minus_control', 'pocket', 'control'),
        ('absent_minus_parent', 'absent', 'parent'),
    ):
        comparisons[label] = {
            'candidate': {
                stage: paired(candidate_values[left][stage], candidate_values[right][stage])
                for stage in STAGES
            },
            'candidate_by_source': {},
            'case_best': {},
        }
        for source in ('bitseed', 'sdpi'):
            indices = [i for i, value in enumerate(candidate_sources) if value == source]
            comparisons[label]['candidate_by_source'][source] = {
                stage: paired(
                    [candidate_values[left][stage][i] for i in indices],
                    [candidate_values[right][stage][i] for i in indices],
                ) for stage in STAGES
            }
        for arm in sorted(common_arms | {'full_pool'}):
            comparisons[label]['case_best'][arm] = {
                stage: paired(case_values[left][arm][stage], case_values[right][arm][stage])
                for stage in STAGES
            }
    return {
        'schema': 'conditioned_protac_fullpool_audit_v2',
        'scope': ('paired oracle diagnostics over every refined candidate; '
                  'not native-free post-ranker top1'),
        'paths': {name: str(Path(path).resolve()) for name, path in paths.items()},
        'cases': cases, 'expected_candidates_per_case': expected_candidates,
        'identity_checks': {
            'proposal_pool_sha256_equal': True,
            'candidate_initial_state_seed_score_equal': True,
        },
        'candidate_leaderboards': {
            name: {stage: aggregate(values) for stage, values in stages.items()}
            for name, stages in candidate_values.items()
        },
        'case_best_leaderboards': {
            name: {arm: {stage: aggregate(values) for stage, values in stages.items()}
                   for arm, stages in arms.items()}
            for name, arms in case_values.items()
        },
        'comparisons': comparisons, 'rows': rows,
    }


def render(summary):
    lines = [
        '# Conditioned PROTAC Full-Pool Audit', '', summary['scope'] + '.', '',
        '## Candidate-level paired effects', '',
        '| Comparison | Source | Stage | Mean ΔDockQ | W / T / L | ↑ / ↓ 0.23 |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for label, comparison in summary['comparisons'].items():
        for source, stages in [('all', comparison['candidate'])] + list(
                comparison['candidate_by_source'].items()):
            for stage, row in stages.items():
                lines.append(
                    f'| {label} | {source} | {stage} | {row["mean"]:+.5f} | '
                    f'{row["wins"]} / {row["ties"]} / {row["losses"]} | '
                    f'{row["up_0_23"]} / {row["down_0_23"]} |')
    lines.extend(['', '## Case-best oracle effects', '',
                  '| Comparison | Selector | Stage | Mean ΔDockQ | W / T / L |',
                  '| --- | --- | --- | ---: | ---: |'])
    for label, comparison in summary['comparisons'].items():
        for arm, stages in comparison['case_best'].items():
            for stage, row in stages.items():
                lines.append(f'| {label} | {arm} | {stage} | {row["mean"]:+.4f} | '
                             f'{row["wins"]} / {row["ties"]} / {row["losses"]} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in RUNS:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-candidates', type=int, default=256)
    parser.add_argument('--wait-hours', type=float, default=0.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if (args.out.exists() or args.expected_candidates < 1 or args.wait_hours < 0
            or args.poll_seconds <= 0):
        parser.error('output must be new and numeric settings must be positive')
    paths = {name: getattr(args, name) for name in RUNS}
    deadline = time.time() + args.wait_hours * 3600
    while not all((path / 'summary.json').is_file() for path in paths.values()):
        if time.time() >= deadline:
            raise RuntimeError('timed out waiting for complete full-pool runs')
        time.sleep(args.poll_seconds)
    summary = audit(paths, args.expected_candidates)
    args.out.mkdir(parents=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'REPORT.md').write_text(render(summary))


if __name__ == '__main__':
    main()
