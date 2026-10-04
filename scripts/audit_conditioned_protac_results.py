"""Paired audit for pocket, absent-input, and absent-training PROTAC runs."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time


RUN_NAMES = ('default_pocket', 'default_absent', 'control_absent')
STAGES = ('initial', 'refined', 'safe_union')


def read_complete_run(path, arm):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary.get('failures') or summary['requested_cases'] != len(summary['completed_cases']):
        raise ValueError(f'{path}: run is incomplete')
    rows = {}
    for case in summary['completed_cases']:
        report = json.loads((path / f'{case}.json').read_text())
        if report.get('status') != 'complete' or arm not in report['arms']:
            raise ValueError(f'{path}/{case}: missing complete arm')
        rows[case] = report
    return summary, rows


def aggregate(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('paired metrics must be finite and nonempty')
    return {
        'mean': statistics.fmean(values), 'median': statistics.median(values),
        'hits_0.23': sum(value >= .23 for value in values),
        'hits_0.49': sum(value >= .49 for value in values), 'cases': len(values),
    }


def paired_delta(left, right):
    delta = [a - b for a, b in zip(left, right)]
    return {
        **aggregate(delta), 'wins': sum(value > 1e-8 for value in delta),
        'ties': sum(abs(value) <= 1e-8 for value in delta),
        'losses': sum(value < -1e-8 for value in delta),
    }


def audit(run_paths, arm='hybrid_56_8'):
    if set(run_paths) != set(RUN_NAMES):
        raise ValueError(f'exact run names required: {RUN_NAMES}')
    loaded = {name: read_complete_run(run_paths[name], arm) for name in RUN_NAMES}
    case_orders = [value[0]['completed_cases'] for value in loaded.values()]
    if any(set(order) != set(case_orders[0]) for order in case_orders[1:]):
        raise ValueError('paired runs do not contain the same cases')
    cases = case_orders[0]
    per_case = []
    for case in cases:
        reports = {name: loaded[name][1][case] for name in RUN_NAMES}
        if len({report['proposal_pool_sha256'] for report in reports.values()}) != 1:
            raise ValueError(f'{case}: proposal pools differ')
        selected = [report['arms'][arm]['selected_candidate_ids'] for report in reports.values()]
        if any(value != selected[0] for value in selected[1:]):
            raise ValueError(f'{case}: selected candidate order differs')
        identities = []
        for report in reports.values():
            identities.append({
                repr(row['candidate_id']): (
                    row['initial_state_sha256'], row['diffusion_seed'], row['initial_dockq'])
                for row in report['refinements']
            })
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(f'{case}: initial state/seed/score identity differs')
        scores = {
            name: {stage: float(report['arms'][arm][stage]['best'])
                   for stage in STAGES}
            for name, report in reports.items()
        }
        per_case.append({'case': case, 'proposal_pool_sha256': reports[RUN_NAMES[0]][
            'proposal_pool_sha256'], 'scores': scores})
    leaderboards = {}
    for name in RUN_NAMES:
        leaderboards[name] = {
            stage: aggregate([row['scores'][name][stage] for row in per_case])
            for stage in STAGES
        }
    comparisons = {}
    for label, left, right in (
        ('pocket_input_effect', 'default_pocket', 'default_absent'),
        ('condition_training_effect_when_absent', 'default_absent', 'control_absent'),
        ('complete_conditioned_system_effect', 'default_pocket', 'control_absent'),
    ):
        comparisons[label] = {
            stage: paired_delta(
                [row['scores'][left][stage] for row in per_case],
                [row['scores'][right][stage] for row in per_case])
            for stage in STAGES
        }
    return {
        'schema': 'conditioned_protac_paired_audit_v1', 'arm': arm,
        'scope': ('case-best oracle within a fixed native-free 64-candidate arm; '
                  'not post-ranker top1'),
        'run_paths': {name: str(Path(path).resolve()) for name, path in run_paths.items()},
        'cases': cases, 'identity_checks': {
            'proposal_pool_sha256_equal': True,
            'selected_candidate_ids_equal': True,
            'initial_state_seed_and_dockq_equal': True,
        },
        'leaderboards': leaderboards, 'paired_comparisons': comparisons,
        'rows': per_case,
    }


def render(summary):
    lines = [
        '# Conditioned PROTAC Paired Audit', '',
        f'Arm: `{summary["arm"]}`. {summary["scope"]}.', '',
        '| Run | Stage | Mean best DockQ | Median | ≥0.23 | ≥0.49 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for name in RUN_NAMES:
        for stage in STAGES:
            row = summary['leaderboards'][name][stage]
            lines.append(f'| {name} | {stage} | {row["mean"]:.4f} | '
                         f'{row["median"]:.4f} | {row["hits_0.23"]}/{row["cases"]} | '
                         f'{row["hits_0.49"]}/{row["cases"]} |')
    lines.extend(['', '## Paired deltas', '',
                  'Positive deltas favor the first system named by each comparison.', '',
                  '| Comparison | Stage | Mean ΔDockQ | Wins / ties / losses |',
                  '| --- | --- | ---: | ---: |'])
    for label, stages in summary['paired_comparisons'].items():
        for stage, row in stages.items():
            lines.append(f'| {label} | {stage} | {row["mean"]:+.4f} | '
                         f'{row["wins"]} / {row["ties"]} / {row["losses"]} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in RUN_NAMES:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', default='hybrid_56_8')
    parser.add_argument('--wait-hours', type=float, default=0.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if args.out.exists() or args.wait_hours < 0 or args.poll_seconds <= 0:
        parser.error('output must be new and wait settings valid')
    paths = {name: getattr(args, name) for name in RUN_NAMES}
    deadline = time.time() + args.wait_hours * 3600
    while not all((path / 'summary.json').is_file() for path in paths.values()):
        if time.time() >= deadline:
            raise RuntimeError('timed out waiting for all conditioned PROTAC runs')
        time.sleep(args.poll_seconds)
    summary = audit(paths, args.arm)
    args.out.mkdir(parents=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'REPORT.md').write_text(render(summary))


if __name__ == '__main__':
    main()
