"""Audit paired pocket/absent evaluations across checkpoints and seeds."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time


STAGES = ('initial', 'refined', 'safe_union')


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


def read_run(path, arm):
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    cases = summary.get('completed_cases', [])
    if summary.get('failures') or summary.get('requested_cases') != len(cases):
        raise ValueError(f'{path}: run is incomplete')
    reports = {}
    for case in cases:
        report = json.loads((path / f'{case}.json').read_text())
        if report.get('status') != 'complete' or arm not in report.get('arms', {}):
            raise ValueError(f'{path}/{case}: missing complete {arm} result')
        reports[case] = report
    return summary, reports


def refinement_identity(report):
    return {
        repr(row['candidate_id']): (
            row['initial_state_sha256'], row['diffusion_seed'],
            float(row['initial_dockq']))
        for row in report['refinements']
    }


def audit_pair(pocket_path, absent_path, arm):
    pocket_summary, pocket = read_run(pocket_path, arm)
    absent_summary, absent = read_run(absent_path, arm)
    cases = pocket_summary['completed_cases']
    if set(cases) != set(absent_summary['completed_cases']):
        raise ValueError('pocket and absent runs have different cases')
    values = {
        condition: {stage: [] for stage in STAGES}
        for condition in ('pocket', 'absent')
    }
    rows = []
    for case in cases:
        left, right = pocket[case], absent[case]
        if left['proposal_pool_sha256'] != right['proposal_pool_sha256']:
            raise ValueError(f'{case}: proposal pool differs within pair')
        if (left['arms'][arm]['selected_candidate_ids']
                != right['arms'][arm]['selected_candidate_ids']):
            raise ValueError(f'{case}: selected candidates differ within pair')
        if refinement_identity(left) != refinement_identity(right):
            raise ValueError(f'{case}: initial state/seed identity differs within pair')
        scores = {}
        for condition, report in (('pocket', left), ('absent', right)):
            scores[condition] = {}
            for stage in STAGES:
                value = float(report['arms'][arm][stage]['best'])
                values[condition][stage].append(value)
                scores[condition][stage] = value
        rows.append({'case': case, 'scores': scores})
    return {
        'paths': {'pocket': str(Path(pocket_path).resolve()),
                  'absent': str(Path(absent_path).resolve())},
        'cases': cases,
        'identity_checks': {
            'proposal_pool_equal': True, 'selected_candidates_equal': True,
            'initial_state_seed_score_equal': True,
        },
        'leaderboards': {
            condition: {stage: aggregate(stage_values)
                        for stage, stage_values in condition_values.items()}
            for condition, condition_values in values.items()
        },
        'pocket_minus_absent': {
            stage: paired(values['pocket'][stage], values['absent'][stage])
            for stage in STAGES
        },
        'rows': rows,
    }


def audit(pairs, arm='hybrid_56_8'):
    if not pairs or len({name for name, _, _ in pairs}) != len(pairs):
        raise ValueError('pair labels must be nonempty and unique')
    return {
        'schema': 'conditioned_checkpoint_seed_sweep_v1',
        'scope': ('case-best oracle within a fixed native-free 64-candidate arm; '
                  'each pocket/absent contrast is strictly paired'),
        'arm': arm,
        'pairs': {
            name: audit_pair(pocket, absent, arm)
            for name, pocket, absent in pairs
        },
    }


def render(summary):
    lines = [
        '# Conditioned Checkpoint and Seed Sweep', '', summary['scope'] + '.', '',
        '| Pair | Stage | Pocket | Absent | Δ | W / T / L | Pocket ≥0.23 | Absent ≥0.23 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, result in summary['pairs'].items():
        for stage in STAGES:
            pocket = result['leaderboards']['pocket'][stage]
            absent = result['leaderboards']['absent'][stage]
            delta = result['pocket_minus_absent'][stage]
            lines.append(
                f'| {name} | {stage} | {pocket["mean"]:.4f} | '
                f'{absent["mean"]:.4f} | {delta["mean"]:+.4f} | '
                f'{delta["wins"]} / {delta["ties"]} / {delta["losses"]} | '
                f'{pocket["hits_0.23"]}/{pocket["count"]} | '
                f'{absent["hits_0.23"]}/{absent["count"]} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'POCKET_RUN', 'ABSENT_RUN'))
    parser.add_argument('--arm', default='hybrid_56_8')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=0.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    pairs = [(name, Path(pocket), Path(absent))
             for name, pocket, absent in args.pair]
    if (args.out.exists() or args.wait_hours < 0 or args.poll_seconds <= 0
            or len({name for name, _, _ in pairs}) != len(pairs)):
        parser.error('output must be new and pair/wait settings must be valid')
    deadline = time.time() + args.wait_hours * 3600
    paths = [path for _, pocket, absent in pairs for path in (pocket, absent)]
    while not all((path / 'summary.json').is_file() for path in paths):
        if time.time() >= deadline:
            raise RuntimeError('timed out waiting for checkpoint/seed sweep')
        time.sleep(args.poll_seconds)
    summary = audit(pairs, args.arm)
    args.out.mkdir(parents=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'REPORT.md').write_text(render(summary))


if __name__ == '__main__':
    main()
