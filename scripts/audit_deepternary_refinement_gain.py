"""Audit model-top1 conditioned refinement against unchanged poses."""

import argparse
import json
from pathlib import Path
import statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--zero-results', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-candidates', type=int, default=None)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('choose a new output path')
    dataset = json.loads(args.dataset.read_text())
    by_case = {}
    for row in dataset['rows']:
        by_case.setdefault(row['case'], []).append(row)
    rows = []
    for case, candidates in sorted(by_case.items()):
        if args.expected_candidates is not None and len(candidates) != args.expected_candidates:
            raise ValueError(
                f'{case}: expected {args.expected_candidates} candidates, found {len(candidates)}'
            )
        model_top1 = min(candidates, key=lambda row: (row['features'][0], row['seed']))
        zero_paths = [root / f'{case}.json' for root in args.zero_results]
        zero_paths = [path for path in zero_paths if path.is_file()]
        if len(zero_paths) != 1:
            raise ValueError(f'{case}: expected one unchanged-pose result, found {zero_paths}')
        zero = json.loads(zero_paths[0].read_text())
        if zero['case'] != case or zero['run_config']['condition_guidance'] != 'residual':
            raise ValueError(f'{case}: invalid unchanged-pose report')
        initial = float(zero['structure']['dockq_best'])
        refined = float(model_top1['refined_dockq'])
        rows.append({'case': case, 'model_top1_seed': model_top1['seed'],
                     'candidate_count': len(candidates),
                     'initial_dockq': initial, 'refined_dockq': refined,
                     'delta': refined - initial})
    deltas = [row['delta'] for row in rows]
    report = {
        'protocol': 'deepternary_model_top1_conditioned_refinement_gain_v2',
        'selection': 'minimum model-predicted p2 RMSD; no native score used',
        'scope': ('development audit under one bound-template reconstruction metric; '
                  'not comparable to a different candidate-count protocol'),
        'dataset': str(args.dataset.resolve()),
        'zero_result_roots': [str(path.resolve()) for path in args.zero_results],
        'summary': {
            'cases': len(rows),
            'candidate_counts': sorted({row['candidate_count'] for row in rows}),
            'initial_mean': statistics.mean(row['initial_dockq'] for row in rows),
            'refined_mean': statistics.mean(row['refined_dockq'] for row in rows),
            'mean_delta': statistics.mean(deltas),
            'positive': sum(value > 0 for value in deltas),
            'negative': sum(value < 0 for value in deltas),
        },
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
