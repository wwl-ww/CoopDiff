"""Summarize a sharded benchmark without dropping missing cases or seeds."""

import argparse
import json
from pathlib import Path

from evaluation_common import sha256_file
from make_deepternary_index import read_case_list
from summarize_evaluation_reports import load_row, summarize_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-root', required=True)
    parser.add_argument('--cases-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n-walkers', type=int, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cases = read_case_list(args.cases_file)
    if len(cases) != len(set(cases)) or args.n_walkers < 1:
        raise ValueError('unique cases and positive walker count required')
    expected_hash = sha256_file(args.checkpoint)
    expected_seeds = set(range(args.seed, args.seed + args.n_walkers))
    rows, failures, seed_rows = [], [], []
    exact_total = 0
    for case in cases:
        paths = list(Path(args.reports_root).rglob(f'{case}.json'))
        if len(paths) != 1:
            failures.append({'case': case, 'error': f'expected one report, found {len(paths)}'})
            continue
        report = json.loads(paths[0].read_text())
        config, ensemble = report['run_config'], report['ensemble']
        expected = dict(checkpoint_sha256=expected_hash, n_walkers=args.n_walkers,
                        steps=args.steps, seed=args.seed, init_mode='sdpi', freeze_poi=False)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f'configuration mismatch for {case}')
        if ensemble['selection_metric'] != 'dockq':
            failures.append({'case': case, 'error': 'no exact DockQ selection'})
        walkers = report.get('walkers', [])
        seeds = [row['seed'] for row in walkers]
        invalid = ensemble.get('invalid_seeds', [])
        if len(seeds) != len(set(seeds)) or set(seeds) & set(invalid):
            raise ValueError(f'duplicate seed for {case}')
        if set(seeds) | set(invalid) != expected_seeds:
            raise ValueError(f'seed accounting mismatch for {case}')
        exact = [row for row in walkers if 'dockq' in row]
        exact_total += len(exact)
        if len(exact) != args.n_walkers:
            failures.append({'case': case, 'error': 'missing valid exact-scored seeds',
                             'exact': len(exact), 'requested': args.n_walkers,
                             'invalid_seeds': invalid})
        seed_rows.extend({'case': case, **row} for row in walkers)
        rows.append(load_row(paths[0]))
    summary = summarize_rows(rows)
    summary.update({
        'expected_cases': len(cases),
        'expected_walkers': len(cases) * args.n_walkers,
        'exact_scored_walkers': exact_total,
        'complete_all_cases_and_seeds': not failures,
        'incomplete_cases': failures,
        'selection': 'oracle-best, not model-ranked top-1' if args.n_walkers > 1 else 'single-seed',
        'checkpoint_sha256': expected_hash,
        'case_success_rates_including_missing': {
            str(threshold): {
                'successful': sum(row.get('dockq_best') is not None and row['dockq_best'] >= threshold
                                  for row in rows),
                'denominator': len(cases),
                'rate': sum(row.get('dockq_best') is not None and row['dockq_best'] >= threshold
                            for row in rows) / len(cases),
            } for threshold in (.23, .49, .80)
        },
    })
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'summary': summary, 'runs': rows, 'walkers': seed_rows},
                                      indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
