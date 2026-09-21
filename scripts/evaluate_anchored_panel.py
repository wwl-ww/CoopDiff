"""Foreground fixed-budget oracle panel; no ranking or benchmark weight selection."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def summarize_panel(rows, expected_count):
    completed = [row for row in rows if row['status'] == 'completed']
    complete = len(rows) == expected_count and len(completed) == expected_count
    return {'expected_cases': expected_count, 'completed_cases': len(completed),
            'complete': complete,
            'mean_case_oracle_dockq': (sum(row['oracle_dockq'] for row in completed) / expected_count
                                      if complete else None),
            'model_top1_computed': False, 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'expert-model-dir', 'case-root', 'manifest-dir', 'cache-dir',
                 'deepternary-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--n-walkers', type=int, default=32)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--stratified-mixture', action='store_true',
                        help='Explicit stratified mixture ensemble; distinct from the historical iid component protocol')
    parser.add_argument('--allow-chemical-repeats', action='store_true',
                        help='Allow audited chemical cycles and reuse identical trajectories')
    parser.add_argument('--conserved-core-chemistry-report', type=Path,
                        help='Explicit experimental extra binary-core conditions; separately recorded protocol')
    args = parser.parse_args()
    if args.allow_chemical_repeats and not args.stratified_mixture:
        parser.error('chemical repeats require stratified mixture')
    if args.out.exists():
        raise FileExistsError('panel outputs must use a new directory')
    if (len(set(args.cases)) != len(args.cases)
            or any(case != Path(case).name or case in ('.', '..') for case in args.cases)
            or min(args.n_walkers, args.steps) < 1 or args.seed < 0):
        parser.error('require distinct safe case names and positive sampling budget')
    selection = json.loads(args.selection.read_text())
    if (selection.get('scope') != 'training'
            or selection.get('criterion') != 'ema_fixed_case_prior_times_mean_loss_v1'):
        raise ValueError('panel requires preselected EMA development-loss weights')
    checkpoint = Path(selection['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('selected checkpoint hash mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / 'scripts/sample_anchored_entry.py',
                    ROOT / 'scripts/score_anchored_prediction.py', *sorted((ROOT / 'tricoopdiff').rglob('*.py'))]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    with (args.out / 'config.json').open('x') as handle:
        json.dump({**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   'selection': selection, 'scope': 'explicit development panel; not full benchmark unless declared',
                   'selection_used_native_oracle': False, 'source_sha256': source_hashes}, handle, indent=2)
    rows = []
    for case in args.cases:
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
               for name, expected in source_hashes.items()):
            raise RuntimeError('prediction/scoring source changed during the fixed panel')
        print(json.dumps({'case': case, 'status': 'started'}), flush=True)
        output = args.out / case
        output.mkdir()
        commands = [
            [sys.executable, str(ROOT / 'scripts/sample_anchored_entry.py'),
             '--checkpoint', str(checkpoint), '--expert-model-dir', str(args.expert_model_dir),
             '--case-dir', str(args.case_root / case), '--condition-manifest', str(args.manifest_dir / f'{case}.json'),
             '--cache-dir', str(args.cache_dir), '--seed', str(args.seed), '--steps', str(args.steps),
             '--n-walkers', str(args.n_walkers), '--out', str(output / 'predictions.json')],
            [sys.executable, str(ROOT / 'scripts/score_anchored_prediction.py'),
             '--prediction', str(output / 'predictions.json'), '--case-dir', str(args.case_root / case),
             '--deepternary-root', str(args.deepternary_root), '--out', str(output / 'scores.json')]]
        if args.conserved_core_chemistry_report:
            commands[0].extend(['--conserved-core-chemistry-report', str(args.conserved_core_chemistry_report)])
        if args.stratified_mixture:
            commands[0].append('--stratified-mixture')
        if args.allow_chemical_repeats:
            commands[0].append('--allow-chemical-repeats')
        try:
            for name, command in zip(('sampling', 'scoring'), commands):
                with (output / f'{name}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            score = json.loads((output / 'scores.json').read_text())
            oracle = score['oracle_dockq'] if args.n_walkers > 1 else score['metric']['dockq']
            row = {'case': case, 'status': 'completed', 'oracle_dockq': oracle,
                   'n_walkers': args.n_walkers, 'score_path': str((output / 'scores.json').resolve())}
        except subprocess.CalledProcessError as error:
            row = {'case': case, 'status': 'failed', 'stage': name, 'returncode': error.returncode,
                   'log': str((output / f'{name}.log').resolve())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = summarize_panel(rows, len(args.cases))
    with (args.out / 'summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    if not summary['complete']:
        raise SystemExit('panel incomplete; no full-panel mean reported')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
