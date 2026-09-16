"""CPU-only paired case audit after every predeclared overnight ranker finishes."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import sha256_file
from run_ranking_night_queue import SEEDS, SIZES
from tricoopdiff.ranking_comparison import paired_case_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--night-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=0.)
    args = parser.parse_args()
    if not 0 <= args.wait_hours <= 12 or args.out.exists():
        parser.error('use a new output directory and a wait budget in [0,12] hours')
    # Resolve caller paths before entering a frozen runtime snapshot whose ROOT
    # deliberately differs from the live checkout.
    run, out = args.night_run.resolve(), args.out.resolve()
    # Some vendored imports change cwd; project-relative CLI paths belong at ROOT.
    os.chdir(ROOT)
    deadline = time.monotonic() + args.wait_hours * 3600
    last_stage = None
    while True:
        state = json.loads((run / 'queue_status.json').read_text())
        stage = state['stage']
        if stage != last_stage:
            print(f'UPSTREAM: {stage}', flush=True)
            last_stage = stage
        if stage == 'complete':
            break
        if stage not in ('waiting_upstream', 'collecting_extension', 'scoring_extension',
                         'auditing_extension', 'training_curves', 'summarizing'):
            raise RuntimeError(f'upstream did not complete: {stage}')
        if time.monotonic() >= deadline:
            raise TimeoutError('no complete learning curve yet; do not audit a successful subset')
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    panels, signatures = {}, {}
    for mode in ('geometry', 'frozen_geometry'):
        for size in SIZES:
            reports = []
            for seed in SEEDS:
                folder = run / 'rankers' / f'{mode}_n{size}_s{seed}'
                training = json.loads((folder / 'report.json').read_text())
                if (training['seed'] != seed or training['train_cases'] != size or training['val_cases'] != 8
                        or training['features'] != ('geometry_only' if mode == 'geometry' else 'frozen_plus_geometry')):
                    raise ValueError('training report does not match the predeclared grid')
                path = folder / 'audit.json'
                reports.append(json.loads(path.read_text()))
                signatures[str(path)] = sha256_file(path)
                signatures[str(folder / 'report.json')] = sha256_file(folder / 'report.json')
            panels[f'{mode}_n{size}'] = reports
    pairs = [(f'{mode}_n{size}', f'{mode}_n32') for mode in ('geometry', 'frozen_geometry') for size in (80, 160)]
    pairs += [(f'frozen_geometry_n{size}', f'geometry_n{size}') for size in SIZES]
    comparisons = {}
    for a, b in pairs:
        comparisons[f'{a} minus {b}'] = {
            f'{source}/{stage}': paired_case_comparison(panels[a], panels[b], f'{source}/{stage}')
            for source in ('all', 'bitseed', 'sdpi') for stage in ('initial', 'final', 'union')}
    payload = {'scope': 'post-hoc internal paired comparisons; no model/seed/case selection',
               'warning': '8 cases, repeated validation use and multiple comparisons; intervals are descriptive only',
               'compute': '100 epochs at every data size; updates are not matched',
               'source_sha256': {str(p): sha256_file(p) for p in (
                   Path(__file__), ROOT / 'tricoopdiff/ranking_comparison.py')},
               'input_sha256': signatures, 'comparisons': comparisons}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'paired.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Paired internal ranking audit', '', payload['warning'], '',
             'Differences are left minus right. Seeds are averaged within each case; cases are resampled together.',
             'Case-bootstrap intervals below are not seed SD, selection-adjusted intervals, or holdout evidence.', '',
             '| Comparison | Stage | Δ top1 | Descriptive 95% interval | Case W/L/T | Δ ≥0.23 hits (seed mean) |',
             '| --- | --- | ---: | --- | --- | ---: |']
    for name, strata in comparisons.items():
        for stage in ('initial', 'final', 'union'):
            row = strata[f'all/{stage}']
            lo, hi = row['descriptive_case_bootstrap_95pct']
            lines.append(f'| {name} | {stage} | {row["mean_delta_top1"]:.4f} | [{lo:.4f}, {hi:.4f}] | '
                         f'{row["case_wins"]}/{row["case_losses"]}/{row["case_ties"]} | '
                         f'{row["mean_seed_hit_count_delta"]["0.23"]:.2f} |')
    (out / 'paired.md').write_text('\n'.join(lines) + '\n')
    print(f'COMPLETE: {out / "paired.md"}', flush=True)


if __name__ == '__main__':
    main()
