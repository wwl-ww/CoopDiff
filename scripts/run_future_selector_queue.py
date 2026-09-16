"""Wait for the current-quality selector, then fit paired C40 future-value selectors."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7315, 7316, 7317)
TARGETS = ('final', 'union')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--night-run', type=Path, required=True)
    parser.add_argument('--initial-selector-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=24.)
    args = parser.parse_args(argv)
    base, night, upstream, out, project = (path.resolve() for path in (
        args.base_run, args.night_run, args.initial_selector_run, args.run_dir,
        args.project_root))
    if out.exists() or not 0 <= args.wait_hours <= 48:
        parser.error('use a new output directory and wait-hours in [0,48]')
    out.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'runtime': str(ROOT), 'plan': {
        'refiner': 'C40 EMA', 'train_cases': 160, 'validation_cases': 8,
        'validation_pool': 256, 'keep': 64, 'targets': list(TARGETS),
        'seeds': list(SEEDS), 'fits': len(TARGETS) * len(SEEDS)}}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = out / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json')
        print(stage, details, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def fit(seed):
        codes = []
        for target in TARGETS:
            name = f'{target}_n160_s{seed}'
            command = [sys.executable, '-u', str(ROOT / 'scripts/train_future_selector.py'),
                       '--train-archives', str(base / 'train_archives'),
                       str(night / 'extra_train_archives'), '--train-labels',
                       str(base / 'train_labels'), str(night / 'extra_train_labels'),
                       '--val-archives', str(base / 'val_archives'), '--val-labels',
                       str(base / 'val_labels'), '--split-audit',
                       str(night / 'indexes/audit.json'), '--target', target,
                       '--train-limit', '160', '--epochs', '100', '--seed', str(seed),
                       '--device', 'cpu', '--out', str(out / 'rankers' / name)]
            with (out / f'{name}.log').open('x') as handle:
                print(json.dumps({'command': command}), file=handle, flush=True)
                codes.append(subprocess.run(command, cwd=project, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT).returncode)
            if codes[-1]:
                break
        return codes

    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        update('waiting_current_quality_selector')
        while True:
            state = json.loads((upstream / 'status.json').read_text())
            if state['stage'] == 'complete':
                break
            if state['stage'] == 'failed':
                raise RuntimeError('current-quality selector failed upstream')
            if time.monotonic() >= deadline:
                raise TimeoutError('current-quality selector exceeded wait deadline')
            time.sleep(30)
        update('training_future_selectors')
        (out / 'rankers').mkdir()
        with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
            codes = list(pool.map(fit, SEEDS))
        if any(code for seed_codes in codes for code in seed_codes):
            raise RuntimeError('future-selector grid incomplete')
        update('summarizing')
        runs = []
        for target in TARGETS:
            for seed in SEEDS:
                report = json.loads((out / 'rankers' / f'{target}_n160_s{seed}' /
                                     'report.json').read_text())
                runs.append({'target': target, 'seed': seed,
                             'best_epoch': report['best_epoch'], **report['metrics']})
        payload = {
            'scope': 'internal full-pool validation; validation reused for epoch selection',
            'comparison': 'same initial-only features, final versus union training target',
            'warning': 'three seeds are all reported; no best-seed selection', 'runs': runs,
        }
        (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
        lines = ['# C40 future-value selector', '', payload['scope'], '',
                 '| Target | Final selected/oracle | Final retention | Final ≥0.23 | '
                 'Union selected/oracle | Union retention | Union ≥0.23 | Rescued cases retained |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for target in TARGETS:
            rows = [row for row in runs if row['target'] == target]
            mean_sd = lambda stage, key: (f'{statistics.mean(r[stage][key] for r in rows):.4f} '
                                           f'± {statistics.stdev(r[stage][key] for r in rows):.4f}')
            hit_range = lambda stage: (f'{min(r[stage]["selected_hits_0.23"] for r in rows)}–'
                                       f'{max(r[stage]["selected_hits_0.23"] for r in rows)}/8')
            rescue = (f'{min(r["rescued_cases_retained"] for r in rows)}–'
                      f'{max(r["rescued_cases_retained"] for r in rows)}/'
                      f'{rows[0]["rescued_cases_available"]}')
            lines.append(f'| {target} | {mean_sd("final", "mean_selected_best")} / '
                         f'{statistics.mean(r["final"]["mean_full_pool_oracle"] for r in rows):.4f} | '
                         f'{mean_sd("final", "mean_case_oracle_retention")} | {hit_range("final")} | '
                         f'{mean_sd("union", "mean_selected_best")} / '
                         f'{statistics.mean(r["union"]["mean_full_pool_oracle"] for r in rows):.4f} | '
                         f'{mean_sd("union", "mean_case_oracle_retention")} | {hit_range("union")} | '
                         f'{rescue} |')
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        update('complete')
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
