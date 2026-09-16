"""Wait for expanded features, then run the ranking-only loss ablation."""

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
TEMPERATURES = (.05, .01)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expanded-run', type=Path, required=True)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--night-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=36.)
    args = parser.parse_args(argv)
    expanded, base, night, out = (path.resolve() for path in (
        args.expanded_run, args.base_run, args.night_run, args.out))
    if out.exists() or not 0 <= args.wait_hours <= 72:
        parser.error('use a new output and wait-hours in [0,72]')
    out.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'stage': 'waiting_expanded_ranker',
              'updated_unix': time.time(), 'temperatures': list(TEMPERATURES),
              'seeds': list(SEEDS)}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = out / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            upstream = json.loads((expanded / 'status.json').read_text())
            if upstream.get('stage') == 'complete':
                break
            if upstream.get('stage') == 'failed':
                raise RuntimeError('expanded-ranker upstream failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('expanded-ranker missed objective-ablation deadline')
            time.sleep(30)
        update('training')
        rankers = out / 'rankers'; rankers.mkdir()

        def fit(item):
            temperature, seed = item
            name = f'rank_t{temperature:.2f}_s{seed}'
            command = [
                sys.executable, '-u', str(ROOT / 'scripts/train_expanded_ranker_objective.py'),
                '--train-archives', str(base / 'train_archives'),
                '--train-features', str(expanded / 'features/train32'),
                '--train-labels', str(base / 'train_labels'),
                '--additional-train-archives', str(night / 'extra_train_archives'),
                '--additional-train-features', str(expanded / 'features/train128'),
                '--additional-train-labels', str(night / 'extra_train_labels'),
                '--val-archives', str(base / 'val_archives'),
                '--val-features', str(expanded / 'features/val8'),
                '--val-labels', str(base / 'val_labels'),
                '--split-audit', str(night / 'indexes/audit.json'),
                '--train-limit', '160', '--epochs', '100', '--seed', str(seed),
                '--temperature', str(temperature), '--device', 'cpu',
                '--out', str(rankers / name),
            ]
            with (out / f'{name}.log').open('x') as handle:
                return subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT).returncode

        grid = [(temperature, seed) for temperature in TEMPERATURES for seed in SEEDS]
        with ThreadPoolExecutor(max_workers=3) as pool:
            codes = list(pool.map(fit, grid))
        if any(codes):
            raise RuntimeError('ranking-only objective grid incomplete')
        update('summarizing')
        runs = []
        for temperature, seed in grid:
            report = json.loads((rankers / f'rank_t{temperature:.2f}_s{seed}' /
                                 'report.json').read_text())
            runs.append({'temperature': temperature, 'seed': seed,
                         'best_epoch': report['best_epoch'],
                         **report['metrics']['leaderboards']})
        payload = {
            'scope': 'internal validation reused for epoch and loss diagnostics',
            'comparison': 'same 51D features/network/data; ranking-only temperatures',
            'runs': runs,
        }
        (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
        lines = ['# Expanded-ranker objective ablation', '', payload['scope'], '',
                 '| Objective | Initial top1 | Final top1 | Union top1 | Union ≥0.23 |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for temperature in TEMPERATURES:
            rows = [row for row in runs if row['temperature'] == temperature]
            fmt = lambda stage: (f'{statistics.mean(r[stage]["mean_top1"] for r in rows):.4f} '
                                 f'± {statistics.stdev(r[stage]["mean_top1"] for r in rows):.4f}')
            hits = [row['union']['hits_0.23'] for row in rows]
            lines.append(f'| ranking-only T={temperature:.2f} | {fmt("initial")} | '
                         f'{fmt("final")} | {fmt("union")} | {min(hits)}–{max(hits)}/8 |')
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        update('complete')
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
