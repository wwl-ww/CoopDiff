"""Wait for the post-ranker grid, then run the predeclared 256-to-64 selector grid."""

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
SIZES = (32, 80, 160)
SEEDS = (7315, 7316, 7317)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--night-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=12.)
    args = parser.parse_args(argv)
    base, night, out, project = (path.resolve() for path in
                                 (args.base_run, args.night_run, args.run_dir, args.project_root))
    if out.exists() or not 0 <= args.wait_hours <= 24:
        parser.error('use a new output directory and wait-hours in [0,24]')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'claim.json').write_text(json.dumps({'pid': os.getpid(), 'started_unix': time.time()}, indent=2))
    status = {'pid': os.getpid(), 'base_run': str(base), 'night_run': str(night),
              'runtime': str(ROOT), 'plan': {'pool': 256, 'keep': 64,
              'sizes': list(SIZES), 'seeds': list(SEEDS), 'fits': 9}}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = out / 'status.json.tmp'; temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json'); print(stage, details, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')

    def execute(script, argv, log):
        command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, argv)]
        with log.open('x') as handle:
            print(json.dumps({'command': command}), file=handle, flush=True)
            return subprocess.run(command, cwd=project, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode

    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        update('waiting_post_ranker')
        while True:
            upstream = json.loads((night / 'queue_status.json').read_text())
            if upstream['stage'] == 'complete':
                break
            if upstream['stage'] == 'failed':
                raise RuntimeError('post-ranker upstream failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('post-ranker did not complete before selector deadline')
            time.sleep(30)
        jobs = [
            ('train32', base / 'train_archives', base / 'indexes/train_label_refs.json'),
            ('train128', night / 'extra_train_archives', night / 'indexes/extra_train_label_refs.json'),
            ('val8', base / 'val_archives', base / 'indexes/val_label_refs.json'),
        ]
        for name, archives, refs in jobs:
            update('scoring_' + name)
            labels = out / (name + '_labels')
            def score(shard):
                return execute('score_initial_pool.py', [
                    '--archives', archives, '--label-refs', refs, '--out', labels,
                    '--deepternary-root', project / 'DeepTernary', '--project-root', project,
                    '--num-shards', 2, '--shard-index', shard], out / f'{name}_score_{shard}.log')
            with ThreadPoolExecutor(max_workers=2) as pool:
                codes = list(pool.map(score, (0, 1)))
            if any(codes):
                raise RuntimeError(f'{name} full-pool scoring failed; no subset fitting')
        update('training_selectors')
        rankers = out / 'rankers'; rankers.mkdir()
        grid = [(size, seed) for size in SIZES for seed in SEEDS]
        def fit(item):
            size, seed = item; name = f'n{size}_s{seed}'
            return execute('train_initial_selector.py', [
                '--train-archives', base / 'train_archives', night / 'extra_train_archives',
                '--train-labels', out / 'train32_labels', out / 'train128_labels',
                '--val-archives', base / 'val_archives', '--val-labels', out / 'val8_labels',
                '--split-audit', night / 'indexes/audit.json', '--train-limit', size,
                '--epochs', 100, '--seed', seed, '--device', 'cpu', '--out', rankers / name,
            ], out / f'{name}.log')
        with ThreadPoolExecutor(max_workers=3) as pool:
            codes = list(pool.map(fit, grid))
        if any(codes):
            raise RuntimeError('selector grid incomplete; do not report successful subset')
        update('summarizing')
        rows = []
        for size, seed in grid:
            report = json.loads((rankers / f'n{size}_s{seed}/report.json').read_text())
            rows.append({'size': size, 'seed': seed, **report['metrics']['summary']})
        payload = {'scope': 'internal full-pool initial selector; validation reused for epoch selection',
                   'seed_policy': 'all three seeds reported; no best-seed selection', 'runs': rows}
        (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
        lines = ['# Initial 256-to-64 selector learning curve', '', payload['scope'], '',
                 '| Train cases | Selected-best DockQ | Oracle retention | ≥0.23 retained | Oracle pose retained | Mean BiTSeed/SDPI |',
                 '| ---: | ---: | ---: | ---: | ---: | --- |']
        for size in SIZES:
            selected = [row for row in rows if row['size'] == size]
            fmt = lambda key: f'{statistics.mean(r[key] for r in selected):.4f} ± {statistics.stdev(r[key] for r in selected):.4f}'
            hits = [row['selected_hits_0.23'] for row in selected]
            retained = [row['oracle_index_retained'] for row in selected]
            bit = statistics.mean(row['mean_selected_bitseed'] for row in selected)
            sdpi = statistics.mean(row['mean_selected_sdpi'] for row in selected)
            lines.append(f'| {size} | {fmt("mean_selected_best")} | {fmt("mean_case_oracle_retention")} | '
                         f'{min(hits)}–{max(hits)}/8 | {min(retained)}–{max(retained)}/8 | {bit:.1f}/{sdpi:.1f} |')
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        update('complete')
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
