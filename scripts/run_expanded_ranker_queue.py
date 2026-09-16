"""Extract heavy-atom features and compare a three-seed post-ranker to fixed baselines."""

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
        'features': 51, 'train_cases': 160, 'validation_cases': 8,
        'seeds': list(SEEDS), 'fits': len(SEEDS)}}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = out / 'status.json.tmp'; temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json'); print(stage, details, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def execute(script, argv, log):
        command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, argv)]
        with log.open('x') as handle:
            print(json.dumps({'command': command}), file=handle, flush=True)
            return subprocess.run(command, cwd=project, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode

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
        feature_root = out / 'features'; feature_root.mkdir()
        extraction = (
            ('train32', base / 'train_archives', night / 'indexes/train_inference.json'),
            ('train128', night / 'extra_train_archives', night / 'indexes/extra_train_inference.json'),
            ('val8', base / 'val_archives', night / 'indexes/val_inference.json'),
        )
        for name, archives, index in extraction:
            update('extracting_' + name)
            destination = feature_root / name
            def extract(shard):
                return execute('extract_expanded_ranking_features.py', [
                    '--archives', archives, '--index', index, '--out', destination,
                    '--num-shards', 2, '--shard-index', shard],
                    out / f'{name}_extract_{shard}.log')
            with ThreadPoolExecutor(max_workers=2) as pool:
                codes = list(pool.map(extract, (0, 1)))
            if any(codes):
                raise RuntimeError(f'{name}: expanded feature extraction failed')
        update('training_expanded_rankers')
        rankers = out / 'rankers'; rankers.mkdir()
        def fit(seed):
            return execute('train_expanded_geometry_ranker.py', [
                '--train-archives', base / 'train_archives', '--train-features',
                feature_root / 'train32', '--train-labels', base / 'train_labels',
                '--additional-train-archives', night / 'extra_train_archives',
                '--additional-train-features', feature_root / 'train128',
                '--additional-train-labels', night / 'extra_train_labels',
                '--val-archives', base / 'val_archives', '--val-features', feature_root / 'val8',
                '--val-labels', base / 'val_labels', '--split-audit', night / 'indexes/audit.json',
                '--train-limit', 160, '--epochs', 100, '--seed', seed, '--device', 'cpu',
                '--out', rankers / f'n160_s{seed}'], out / f'n160_s{seed}.log')
        with ThreadPoolExecutor(max_workers=3) as pool:
            codes = list(pool.map(fit, SEEDS))
        if any(codes):
            raise RuntimeError('expanded ranker grid incomplete')
        update('summarizing')
        runs = []
        for seed in SEEDS:
            report = json.loads((rankers / f'n160_s{seed}' / 'report.json').read_text())
            runs.append({'seed': seed, 'best_epoch': report['best_epoch'],
                         **report['metrics']['leaderboards']})
        payload = {'scope': 'internal validation reused for epoch selection',
                   'comparison': 'expanded geometry uses same labels/loss as fixed n160 baselines',
                   'runs': runs}
        (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
        lines = ['# Expanded-geometry post-ranker', '', payload['scope'], '',
                 '| Stage | Mean top1 | Mean oracle | ≥0.23 hits | ≥0.49 hits |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for stage in ('initial', 'final', 'union'):
            top1 = [run[stage]['mean_top1'] for run in runs]
            oracle = statistics.mean(run[stage]['mean_oracle'] for run in runs)
            h23 = [run[stage]['hits_0.23'] for run in runs]
            h49 = [run[stage]['hits_0.49'] for run in runs]
            lines.append(f'| {stage} | {statistics.mean(top1):.4f} ± '
                         f'{statistics.stdev(top1):.4f} | {oracle:.4f} | '
                         f'{min(h23)}–{max(h23)}/8 | {min(h49)}–{max(h49)}/8 |')
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        update('complete')
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
