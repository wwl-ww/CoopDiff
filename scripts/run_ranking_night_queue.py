"""Serial overnight extension: wait, collect new cases, score, then ranker learning curves.

Reuse the upstream immutable generator runtime and validation archives. Only the
small ranker trainer uses the new runtime; no archive manifests are rewritten.
"""

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


def upstream_state(base):
    status = json.loads((base / 'pipeline_status.json').read_text())
    stage = status['stage']
    if stage not in ('collecting', 'scoring', 'auditing_labels', 'training_rankers', 'complete'):
        raise RuntimeError(f'upstream failed: {stage}')
    return stage == 'complete' and (base / 'validation_ranking_audit/comparison.md').is_file()


def validate_panel(base, run):
    read = lambda path: json.loads(path.read_text())
    previous, current = read(base / 'indexes/audit.json'), read(run / 'indexes/audit.json')
    if (len(previous['train_cases']) != 32 or len(previous['val_cases']) != 8
            or len(current['train_cases']) != 160 or len(set(current['train_cases'])) != 160):
        raise ValueError('overnight plan requires 32 existing +128 new training cases and 8 validation cases')
    for key in ('seed', 'source_sha256', 'benchmark_pdbs_excluded'):
        if previous[key] != current[key]:
            raise ValueError(f'panel source drift: {key}')
    for kind in ('inference', 'label_refs'):
        old = read(base / 'indexes' / f'train_{kind}.json')['entries']
        full = read(run / 'indexes' / f'train_{kind}.json')['entries']
        extra = read(run / 'indexes' / f'extra_train_{kind}.json')['entries']
        if full != old + extra or len(extra) != 128:
            raise ValueError('new-case index is not the predeclared suffix')
        if read(base / 'indexes' / f'val_{kind}.json') != read(run / 'indexes' / f'val_{kind}.json'):
            raise ValueError('validation set must be reused exactly')
    return {'new_cases': 128, 'new_walkers': 64, 'new_trajectories': 8192,
            'training_sizes': list(SIZES), 'seeds': list(SEEDS), 'ranker_fits': 18,
            'validation_cases': 8, 'validation_walkers': 256, 'epochs': 100,
            'warning': 'fixed epochs, not equal optimizer-update budgets; internal validation only'}


def summarize(run, base):
    records = []
    for mode in ('geometry', 'frozen_geometry'):
        for size in SIZES:
            for seed in SEEDS:
                folder = run / 'rankers' / f'{mode}_n{size}_s{seed}'
                report = json.loads((folder / 'report.json').read_text())
                audit = json.loads((folder / 'audit.json').read_text())
                records.append({'features': mode, 'train_cases': size, 'seed': seed,
                                'best_epoch': report['best_epoch'], 'optimizer_updates': report['optimizer_updates'],
                                'label_support': report['label_support'],
                                'fit_diagnostics': report['fit_diagnostics'], 'metrics': audit['summary']})
    baseline = json.loads((base / 'validation_ranking_audit/comparison.json').read_text())['reports']['r0_fixed_geometry']
    payload = {'scope': 'internal validation reused for epoch selection; not holdout or a selector experiment',
               'seed_policy': 'report every predeclared seed; do not select the best seed',
               'compute_policy': '100 epochs per size; optimizer updates grow with training cases',
               'fixed_r0': baseline, 'runs': records}
    (run / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Overnight ranking learning curves', '', payload['scope'], '',
             'Mean ± sample SD across all three seeds. Hits are the seed range, each out of 8 cases.',
             '100 epochs per training size: data coverage and optimizer-update count both increase.', '',
             '| Features | Train cases | Initial top1 | Final top1 | Union top1 | Union ≥0.23 | Union ≥0.49 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode in ('geometry', 'frozen_geometry'):
        for size in SIZES:
            rows = [r for r in records if r['features'] == mode and r['train_cases'] == size]
            values = []
            for stage in ('initial', 'final', 'union'):
                scores = [r['metrics'][f'all/{stage}']['mean_top1'] for r in rows]
                values.append(f'{statistics.mean(scores):.4f} ± {statistics.stdev(scores):.4f}')
            for threshold in ('0.23', '0.49'):
                hits = [r['metrics']['all/union']['thresholds'][threshold]['top1_hit'] for r in rows]
                values.append(f'{min(hits)}–{max(hits)}/8')
            lines.append(f'| {mode} | {size} | ' + ' | '.join(values) + ' |')
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, type=int, required=True)
    args = parser.parse_args()
    base, run, project = args.base_run.resolve(), args.run_dir.resolve(), args.project_root.resolve()
    if len(set(args.gpus)) != 2:
        parser.error('two distinct authorized physical GPUs are required')
    plan = validate_panel(base, run)
    generator_runtime = Path(json.loads((base / 'pipeline_status.json').read_text())['runtime']).resolve()
    # Exclusive queue claim prevents two launches from duplicating GPU workers.
    with (run / 'queue_claim.json').open('x') as handle:
        json.dump({'pid': os.getpid(), 'started_unix': time.time(), 'plan': plan}, handle, indent=2)
    status = {'pid': os.getpid(), 'started_unix': time.time(), 'base_run': str(base),
              'generator_runtime': str(generator_runtime), 'ranking_runtime': str(ROOT),
              'gpus': args.gpus, 'plan': plan}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = run / 'queue_status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(run / 'queue_status.json')
        print(stage, details, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               TRICOOPDIFF_CACHE_DIR=str(project / 'results/runtime_cache/diffdock'))

    def execute(runtime, script, argv, logfile, gpu=None):
        command = [sys.executable, '-u', str(runtime / 'scripts' / script), *map(str, argv)]
        with logfile.open('x') as log:
            print(json.dumps({'command': command, 'gpu': gpu}), file=log, flush=True)
            return subprocess.run(command, cwd=project, env=dict(env, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu)),
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    try:
        update('waiting_upstream')
        deadline = time.monotonic() + 12 * 3600
        while not upstream_state(base):
            if time.monotonic() >= deadline:
                raise TimeoutError('upstream did not complete within 12 hours; no GPU work started')
            time.sleep(30)
            update('waiting_upstream')
        update('collecting_extension')

        def collect(shard):
            return execute(generator_runtime, 'collect_ranking_archive.py', [
                '--project-root', project, '--index', run / 'indexes/extra_train_inference.json',
                '--checkpoint', base / 'checkpoint_epoch40.pt', '--out', run / 'extra_train_archives',
                '--cache-dir', project / 'results/featurize_cache_proposal_oracle', '--walkers', 64,
                '--num-shards', 2, '--shard-index', shard], run / f'collect_gpu{args.gpus[shard]}.log', args.gpus[shard])

        with ThreadPoolExecutor(max_workers=2) as pool:
            collected = list(pool.map(collect, (0, 1)))
        update('scoring_extension', collection_returncodes=collected)
        def score(shard):
            return execute(ROOT, 'score_ranking_archive.py', [
                '--project-root', project, '--deepternary-root', project / 'DeepTernary',
                '--archives', run / 'extra_train_archives', '--label-refs', run / 'indexes/extra_train_label_refs.json',
                '--out', run / 'extra_train_labels', '--num-shards', 2, '--shard-index', shard],
                run / f'score_extension_{shard}.log')

        with ThreadPoolExecutor(max_workers=2) as pool:
            scored = list(pool.map(score, (0, 1)))
        if any(collected) or any(scored):
            raise RuntimeError('extension collection/labels incomplete; no subset fitting')
        update('auditing_extension')
        if execute(ROOT, 'audit_ranking_label_balance.py', [
                '--archives', run / 'extra_train_archives', '--labels', run / 'extra_train_labels',
                '--out', run / 'extra_train_label_support.json'], run / 'audit_extension.log'):
            raise RuntimeError('extension label identity audit failed')
        update('training_curves')
        (run / 'rankers').mkdir(exist_ok=False)

        def train_grid(shard):
            mode = ('geometry', 'frozen_geometry')[shard]
            for size in SIZES:
                for seed in SEEDS:
                    name = f'{mode}_n{size}_s{seed}'
                    out = run / 'rankers' / name
                    argv = ['--train-archives', base / 'train_archives', '--train-labels', base / 'train_labels',
                            '--additional-train-archives', run / 'extra_train_archives',
                            '--additional-train-labels', run / 'extra_train_labels',
                            '--val-archives', base / 'val_archives', '--val-labels', base / 'val_labels',
                            '--split-audit', run / 'indexes/audit.json', '--train-limit', size, '--seed', seed,
                            '--epochs', 100, '--diagnostics', '--device', 'cuda', '--out', out]
                    if shard == 0:
                        argv.append('--geometry-only')
                    if execute(ROOT, 'train_state_ranker.py', argv, run / f'{name}.log', args.gpus[shard]):
                        return 1
                    if execute(ROOT, 'rank_pose_archive.py', ['--archives', base / 'val_archives',
                            '--ranker', out / 'inference.pt', '--out', out / 'predictions'], out / 'inference.log'):
                        return 1
                    if execute(ROOT, 'audit_state_ranking.py', ['--predictions', out / 'predictions',
                            '--labels', base / 'val_labels', '--out', out / 'audit.json'], out / 'audit.log'):
                        return 1
            return 0

        with ThreadPoolExecutor(max_workers=2) as pool:
            trained = list(pool.map(train_grid, (0, 1)))
        if any(trained):
            raise RuntimeError('ranking grid incomplete; inspect per-run logs, do not report a successful subset')
        update('summarizing')
        summarize(run, base)
        update('complete')
        return 0
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    sys.exit(main())
