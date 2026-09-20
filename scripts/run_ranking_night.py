"""Ten-hour soft-budget serial worker for the paired full-pool ranking experiment."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('5', '7'), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--collector-pid', type=int, required=True)
    parser.add_argument('--legacy-pids', nargs='*', type=int, default=[])
    parser.add_argument('--hours', type=float, default=10.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_root.resolve()
    shard = 0 if args.gpu == '5' else 1
    scope = 'all' if shard == 0 else 'future-top64'
    run.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.hours * 3600
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'OMP_NUM_THREADS': '2',
           'TRICOOPDIFF_CACHE_DIR': str(root / 'results/runtime_cache/diffdock')}
    archives = root / 'results/ranking_fullpool_train32_20260917'
    labels = Path(str(archives) + '_labels')
    old = root / 'results/model_v3_ranking_expand32_20260915'
    output = run / f'gpu{args.gpu}_{scope}'
    identities = {pid: process_token(pid) for pid in [args.collector_pid, *args.legacy_pids]}

    def status(stage, **extra):
        value = {'stage': stage, 'gpu': args.gpu, 'pid': os.getpid(), 'scope': scope,
                 'started_unix': started, 'deadline_unix': deadline, 'updated_unix': time.time(), **extra}
        path = run / f'gpu{args.gpu}_status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
        print(json.dumps(value), flush=True)

    def wait_until(predicate, stage):
        status(stage)
        while not predicate():
            if time.time() >= deadline:
                raise TimeoutError(f'night budget exhausted while {stage}; no training was forced')
            time.sleep(15)

    def finished(pid):
        return identities[pid] is None or process_token(pid) != identities[pid]

    with (run / f'gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_until(lambda: finished(args.collector_pid), 'waiting_existing_train32_collector')
            manifest = json.loads((archives / 'manifest.json').read_text())
            own_cases = manifest['cases'][shard::2]
            if any(not (archives / f'{case}.complete.json').exists() for case in own_cases):
                raise RuntimeError('collector ended without all assigned complete archives')
            status('scoring_train32')
            subprocess.run([sys.executable, '-u', str(root / 'scripts/score_ranking_archive.py'),
                            '--archives', str(archives), '--label-refs', str(old / 'indexes/train_label_refs.json'),
                            '--out', str(labels), '--num-shards', '2', '--shard-index', str(shard)],
                           cwd=root, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
            wait_until(lambda: all((labels / f'{case}.json').exists() for case in manifest['cases']),
                       'waiting_complete_train32_labels')
            wait_until(lambda: all(finished(pid) for pid in args.legacy_pids), 'waiting_legacy_gpu_jobs')
            remaining = (deadline - time.time()) / 3600
            if remaining < .25:
                raise TimeoutError('less than 15 minutes left; not launching a partial training experiment')
            command = [sys.executable, '-u', str(run / 'runtime/scripts/train_structural_ranker.py'),
                       '--train-archives', str(archives), '--train-labels', str(labels),
                       '--val-archives', str(old / 'val_archives'), '--val-labels', str(old / 'val_labels'),
                       '--out', str(output), '--device', 'cuda', '--epochs', '20', '--seed', '7315',
                       '--encoder-checkpoint', str(old / 'checkpoint_epoch40.pt'), '--encoder-weights', 'ema',
                       '--candidate-scope', 'archived-pairs', '--objective', 'decision',
                       '--future-weight', '1', '--future-target', 'union', '--budget-weight', '1',
                       '--top-weight', '1', '--quality-weight', '0', '--checkpoint-task', 'future',
                       '--quality-walker-scope', scope, '--backward-chunk-size', '4',
                       '--shuffle-repeats', '0', '--max-hours', str(remaining), '--save-epoch-checkpoints']
            status('training', command=command, remaining_hours=remaining)
            with (run / f'gpu{args.gpu}_training.log').open('a') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
            # Do not leave an early-finishing arm idle: extend the remaining
            # training-only pool in resumable chunks until the night budget ends.
            extra = root / 'results/model_v3_ranking_night160_20260915'
            extra_output = root / 'results/ranking_fullpool_train128_20260917'
            extra_cases = json.loads((extra / 'extra_train_archives/manifest.json').read_text())['cases'][shard::2]
            batches = 0
            while time.time() < deadline:
                if all((extra_output / f'{case}.complete.json').exists() for case in extra_cases):
                    status('scoring_completed_train128', completed_epoch=rows[-1]['epoch'])
                    subprocess.run([sys.executable, '-u', str(root / 'scripts/score_ranking_archive.py'),
                                    '--archives', str(extra_output),
                                    '--label-refs', str(extra / 'indexes/extra_train_label_refs.json'),
                                    '--out', str(extra_output) + '_labels', '--num-shards', '2',
                                    '--shard-index', str(shard)], cwd=root,
                                   env={**env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
                    break  # Never fabricate work if the remaining dataset is exhausted.
                status('extending_train128', completed_epoch=rows[-1]['epoch'],
                       continuation_batch=batches + 1, max_new_pairs=64)
                subprocess.run([sys.executable, '-u', str(root / 'scripts/extend_ranking_archive.py'),
                                '--runtime-root', str(extra / 'runtime_v1'),
                                '--archives', str(extra / 'extra_train_archives'),
                                '--checkpoint', str(old / 'checkpoint_epoch40.pt'), '--out', str(extra_output),
                                '--num-shards', '2', '--shard-index', str(shard), '--max-new-pairs', '64'],
                               cwd=root, env=env, check=True)
                batches += 1
            status('complete', completed_epoch=rows[-1]['epoch'],
                   continuation_batches=batches, elapsed_hours=(time.time() - started) / 3600,
                   note='compare the two arms at a common saved epoch, not their independently selected best epochs')
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
