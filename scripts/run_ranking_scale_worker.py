#!/usr/bin/env python
"""Collect and label one resumable shard of an expanded ranking dataset."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def atomic_status(path, stage, **extra):
    payload = {'stage': stage, 'updated_unix': time.time(), **extra}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard index')

    run = args.run_root.resolve()
    indexes = run / 'indexes'
    archives = run / 'archives'
    labels = run / 'labels'
    status_path = run / f'worker_{args.shard_index}_status.json'
    lock_path = run / f'worker_{args.shard_index}.lock'
    environment = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'PYTHONUNBUFFERED': '1',
        'TRICOOPDIFF_CACHE_DIR': str(ROOT / 'results/runtime_cache/diffdock'),
    }
    common = ['--num-shards', str(args.num_shards),
              '--shard-index', str(args.shard_index)]
    collect = [
        sys.executable, '-u', str(ROOT / 'scripts/collect_ranking_archive.py'),
        '--index', str(indexes / 'extra_train_inference.json'),
        '--checkpoint', str(ROOT / 'results/model_v3_ranking_expand32_20260915/checkpoint_epoch40.pt'),
        '--out', str(archives), '--cache-dir', str(ROOT / 'results/featurize_cache'),
        '--device', 'cuda', '--walkers', '64', '--steps', '20', '--seed', '31000',
        *common,
    ]
    score = [
        sys.executable, '-u', str(ROOT / 'scripts/score_ranking_archive.py'),
        '--archives', str(archives),
        '--label-refs', str(indexes / 'extra_train_label_refs.json'),
        '--out', str(labels), *common,
    ]
    run.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            atomic_status(status_path, 'collecting', pid=os.getpid(), gpu=args.gpu)
            subprocess.run(collect, cwd=ROOT, env=environment, check=True)
            atomic_status(status_path, 'scoring', pid=os.getpid(), gpu=args.gpu)
            subprocess.run(score, cwd=ROOT,
                           env={**environment, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
            atomic_status(status_path, 'complete', pid=os.getpid(), gpu=args.gpu)
        except Exception as error:
            atomic_status(status_path, 'failed', pid=os.getpid(), gpu=args.gpu,
                          error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
