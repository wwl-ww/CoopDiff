#!/usr/bin/env python
"""Wait for an expanded sealed dataset, then train one setwise seed."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def gpu_idle(gpu):
    result = subprocess.run(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    run = args.run_root.resolve()
    archive_manifest = json.loads((run / 'archives/manifest.json').read_text())
    cases = archive_manifest['cases']
    status_path = args.out.parent / f'{args.out.name}_queue_status.json'
    lock_path = args.out.parent / f'{args.out.name}_queue.lock'
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        atomic_json(status_path, {'stage': stage, 'pid': os.getpid(),
                                  'seed': args.seed, 'gpu': args.gpu,
                                  'updated_unix': time.time(), **extra})

    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                completed = sum((run / 'labels' / f'{case}.json').is_file()
                                for case in cases)
                if completed == len(cases):
                    break
                status('waiting_labels', completed=completed, expected=len(cases))
                time.sleep(args.poll_seconds)
            while not gpu_idle(args.gpu):
                status('waiting_gpu', completed=len(cases), expected=len(cases))
                time.sleep(args.poll_seconds)
            command = [
                sys.executable, '-u', str(ROOT / 'scripts/train_setwise_future_selector.py'),
                '--train-archives',
                str(ROOT / 'results/ranking_fullpool_train32_20260917'),
                str(ROOT / 'results/ranking_fullpool_train128_20260917'),
                str(run / 'archives'),
                '--train-labels',
                str(ROOT / 'results/ranking_fullpool_train32_20260917_labels'),
                str(ROOT / 'results/ranking_fullpool_train128_20260917_labels'),
                str(run / 'labels'),
                '--val-archives',
                str(ROOT / 'results/model_v3_ranking_expand32_20260915/val_archives'),
                '--val-labels',
                str(ROOT / 'results/model_v3_ranking_expand32_20260915/val_labels'),
                '--split-audit', str(run / 'indexes/audit.json'),
                '--target', 'union', '--out', str(args.out), '--epochs', '50',
                '--seed', str(args.seed), '--device', 'cuda',
            ]
            status('training', command=command)
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
                           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                           'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            status('complete')
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
