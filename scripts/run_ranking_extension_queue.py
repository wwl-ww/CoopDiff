"""One persistent worker: train32 extension/scoring, then train128 extension/scoring."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('5', '7'), required=True)
    parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'OMP_NUM_THREADS': '2',
           'TRICOOPDIFF_CACHE_DIR': str(root / 'results/runtime_cache/diffdock')}
    checkpoint = root / 'results/model_v3_ranking_expand32_20260915/checkpoint_epoch40.pt'
    with (root / f'results/ranking_extension_worker{args.shard_index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, parent, archive_name, refs in (
            ('train32', 'model_v3_ranking_expand32_20260915', 'train_archives', 'train_label_refs.json'),
            ('train128', 'model_v3_ranking_night160_20260915', 'extra_train_archives', 'extra_train_label_refs.json'),
        ):
            source = root / 'results' / parent
            out = root / 'results' / f'ranking_fullpool_{name}_20260917'
            common = ['--num-shards', '2', '--shard-index', str(args.shard_index)]
            print(f'START extend {name} GPU={args.gpu}', flush=True)
            subprocess.run([sys.executable, '-u', str(root / 'scripts/extend_ranking_archive.py'),
                            '--runtime-root', str(source / 'runtime_v1'), '--archives', str(source / archive_name),
                            '--checkpoint', str(checkpoint), '--out', str(out), *common],
                           cwd=root, env=env, check=True)
            print(f'START score {name}', flush=True)
            subprocess.run([sys.executable, '-u', str(root / 'scripts/score_ranking_archive.py'),
                            '--archives', str(out), '--label-refs', str(source / 'indexes' / refs),
                            '--out', str(out) + '_labels', *common],
                           cwd=root, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
            print(f'DONE {name}', flush=True)
        print('COMPLETE training-pool extension and labeling', flush=True)


if __name__ == '__main__':
    main()
