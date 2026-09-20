"""Detached random/EMA encoder comparison with frozen source snapshots."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 7])
    parser.add_argument('--arms', choices=('random', 'ema'), nargs='+', default=['random', 'ema'])
    parser.add_argument('--interface-layers', type=int, default=0)
    parser.add_argument('--top-weight', type=float, default=0.)
    parser.add_argument('--future-weight', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if (len(args.gpus) != len(args.arms) or len(set(args.gpus)) != len(args.gpus)
            or len(set(args.arms)) != len(args.arms) or args.interface_layers < 0
            or not math.isfinite(args.top_weight) or args.top_weight < 0
            or not math.isfinite(args.future_weight) or args.future_weight < 0
            or args.epochs < 1 or args.out.exists()):
        parser.error('one distinct GPU per unique arm, nonnegative interface layers, positive epochs and a new output directory required')
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'runtime'
    (runtime / 'scripts').mkdir(parents=True)
    shutil.copytree(root / 'tricoopdiff', runtime / 'tricoopdiff',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(root / 'scripts/train_structural_ranker.py', runtime / 'scripts/train_structural_ranker.py')
    base = root / 'results/model_v3_ranking_expand32_20260915'
    extra = root / 'results/model_v3_ranking_night160_20260915'
    initial = root / 'results/model_v3_initial_selector_20260916'
    common = [sys.executable, '-u', str(runtime / 'scripts/train_structural_ranker.py'),
              '--candidate-scope', 'joint', '--objective', 'legacy',
              '--train-archives', str(base / 'train_archives'), str(extra / 'extra_train_archives'),
              '--train-labels', str(base / 'train_labels'), str(extra / 'extra_train_labels'),
              '--train-initial-labels', str(initial / 'train32_labels'), str(initial / 'train128_labels'),
              '--val-archives', str(base / 'val_archives'), '--val-labels', str(base / 'val_labels'),
              '--val-initial-labels', str(initial / 'val8_labels'), '--device', 'cuda',
              '--epochs', str(args.epochs), '--seed', '7315', '--hidden-dim', '128',
              '--lr', '0.0003', '--top-weight', str(args.top_weight), '--quality-weight', '0', '--shuffle-repeats', '8',
              '--interface-layers', str(args.interface_layers)]
    common += ['--future-weight', str(args.future_weight)]
    receipt = {'purpose': 'controlled structural ranking experiment; internal validation only',
               'runtime': str(runtime), 'jobs': []}
    for arm, gpu in zip(args.arms, args.gpus):
        command = common + ['--out', str(out / arm)]
        if arm == 'ema':
            command += ['--encoder-checkpoint', str(base / 'checkpoint_epoch40.pt'), '--encoder-weights', 'ema']
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        log_path = out / f'{arm}.log'
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['jobs'].append({'arm': arm, 'gpu': gpu, 'pid': process.pid,
                                'command': command, 'log': str(log_path)})
        (out / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
