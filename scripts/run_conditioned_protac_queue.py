"""Wait for conditioned training, then run fixed-arm PROTAC evaluations.

Each ``--task`` is ``NAME CHECKPOINT MANIFEST_DIR``. Tasks in one queue run
serially on one physical GPU; multiple queue processes may use different GPUs.
The checkpoint's sibling ``final_model.pt`` is the completion sentinel, while
the named checkpoint (normally ``best_model.pt``) is evaluated.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = re.compile(r'checkpoint_epoch[1-9][0-9]*[.]pt')


def supported_checkpoint_name(name):
    """Accept explicit best/final or numbered periodic checkpoints."""
    return name in ('best_model.pt', 'final_model.pt') or bool(
        CHECKPOINT_NAME.fullmatch(name))


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def evaluation_command(args, task, destination):
    name, checkpoint, manifests = task
    return [
        sys.executable, '-u',
        str(args.runtime_root / 'scripts/evaluate_selected_refinement.py'),
        '--runtime-root', str(args.runtime_root),
        '--project-root', str(args.project_root),
        '--index', str(args.index),
        '--checkpoint', str(checkpoint),
        '--condition-manifest-dir', str(manifests),
        '--allow-conditioned-checkpoint',
        '--out-dir', str(destination),
        '--device', 'cuda:0',
        '--cache-dir', str(args.cache_dir),
        '--seed', str(args.seed),
        '--steps', str(args.steps),
        '--arms', args.arms,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', nargs=3, action='append', required=True,
                        metavar=('NAME', 'CHECKPOINT', 'MANIFEST_DIR'))
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('results/featurize_cache_proposal_oracle'))
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--arms', default='hybrid_56_8')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=31000)
    parser.add_argument('--wait-hours', type=float, default=72.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if (args.physical_gpu < 0 or args.steps < 1 or args.wait_hours < 0
            or args.poll_seconds <= 0 or args.out.exists()):
        parser.error('invalid GPU/step/wait settings or output directory already exists')
    args.project_root = args.project_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.index = args.index.resolve()
    args.cache_dir = (args.project_root / args.cache_dir).resolve()
    tasks = [(name, Path(checkpoint).resolve(), Path(manifests).resolve())
             for name, checkpoint, manifests in args.task]
    if len({name for name, _, _ in tasks}) != len(tasks):
        raise ValueError('task names must be unique')
    for name, checkpoint, manifests in tasks:
        if name != Path(name).name or name in ('.', '..'):
            raise ValueError('unsafe task name')
        if not manifests.is_dir():
            raise FileNotFoundError(manifests)
        if not supported_checkpoint_name(checkpoint.name):
            raise ValueError(
                'evaluate an explicit best_model.pt, final_model.pt, or '
                'checkpoint_epoch<N>.pt')
    args.out.mkdir(parents=True)
    status_path = args.out / 'status.json'
    status = {
        'protocol': 'conditioned_protac_evaluation_queue_v1',
        'pid': os.getpid(), 'physical_gpu': args.physical_gpu,
        'tasks': [name for name, _, _ in tasks], 'completed': [],
        'stage': 'waiting_training', 'current': None,
    }
    atomic_json(status, status_path)
    deadline = time.time() + args.wait_hours * 3600
    while True:
        ready = all((checkpoint.parent / 'final_model.pt').is_file()
                    and checkpoint.is_file() for _, checkpoint, _ in tasks)
        if ready:
            break
        if time.time() >= deadline:
            status.update(stage='failed', error='timed out waiting for training checkpoints')
            atomic_json(status, status_path)
            raise RuntimeError(status['error'])
        time.sleep(args.poll_seconds)
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    environment['TRICOOPDIFF_CACHE_DIR'] = str(
        args.project_root / 'results/runtime_cache/diffdock')
    for task in tasks:
        name = task[0]
        destination = args.out / name
        status.update(stage='evaluating', current=name, updated_unix=time.time())
        atomic_json(status, status_path)
        command = evaluation_command(args, task, destination)
        with (args.out / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=args.project_root, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            status.update(stage='failed', returncode=result.returncode,
                          error=f'{name} evaluation failed')
            atomic_json(status, status_path)
            raise SystemExit(result.returncode)
        summary = json.loads((destination / 'summary.json').read_text())
        if summary['requested_cases'] != len(summary['completed_cases']) or summary['failures']:
            status.update(stage='failed', error=f'{name} did not complete every case')
            atomic_json(status, status_path)
            raise RuntimeError(status['error'])
        status['completed'].append(name)
    status.update(stage='complete', current=None, updated_unix=time.time())
    atomic_json(status, status_path)


if __name__ == '__main__':
    main()
