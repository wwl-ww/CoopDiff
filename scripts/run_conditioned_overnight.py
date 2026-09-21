"""Run one time-bounded GPU queue: train, sample, then continue training.

Each epoch is checkpointed. At the wall-clock deadline the active child process
group is terminated; the last complete epoch is retained, not called a final run.
No DeepTernary prediction, refinement or ablation task is launched here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def interrupted(signum, frame):
    raise InterruptedError(f'queue received signal {signum}')


def run_task(command, name, root, environment, state, deadline):
    """Only this foreground child may occupy this queue's GPU."""
    if deadline is not None and time.time() >= deadline:
        return False
    record = {'name': name, 'command': command, 'start_unix': time.time(),
              'log': str(root / f'{name}.log')}
    state['tasks'].append(record)
    state.update(stage='running', current=name)
    print(f'START {name}', flush=True)
    with (root / f'{name}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=environment['QUEUE_RUNTIME'],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record['pid'] = child.pid
        atomic_json(root / 'status.json', state)
        try:
            child.wait(timeout=None if deadline is None else max(.01, deadline - time.time()))
        except subprocess.TimeoutExpired:
            stop_child(child)
            record['deadline_interrupted'] = True
        except BaseException:
            stop_child(child)
            state.update(stage='interrupted', current=None)
            record.update(returncode=child.returncode, end_unix=time.time())
            atomic_json(root / 'status.json', state)
            raise
    record.update(returncode=child.returncode, end_unix=time.time())
    if record.get('deadline_interrupted'):
        state.update(stage='deadline_reached', current=None)
    elif child.returncode:
        state.update(stage='failed', current=None)
    else:
        state.update(stage='between_tasks', current=None)
    atomic_json(root / 'status.json', state)
    print(f'END {name}: {state["stage"]}, exit={child.returncode}', flush=True)
    if child.returncode and not record.get('deadline_interrupted'):
        raise RuntimeError(f'{name} failed; subsequent tasks not launched')
    return not record.get('deadline_interrupted')


def training_command(args, target, output, resume, resume_best):
    command = [sys.executable, '-u', str(args.runtime / 'scripts/train_conditioned_entry.py'),
               '--parent', str(args.parent), '--train-data', str(args.prepared / 'train_index.json'),
               '--val-data', str(args.prepared / 'val_index.json'),
               '--expert-model-dir', str(args.project_root / 'checkpoints/score_model'),
               '--cache-dir', str(args.project_root / 'results/featurize_cache'),
               '--bitseed-lazy-cache-dir', str(args.endpoint_cache),
               '--output-dir', str(output), '--epochs', str(target),
               '--save-interval', '1', '--log-interval', '50',
               '--condition-pocket-only', '0', '--seed', str(args.seed)]
    if resume:
        command.extend(['--resume', str(resume), '--resume-best', str(resume_best)])
    return command


def verify_training(path, target):
    import torch

    payload = torch.load(path / 'final_model.pt', map_location='cpu', weights_only=False)
    if payload['training_state']['epoch'] != target:
        raise ValueError('final checkpoint epoch mismatch')
    for field in ('trifuse_state_dict', 'ema_trifuse_state_dict'):
        if not all(torch.isfinite(t).all() for t in payload[field].values()):
            raise ValueError('nonfinite final checkpoint')


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--hours', type=float, default=10.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--endpoint-cache', type=Path)
    args = parser.parse_args()
    if args.physical_gpu < 0 or not 0 < args.hours <= 24:
        parser.error('GPU must be nonnegative; duration must be in (0, 24] hours')
    for key in ('out', 'prepared', 'project_root', 'parent', 'manifest', 'resume', 'endpoint_cache'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.runtime = args.prepared / 'runtime'
    if args.endpoint_cache is None:
        args.endpoint_cache = args.out / 'bitseed_c1m1_cache'
    preparation = json.loads((args.prepared / 'preparation.json').read_text())
    for relative, expected in preparation['runtime_sha256'].items():
        if hashlib.sha256((args.runtime / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'runtime source changed: {relative}')
    for path in (args.parent, args.manifest, *([args.resume] if args.resume else [])):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.hours * 3600
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TRICOOPDIFF_CACHE_DIR=str(args.project_root / 'results/runtime_cache/diffdock'),
                       QUEUE_RUNTIME=str(args.runtime))
    state = {'pid': os.getpid(), 'physical_gpu': args.physical_gpu, 'seed': args.seed,
             'start_unix': started, 'deadline_unix': deadline, 'hours': args.hours,
             'stage': 'starting', 'current': None, 'tasks': [],
             'resume': str(args.resume) if args.resume else None,
             'resume_scope': 'optimizer/EMA restored; legacy checkpoint does not preserve RNG state',
             'policy': 'fixed architecture; independent generation; no ablation or DT refinement',
             'deadline_policy': 'stop active task, retain last completed epoch; no success claim',
             'prepared': str(args.prepared)}
    atomic_json(args.out / 'status.json', state)
    atomic_json(args.out / 'plan.json', {
        'training_targets': [40, 80, 120, 160],
        'after_each_target': '20-step SDPI and BiTSeed implementation sampling, no accuracy claim',
        'save_interval': 1, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(args).items()}})
    resume = args.resume
    resume_best = args.resume
    try:
        for target in (40, 80, 120, 160):
            if time.time() >= deadline:
                break
            output = args.out / f'train_to_epoch{target}'
            command = training_command(args, target, output, resume, resume_best)
            if not run_task(command, f'train_to_epoch{target}', args.out,
                            environment, state, deadline):
                break
            verify_training(output, target)
            resume = output / 'final_model.pt'
            resume_best = output / 'best_model.pt'
            command = [sys.executable, '-u', str(args.runtime / 'scripts/smoke_binary_geometry.py'),
                       '--project-root', str(args.project_root),
                       '--checkpoint', str(output / 'best_model.pt'),
                       '--case-dir', str(args.project_root / 'DeepTernary/output/protac22/6HAY_B_A_FX8'),
                       '--condition-manifest', str(args.manifest), '--seed', str(args.seed),
                       '--out', str(args.out / f'sampling_epoch{target}.json')]
            if not run_task(command, f'sampling_epoch{target}', args.out,
                            environment, state, deadline):
                break
        if state['stage'] != 'deadline_reached':
            state.update(stage='deadline_reached' if time.time() >= deadline else 'complete',
                         current=None)
    except BaseException as error:
        state.update(stage='failed', current=None, error=repr(error))
        raise
    finally:
        state['end_unix'] = time.time()
        state['retained_checkpoints'] = sorted(str(p) for p in args.out.glob('train_*/*.pt'))
        atomic_json(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
