"""One explicit GPU, one training trajectory, then fixed oracle evaluation.

This entry point does not detach itself. Start it in the background only after
the user authorizes the GPU and launch. All child processes remain serial.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_conditioned_overnight import atomic_json, interrupted, run_task


def build_training_command(runtime, config, out, seconds):
    command = [sys.executable, str(runtime / 'scripts/train_anchored_entry.py')]
    for key, value in config.items():
        if value is None or value is False:
            continue
        command.append('--' + key.replace('_', '-'))
        if value is not True:
            command.append(str(value))
    command.extend(['--out', str(out)])
    if seconds is not None:
        command.extend(['--max-seconds', str(seconds)])
    return command


def resume_at_checkpoint(source, gpu):
    """Hand over only our recorded queue, after an atomic nonzero checkpoint exists."""
    import torch

    state = json.loads((source / 'status.json').read_text())
    if state['physical_gpu'] != gpu or state['current'] != 'training':
        raise ValueError('source must be this GPU\'s running training queue')
    parent_pid = state['pid']
    child_pid = state['tasks'][-1]['pid']
    expected = str(source.resolve())
    for pid in (parent_pid, child_pid):
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if expected not in command or 'anchored' not in command:
            raise ValueError('source process identity mismatch; will not signal it')
    while True:
        candidates = sorted((source / 'training').glob('step_*.pt'),
                            key=lambda path: int(path.stem.split('_')[-1]))
        candidates = [path for path in candidates if int(path.stem.split('_')[-1]) > 0]
        if candidates:
            checkpoint = candidates[-1]
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            best = source / 'training' / f'step_{saved["validation_state"]["best_step"]}.pt'
            if not best.is_file():
                raise FileNotFoundError(best)
            break
        if not Path(f'/proc/{parent_pid}').exists() or not Path(f'/proc/{child_pid}').exists():
            raise RuntimeError('source exited before handover checkpoint; not restarting blindly')
        time.sleep(5)
    os.kill(parent_pid, signal.SIGTERM)
    # The old supervisor stops its owned process group on SIGTERM.
    for _ in range(60):
        stat = Path(f'/proc/{child_pid}/stat')
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            return checkpoint, best
        time.sleep(1)
    raise RuntimeError('old training did not stop; refusing overlapping GPU jobs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'train-config', 'panel-config', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True, help='User-assigned physical GPU ID')
    parser.add_argument('--hours', type=float, default=None,
                        help='Optional wall-clock limit; omitted means no time limit')
    parser.add_argument('--resume-from-queue', type=Path,
                        help='Wait for a saved update, stop this owned queue, and restore its full state')
    parser.add_argument('--evaluation-minutes', type=float, default=45.)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (args.gpu < 0 or (args.hours is not None and not 0 < args.hours <= 24)
            or args.evaluation_minutes <= 0 or args.out.exists()):
        parser.error('invalid budget/GPU or existing output directory')
    runtime, out = args.runtime_root.resolve(), args.out.resolve()
    train = json.loads(args.train_config.read_text())
    panel = json.loads(args.panel_config.read_text())
    if (train.get('scope') != 'training' or train.get('seed') != 7315
            or train.get('conditioning_task') not in ('dual_binary', 'dual_core', 'dual_contact_core', 'dual_assembly')
            or not (train.get('coupled_linker_action') is True or train.get('prior_joint_action') is True)
            or train.get('reset_action_adapter') is not True
            or any(key in train for key in ('out', 'resume', 'resume_best', 'max_seconds'))):
        raise ValueError('night queue requires an explicit fresh unified-action training contract')
    if panel.get('n_walkers') != 32 or panel.get('steps') != 20 or panel.get('seed') != 7315:
        raise ValueError('fixed single-seed evaluation budget required')
    seconds = None if args.hours is None else args.hours * 3600
    train_seconds = None if seconds is None else seconds - args.evaluation_minutes * 60 - 300
    if train_seconds is not None and train_seconds <= 0:
        raise ValueError('reserve five minutes for final validation/checkpoint publication')
    if seconds is None:
        train['max_updates'] = 0
    training_command = build_training_command(runtime, train, out / 'training', train_seconds)
    evaluation_command = [sys.executable, str(runtime / 'scripts/evaluate_anchored_panel.py'),
        '--selection', str(out / 'training/best.json'), '--out', str(out / 'evaluation')]
    for key, value in panel.items():
        if value is None or value is False:
            continue
        evaluation_command.append('--' + key.replace('_', '-'))
        if isinstance(value, list):
            evaluation_command.extend(map(str, value))
        elif value is not True:
            evaluation_command.append(str(value))
    plan = {'physical_gpu': args.gpu, 'hours': args.hours, 'seed': 7315,
            'training': training_command, 'evaluation': evaluation_command,
            'ranking': False, 'tasks': [], 'status': 'prepared'}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'plan.json', plan)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', QUEUE_RUNTIME=str(runtime))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    if args.resume_from_queue:
        plan.update(stage='waiting_for_handover_checkpoint', pid=os.getpid(),
                    resume_from_queue=str(args.resume_from_queue.resolve()), deadline_unix=None)
        atomic_json(out / 'status.json', plan)
        checkpoint, best = resume_at_checkpoint(args.resume_from_queue.resolve(), args.gpu)
        train.update(reset_action_adapter=False, resume=str(checkpoint), resume_best=str(best))
        training_command = build_training_command(runtime, train, out / 'training', train_seconds)
        plan.update(training=training_command, resume_checkpoint=str(checkpoint))
        atomic_json(out / 'plan.json', plan)
    start = time.time()
    deadline = None if seconds is None else start + seconds
    plan.update(pid=os.getpid(), start_unix=start, deadline_unix=deadline)
    if run_task(training_command, 'training', out, environment, plan, deadline):
        if not (out / 'training/best.json').exists():
            raise RuntimeError('training did not publish a selected checkpoint; do not evaluate partial weights')
        if run_task(evaluation_command, 'evaluation', out, environment, plan, deadline):
            plan.update(stage='completed', current=None)
    plan.update(elapsed_seconds=time.time() - start, deadline_unix=deadline)
    atomic_json(out / 'status.json', plan)


if __name__ == '__main__':
    main()
