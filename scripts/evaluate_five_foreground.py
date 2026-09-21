"""Foreground supervisor for the five selected overnight checkpoints.

Uses each recorded frozen evaluation command; never restarts training.
SIGINT/SIGTERM stop all owned evaluation process groups.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_conditioned_overnight import atomic_json, stop_child


def main():
    runs = [('three_mode_unified_20260920', task) for task in
            ('dual_binary', 'dual_core', 'dual_contact_core')]
    runs += [('prior_modes_20260920', task) for task in ('prior_joint', 'soft_assembly')]
    children = []

    def interrupted(signum, frame):
        raise InterruptedError(f'foreground evaluation received signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        for folder, task in runs:
            root = (Path('results') / folder / task / 'night_unlimited').resolve()
            old = json.loads((root / 'status.json').read_text())
            old_pid = old['tasks'][-1]['pid']
            stat = Path(f'/proc/{old_pid}/stat')
            if stat.exists() and stat.read_text().split()[2] != 'Z':
                raise RuntimeError(f'{task}: background evaluator still alive')
            command = old['evaluation'].copy()
            output = root / 'evaluation_foreground'
            if output.exists():
                raise FileExistsError(output)
            command[command.index('--out') + 1] = str(output)
            runtime = Path(command[1]).resolve().parents[1]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(old['physical_gpu']),
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                TRICOOPDIFF_CACHE_DIR='/mnt/disk/wl/MoRE-tri/CoopDiff/results/runtime_cache/diffdock')
            with (root / 'evaluation_foreground.log').open('x') as log:
                child = subprocess.Popen(command, cwd=runtime, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(task=task, gpu=old['physical_gpu'], pid=child.pid,
                          supervisor_pid=os.getpid(), command=command, stage='running',
                          started_unix=time.time(), mode='foreground_supervised')
            children.append((child, root, record))
            atomic_json(root / 'foreground_status.json', record)
            print(f'START {task}: GPU {record["gpu"]}, PID {child.pid}', flush=True)
        while any(child.poll() is None for child, _, _ in children):
            for child, root, record in children:
                panel = root / 'evaluation_foreground'
                completed = len(list(panel.glob('*/scores.json')))
                progress = []
                for path in panel.glob('*/sampling.log'):
                    lines = path.read_text().splitlines()
                    if lines:
                        progress.append(f'{path.parent.name}: {lines[-1][:180]}')
                print(f'{record["task"]}: scored={completed}/7, exit={child.poll()}, '
                      + ' | '.join(progress[-1:]), flush=True)
            time.sleep(20)
        if any(child.returncode != 0 for child, _, _ in children):
            raise RuntimeError('one or more evaluation panels failed; inspect foreground logs')
    finally:
        for child, root, record in children:
            stop_child(child)
            record.update(stage='completed' if child.returncode == 0 else 'stopped_or_failed',
                          returncode=child.returncode, ended_unix=time.time())
            atomic_json(root / 'foreground_status.json', record)


if __name__ == '__main__':
    main()
