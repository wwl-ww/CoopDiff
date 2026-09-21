"""Run one frozen data-scale arm only after sealed inputs and an idle GPU exist."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def missing_inputs(plan):
    return [p for p in plan['required_paths'] if not Path(p).is_file()]


def gpu_idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip()


def preflight_command(command, out):
    result = list(command)
    result[result.index('--out') + 1] = str(out)
    return result + ['--audit-only']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    directory = args.plan.parent
    lock = (directory / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        path = directory / 'status.tmp'
        path.write_text(json.dumps({'stage': stage, 'time': time.time(), **extra}))
        path.replace(directory / 'status.json')
    try:
        while True:
            missing = missing_inputs(plan)
            if not missing:
                break
            status('waiting_inputs', missing=len(missing))
            time.sleep(30)
        env = {**os.environ, **plan['environment']}
        status('preflight')
        subprocess.run(preflight_command(plan['command'], directory / 'preflight'),
                       cwd=plan['cwd'], env={**env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        while not gpu_idle(plan['gpu']):
            status('waiting_gpu', gpu=plan['gpu'])
            time.sleep(30)
        status('training', gpu=plan['gpu'])
        subprocess.run(plan['command'], cwd=plan['cwd'], env=env, check=True)
        status('complete')
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
