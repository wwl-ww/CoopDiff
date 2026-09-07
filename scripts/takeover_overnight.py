"""Detach supervision of an already-running v2 train stage without restarting it.

Linux-only, explicit PID handoff. Prior gates are revalidated before the old
supervisor is stopped. The training process already has its own session and
file-backed output. Only the verified old supervisor is killed, never its group.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_overnight import (ROOT, gpu3_visibility, run_command, validate_plan,
                           validate_report, write_json)


def process_info(pid):
    """Identity includes kernel start ticks, preventing accidental PID reuse."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return {'pid': pid, 'state': fields[0], 'parent': int(fields[1]),
                'session': int(fields[3]), 'ticks': int(fields[19]), 'command': command}
    except FileNotFoundError:
        return None


def same_process(identity):
    current = process_info(identity['pid'])
    return bool(current and current['ticks'] == identity['ticks'] and current['state'] != 'Z')


def verify_processes(output, runner, worker):
    if not runner or not worker or worker['parent'] != runner['pid']:
        raise ValueError('training worker is not a child of the specified supervisor')
    for info, script in ((runner, 'run_overnight.py'), (worker, 'v2_experiment.py')):
        command = info['command']
        if len(command) < 2 or Path(command[1]).resolve() != ROOT / 'scripts' / script:
            raise ValueError('unexpected command identity')
        expected_output = output if info is runner else output / 'train'
        if '--out' not in command or Path(command[command.index('--out') + 1]).resolve() != expected_output:
            raise ValueError('process belongs to a different run')
    command = worker['command']
    if '--stage' not in command or command[command.index('--stage') + 1] != 'train':
        raise ValueError('only the active train stage can be transferred')
    if worker['session'] != worker['pid'] or worker['session'] == runner['session']:
        raise ValueError('training must already have an independent session')
    for fd in (1, 2):
        if Path(os.readlink(f'/proc/{worker["pid"]}/fd/{fd}')) != output / 'train.log':
            raise ValueError('training output is not independently file-backed')


def terminate_worker(identity):
    if not same_process(identity):
        return
    os.killpg(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while same_process(identity) and time.monotonic() < deadline:
        time.sleep(.1)
    if same_process(identity):
        os.killpg(identity['pid'], signal.SIGKILL)


def supervise(output, runner_pid, worker_pid):
    plan = json.loads((output / 'plan.json').read_text())
    total = validate_plan(plan)
    state = json.loads((output / 'status.json').read_text())
    if state['architecture'] != 'residual_consensus_v2' or state['status'] != 'running':
        raise ValueError('expected a running single-state v2 experiment')
    names = [s['name'] for s in plan['stages']]
    index = names.index('train')
    if len(state['stages']) != index + 1 or state['stages'][-1] != {'name': 'train', 'status': 'running'}:
        raise ValueError('supervisor is no longer at the expected train stage')
    checkpoint = None
    for stage in state['stages'][:-1]:
        if stage['status'] != 'passed':
            raise ValueError('an earlier gate did not pass')
        checkpoint = validate_report(output / (stage['name'] + '.json'), state['architecture'],
                                     stage['name'], state['run_id'], checkpoint)
    gpu = gpu3_visibility(direct=True)
    if state['gpu_uuid'] != gpu:
        raise ValueError('physical GPU identity changed')
    runner, worker = process_info(runner_pid), process_info(worker_pid)
    verify_processes(output, runner, worker)
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    run_start = runner['ticks'] / ticks_per_second
    worker_start = worker['ticks'] / ticks_per_second
    deadline = min(worker_start + plan['stages'][index]['seconds'],
                   run_start + total - sum(s['seconds'] for s in plan['stages'][index + 1:]))
    if deadline <= time.monotonic():
        raise ValueError('training budget already expired')

    # Freeze to eliminate a race with the old supervisor starting an evaluation.
    os.kill(runner_pid, signal.SIGSTOP)
    try:
        verify_processes(output, process_info(runner_pid), process_info(worker_pid))
        if not same_process(worker):
            raise ValueError('worker finished during handoff; retry at another boundary')
        # SIGKILL deliberately bypasses its finally block, which would kill the
        # independently running training group. Send to ONE verified PID only.
        os.kill(runner_pid, signal.SIGKILL)
    except BaseException:
        if same_process(runner):
            os.kill(runner_pid, signal.SIGCONT)
        raise
    state['supervision'] = {'mode': 'detached', 'pid': os.getpid(),
                            'replaced_pid': runner_pid, 'training_pid': worker_pid,
                            'training_restarted': False, 'log': str(output / 'supervisor.log')}
    write_json(output / 'status.json', state)
    write_json(output / 'supervisor.json', state['supervision'])
    print(f'ATTACHED: supervisor={os.getpid()}, existing training={worker_pid}', flush=True)
    try:
        while same_process(worker):
            if time.monotonic() >= deadline:
                terminate_worker(worker)
                raise TimeoutError('original training-stage deadline exceeded')
            time.sleep(2)
        checkpoint = validate_report(output / 'train.json', state['architecture'],
                                     'train', state['run_id'], checkpoint)
        state['stages'][-1].update(status='passed', elapsed_seconds=time.monotonic() - worker_start)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                   HF_HOME='/tmp/pymp-f2c2db1a/results/huggingface',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   TRICOOPDIFF_CACHE_DIR='/tmp/pymp-f2c2db1a/results/runtime_cache/diffdock')
        for stage in plan['stages'][index + 1:]:
            name = stage['name']
            record = {'name': name, 'status': 'running'}
            state['stages'].append(record)
            write_json(output / 'status.json', state)
            (output / name).mkdir()
            values = {'python': sys.executable, 'root': str(ROOT), 'out': str(output / name),
                      'report': str(output / (name + '.json')), 'run_id': state['run_id'],
                      'checkpoint': checkpoint, 'seconds': str(stage['seconds'] - 30)}
            argv = list(stage['argv'])
            for key, value in values.items():
                argv = [arg.replace('{' + key + '}', value) for arg in argv]
            stage_env = dict(env, EXPERIMENT_RUN_ID=state['run_id'], EXPERIMENT_STAGE=name,
                             EXPERIMENT_REPORT=values['report'], EXPERIMENT_ARCHITECTURE=state['architecture'])
            seconds = min(stage['seconds'], run_start + total - time.monotonic())
            if seconds <= 0:
                raise TimeoutError('total experiment budget exhausted')
            print(f'[{name}] starting', flush=True)
            tick = time.monotonic()
            run_command(argv, stage_env, output / (name + '.log'), seconds)
            checkpoint = validate_report(Path(values['report']), state['architecture'], name,
                                         state['run_id'], checkpoint)
            record.update(status='passed', elapsed_seconds=time.monotonic() - tick, argv=argv)
        state['status'] = 'complete'
        return 0
    except (Exception, KeyboardInterrupt) as error:
        terminate_worker(worker)
        state.update(status='failed', error=str(error) or type(error).__name__)
        state['stages'][-1]['status'] = 'failed'
        print(f'STOP: {state["error"]}', flush=True)
        return 1
    finally:
        state['checkpoint'] = checkpoint
        state['elapsed_seconds'] = time.monotonic() - run_start
        write_json(output / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runner-pid', type=int, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (output / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise(output, args.runner_pid, args.worker_pid)


if __name__ == '__main__':
    sys.exit(main())
