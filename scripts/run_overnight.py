"""Sequential, fail-closed experiment runner; never substitutes missing experiments.

Commands run without a shell. Each must write its own scientific gate report.
The supplied dual-consensus manifest is intentionally blocked until its actual
experiment entry points exist. This runner does not implement a model or fix bugs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'contracts': ('state_isolation', 'frame_equivariance', 'chemical_geometry',
                  'poi_update', 'reachable_target'),
    'binary_preservation': ('matched_initialization', 'matched_solver', 'binary_parity'),
    'overfit': ('loss_decreased', 'poi_rotation_learned', 'finite_gradients',
                'expert_unchanged', 'checkpoint_roundtrip'),
    'pilot': ('heldout_validation', 'finite_trajectory', 'throughput_measured',
              'physical_collapse'),
    'train': ('finite_training', 'checkpoint_saved', 'validation_recorded'),
    'evaluate_protac': ('fresh_checkpoint_load', 'local_metrics', 'global_metrics',
                        'failures_counted'),
    'evaluate_mgd': ('fresh_checkpoint_load', 'local_metrics', 'global_metrics',
                     'failures_counted'),
}

V2_STAGES = {
    'regression': ('regression_passed', 'inputs_present', 'split_disjoint'),
    'smoke': ('expert_unchanged', 'all_trainable_gradients_present',
              'checkpoint_ema_roundtrip_exact', 'finite_trajectory'),
    'overfit': ('loss_decreased', 'poi_rotation_learned', 'finite_gradients'),
    'pilot': ('heldout_validation', 'throughput_measured', 'checkpoint_saved'),
    'train': ('finite_training', 'checkpoint_saved', 'validation_recorded'),
    'evaluate_protac': STAGES['evaluate_protac'],
    'evaluate_mgd': STAGES['evaluate_mgd'],
}


def stage_contract(architecture):
    return V2_STAGES if architecture == 'residual_consensus_v2' else STAGES


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_plan(plan):
    if not isinstance(plan.get('architecture'), str) or not plan['architecture']:
        raise ValueError('architecture is required')
    stages = plan.get('stages', [])
    expected = stage_contract(plan['architecture'])
    if [s.get('name') for s in stages] != list(expected):
        raise ValueError(f'stages must be ordered exactly as {list(expected)}')
    missing = []
    total = 0
    for stage in stages:
        name, command = stage['name'], stage.get('argv')
        seconds = stage.get('seconds')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f'{name}: seconds must be finite and positive')
        total += seconds
        if not command:
            missing.append(name)
        elif not isinstance(command, list) or not all(isinstance(v, str) and v for v in command):
            raise ValueError(f'{name}: argv must be a nonempty string array')
    if total > 36000:
        raise ValueError('stage budgets exceed the ten-hour limit')
    if missing:
        raise ValueError('unimplemented experiment commands: ' + ', '.join(missing))
    return total


def validate_report(path, architecture, stage, run_id, checkpoint):
    def reject_constant(value):
        raise ValueError(f'non-finite JSON number: {value}')

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        return number

    report = json.loads(path.read_text(), parse_constant=reject_constant,
                        parse_float=finite_float)
    for key, expected in (('architecture', architecture), ('stage', stage), ('run_id', run_id)):
        if report.get(key) != expected:
            raise ValueError(f'{stage}: report {key} mismatch')
    if report.get('passed') is not True:
        raise ValueError(f'{stage}: scientific gate failed')
    checks = report.get('checks', {})
    for name in stage_contract(architecture)[stage]:
        if checks.get(name) is not True:
            raise ValueError(f'{stage}: missing or failed check {name}')
    evidence = report.get('artifacts')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f'{stage}: evidence artifacts required')
    for item in evidence:
        artifact = Path(item)
        if not artifact.is_absolute() or not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f'{stage}: missing/empty evidence {item}')
    if stage == 'train':
        candidate = Path(report.get('checkpoint', ''))
        if not candidate.is_absolute() or not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError('train: missing nonempty checkpoint')
        checkpoint = str(candidate.resolve())
    elif stage.startswith('evaluate_'):
        if checkpoint is None or report.get('checkpoint') != checkpoint:
            raise ValueError(f'{stage}: evaluation used a different checkpoint')
    return checkpoint


def stop_group(process):
    """Terminate only our child's process group, including surviving workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_command(argv, env, log, seconds):
    with log.open('x') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
            if code:
                raise RuntimeError(f'command exited {code}; see {log}')
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'stage timed out after {seconds:.1f}s; see {log}') from error
        finally:
            stop_group(process)


def gpu3_visibility(direct=False):
    """Resolve physical GPU 3 by UUID; direct mode requires explicit CLI opt-in."""
    if not direct and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('run inside SLURM or explicitly select --direct-gpu3')
    probe = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    matches = [line.split(',')[1].strip() for line in probe.splitlines()
               if line.split(',')[0].strip() == '3']
    if len(matches) != 1:
        raise ValueError('cannot identify physical GPU 3')
    if not direct:
        allocation = os.environ.get('SLURM_JOB_GPUS', '').split(',')
        if '3' not in allocation and matches[0] not in allocation:
            raise ValueError('SLURM allocation must include physical GPU 3')
    return matches[0]


def run_plan(plan, output, python, require_allocation=True, direct_gpu3=False):
    total = validate_plan(plan)
    if require_allocation or direct_gpu3:
        visible_gpu = gpu3_visibility(direct=direct_gpu3)
    else:
        visible_gpu = ''  # CPU-only test hook; never exposed as a CLI bypass.
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    state = {'run_id': run_id, 'architecture': plan['architecture'],
             'status': 'running', 'stages': [], 'budget_seconds': total,
             'gpu_uuid': visible_gpu, 'direct_gpu3': direct_gpu3}
    state['git_commit'] = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    state['git_status'] = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    write_json(output / 'plan.json', plan)
    started = time.monotonic()
    deadline = started + total
    checkpoint = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_gpu, HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               HF_HOME=os.environ.get('HF_HOME', str(ROOT / 'results/huggingface')),
               CUBLAS_WORKSPACE_CONFIG=':4096:8',
               TRICOOPDIFF_CACHE_DIR=os.environ.get(
                   'TRICOOPDIFF_CACHE_DIR', str(ROOT / 'results/runtime_cache/diffdock')))
    try:
        for i, stage in enumerate(plan['stages']):
            name = stage['name']
            record = {'name': name, 'status': 'running'}
            state['stages'].append(record)
            write_json(output / 'status.json', state)
            report = output / f'{name}.json'
            stage_output = output / name
            stage_output.mkdir()
            remaining_reserve = sum(s['seconds'] for s in plan['stages'][i + 1:])
            seconds = min(stage['seconds'], deadline - time.monotonic() - remaining_reserve)
            if seconds <= 0:
                raise RuntimeError(f'{name}: no time left after reserving later stages')
            values = {'python': python, 'root': str(ROOT), 'out': str(stage_output),
                      'report': str(report), 'run_id': run_id,
                      'checkpoint': checkpoint or '', 'seconds': str(max(1, int(seconds - 30)))}
            argv = []
            for argument in stage['argv']:
                for key, value in values.items():
                    argument = argument.replace('{' + key + '}', value)
                argv.append(argument)
            record['argv'] = argv
            stage_env = dict(env, EXPERIMENT_RUN_ID=run_id, EXPERIMENT_STAGE=name,
                             EXPERIMENT_REPORT=str(report),
                             EXPERIMENT_ARCHITECTURE=plan['architecture'])
            print(f'[{name}] starting, timeout={seconds:.0f}s', flush=True)
            tick = time.monotonic()
            run_command(argv, stage_env, output / f'{name}.log', seconds)
            checkpoint = validate_report(report, plan['architecture'], name, run_id, checkpoint)
            record.update(status='passed', elapsed_seconds=time.monotonic() - tick)
            write_json(output / 'status.json', state)
        state['status'] = 'complete'
        return 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='failed', error=str(error) or type(error).__name__)
        if state['stages']:
            state['stages'][-1]['status'] = 'failed'
        print(f'STOP: {state["error"]}', file=sys.stderr, flush=True)
        return 1
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        state['checkpoint'] = checkpoint
        write_json(output / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'scripts/overnight_v2.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--direct-gpu3', action='store_true',
                        help='Explicitly authorize physical GPU 3 on this dedicated host without SLURM')
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if args.out is None:
            parser.error('--out must name a new directory')
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        return run_plan(plan, args.out.resolve(), args.python, direct_gpu3=args.direct_gpu3)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'PREFLIGHT FAILED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
