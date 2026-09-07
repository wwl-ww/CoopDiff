"""CPU-only orchestration checks; fake commands never train or access a GPU."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_overnight import STAGES, gpu3_visibility, run_command, run_plan, validate_plan, validate_report


CHILD = '''
import json, os, pathlib, sys
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
stage = os.environ['EXPERIMENT_STAGE']
report = pathlib.Path(os.environ['EXPERIMENT_REPORT'])
artifact = report.parent / stage / 'evidence.json'
artifact.write_text('{"measured": true}')
checkpoint = sys.argv[1]
if stage == 'train':
    target = artifact.parent / 'checkpoint.pt'
    target.write_text('fake test checkpoint')
    checkpoint = str(target.resolve())
payload = dict(run_id=os.environ['EXPERIMENT_RUN_ID'], stage=stage,
               architecture=os.environ['EXPERIMENT_ARCHITECTURE'], passed=True,
               checks={name: True for name in sys.argv[2:]},
               artifacts=[str(artifact)], checkpoint=checkpoint)
report.write_text(json.dumps(payload))
'''


def plan():
    return {'architecture': 'test_only', 'stages': [
        {'name': name, 'seconds': 5,
         'argv': ['{python}', '-c', CHILD, '{checkpoint}', *checks]}
        for name, checks in STAGES.items()
    ]}


def test_missing_implementation_blocks_everything(tmp_path):
    manifest = plan()
    manifest['stages'][3]['argv'] = None
    try:
        run_plan(manifest, tmp_path / 'run', sys.executable, require_allocation=False)
    except ValueError as error:
        assert 'pilot' in str(error)
    else:
        raise AssertionError('missing experiment must block startup')
    assert not (tmp_path / 'run').exists()


def test_success_and_checkpoint_handoff(tmp_path):
    output = tmp_path / 'run'
    assert run_plan(plan(), output, sys.executable, require_allocation=False) == 0
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'complete'
    assert [s['name'] for s in status['stages']] == list(STAGES)
    assert len(list(output.glob('*.log'))) == 7
    assert Path(status['checkpoint']).is_file()
    try:
        run_plan(plan(), output, sys.executable, require_allocation=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError('must not overwrite earlier evidence')


def test_failed_command_stops_before_training(tmp_path):
    manifest = plan()
    manifest['stages'][1]['argv'] = ['{python}', '-c', 'raise SystemExit(7)']
    output = tmp_path / 'run'
    assert run_plan(manifest, output, sys.executable, require_allocation=False) == 1
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert len(status['stages']) == 2
    assert not (output / 'train.log').exists()


def test_exit_zero_without_report_is_failure(tmp_path):
    manifest = plan()
    manifest['stages'][0]['argv'] = ['{python}', '-c', 'pass']
    assert run_plan(manifest, tmp_path / 'run', sys.executable, require_allocation=False) == 1


def test_stale_and_false_gates_are_rejected(tmp_path):
    output = tmp_path / 'run'
    assert run_plan(plan(), output, sys.executable, require_allocation=False) == 0
    report = output / 'contracts.json'
    payload = json.loads(report.read_text())
    for stale in (True, False):
        changed = copy.deepcopy(payload)
        if stale:
            changed['run_id'] = 'old-run'
        else:
            changed['checks']['reachable_target'] = False
        report.write_text(json.dumps(changed))
        try:
            validate_report(report, 'test_only', 'contracts', payload['run_id'], None)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid gate report accepted')


def test_invalid_budget_and_stage_order():
    for value in (0, -1, float('nan'), float('inf'), True, 36001):
        manifest = plan()
        manifest['stages'][0]['seconds'] = value
        try:
            validate_plan(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid time budget accepted: {value}')
    manifest = plan()
    manifest['stages'].reverse()
    try:
        validate_plan(manifest)
    except ValueError:
        pass
    else:
        raise AssertionError('reordered gates accepted')


def test_timeout_terminates_child(tmp_path):
    pid_path = tmp_path / 'pid'
    code = ('import os, pathlib, time; '
            f'pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); '
            'time.sleep(30)')
    try:
        run_command([sys.executable, '-c', code], dict(os.environ), tmp_path / 'log', 1)
    except RuntimeError as error:
        assert 'timed out' in str(error)
    else:
        raise AssertionError('timeout not enforced')
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('child survived timeout')


def test_gpu_allocation_is_required(tmp_path):
    with patch.dict(os.environ, {}, clear=True):
        try:
            run_plan(plan(), tmp_path / 'run', sys.executable)
        except ValueError as error:
            assert 'SLURM' in str(error)
        else:
            raise AssertionError('login-node execution allowed')
    with patch.dict(os.environ, {'SLURM_JOB_ID': 'test', 'SLURM_JOB_GPUS': '2'}, clear=True), \
            patch('run_overnight.subprocess.check_output', return_value='3, GPU-test\n'):
        try:
            run_plan(plan(), tmp_path / 'run', sys.executable)
        except ValueError as error:
            assert 'physical GPU 3' in str(error)
        else:
            raise AssertionError('unallocated GPU 3 allowed')
    assert not (tmp_path / 'run').exists()


def test_explicit_direct_mode_resolves_only_gpu3():
    with patch.dict(os.environ, {}, clear=True), \
            patch('run_overnight.subprocess.check_output',
                  return_value='0, GPU-other\n3, GPU-authorized\n'):
        assert gpu3_visibility(direct=True) == 'GPU-authorized'
    with patch('run_overnight.subprocess.check_output', return_value='0, GPU-other\n'):
        try:
            gpu3_visibility(direct=True)
        except ValueError as error:
            assert 'physical GPU 3' in str(error)
        else:
            raise AssertionError('must not fall back to another GPU')


def test_nonfinite_report_is_rejected(tmp_path):
    report = tmp_path / 'report.json'
    for invalid in ('NaN', 'Infinity', '1e400'):
        report.write_text('{"metric": ' + invalid + '}')
        try:
            validate_report(report, 'test', 'train', 'run', None)
        except ValueError as error:
            assert 'non-finite' in str(error)
        else:
            raise AssertionError('non-finite metric accepted')


def test_evaluation_cannot_switch_checkpoint(tmp_path):
    artifact = tmp_path / 'metric.json'
    artifact.write_text('{}')
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'architecture': 'test', 'stage': 'evaluate_mgd', 'run_id': 'run',
        'passed': True, 'checks': dict.fromkeys(STAGES['evaluate_mgd'], True),
        'artifacts': [str(artifact)], 'checkpoint': '/wrong/checkpoint.pt',
    }))
    try:
        validate_report(report, 'test', 'evaluate_mgd', 'run', '/expected/checkpoint.pt')
    except ValueError as error:
        assert 'different checkpoint' in str(error)
    else:
        raise AssertionError('wrong model evaluated')


if __name__ == '__main__':
    import inspect
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory(prefix='overnight-test-') as directory:
                kwargs = {'tmp_path': Path(directory)} if inspect.signature(test).parameters else {}
                test(**kwargs)
            print(name, 'passed')
