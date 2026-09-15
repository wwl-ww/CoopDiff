"""CPU handoff tests: no real process is signalled, no GPU is used."""

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from takeover_overnight import ROOT, process_info, supervise, verify_processes


def test_proc_identity_matches_current_process():
    info = process_info(os.getpid())
    assert info['parent'] == os.getppid()
    assert info['session'] == os.getsid(0)
    assert info['ticks'] > 0


def test_invalid_parent_rejected_before_signals():
    with patch('takeover_overnight.os.kill') as kill:
        try:
            verify_processes(Path('/tmp/unrelated'), {'pid': 123}, {'parent': 456})
        except ValueError:
            pass
        else:
            raise AssertionError('unrelated process accepted')
        kill.assert_not_called()


def test_handoff_preserves_worker_and_continues_evaluations(tmp_path, foreground=False):
    (tmp_path / "train.log").write_text("train 10 0.5\n")
    plan = json.loads((ROOT / 'scripts/overnight_v2.json').read_text())
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    state = {'architecture': plan['architecture'], 'status': 'running', 'run_id': 'test',
             'gpu_uuid': 'GPU-3', 'stages': [
                 {'name': s['name'], 'status': 'running' if s['name'] == 'train' else 'passed'}
                 for s in plan['stages'][:5]]}
    (tmp_path / 'status.json').write_text(json.dumps(state))
    ticks = int(time.monotonic() * os.sysconf('SC_CLK_TCK'))
    runner = {'pid': 1111, 'parent': 1, 'session': 1111, 'ticks': ticks,
              'command': [sys.executable, '-u', str(ROOT / 'scripts/run_overnight.py'),
                          '--out', str(tmp_path)]}
    worker = {'pid': 2222, 'parent': 1111, 'session': 2222, 'ticks': ticks,
              'command': [sys.executable, str(ROOT / 'scripts/v2_experiment.py'),
                          '--stage', 'train', '--out', str(tmp_path / 'train')]}
    checkpoint = str(tmp_path / 'train/best.pt')
    def report(path, architecture, stage, run_id, previous):
        return checkpoint if stage == 'train' else previous
    with patch('takeover_overnight.process_info', side_effect=lambda pid: runner if pid == 1111 else worker), \
            patch('takeover_overnight.same_process', side_effect=[True, False]), \
            patch('takeover_overnight.os.readlink', return_value=str(tmp_path / 'train.log')), \
            patch('takeover_overnight.os.kill') as kill, \
            patch('takeover_overnight.os.killpg') as killpg, \
            patch('takeover_overnight.gpu3_visibility', return_value='GPU-3'), \
            patch('takeover_overnight.validate_report', side_effect=report), \
            patch('takeover_overnight.run_command') as run:
        assert supervise(tmp_path, 1111, 2222, foreground=foreground) == 0
        assert kill.call_args_list == [((1111, signal.SIGSTOP),), ((1111, signal.SIGKILL),)]
        killpg.assert_not_called()
        assert run.call_count == 2
        for call in run.call_args_list:
            assert checkpoint in call.args[0]
            assert call.args[1]['CUDA_VISIBLE_DEVICES'] == 'GPU-3'
    result = json.loads((tmp_path / 'status.json').read_text())
    assert result['status'] == 'complete'
    assert result['supervision']['training_restarted'] is False
    assert result['supervision']['mode'] == ('foreground' if foreground else 'detached')


if __name__ == '__main__':
    test_proc_identity_matches_current_process()
    test_invalid_parent_rejected_before_signals()
    with tempfile.TemporaryDirectory(prefix='takeover-test-') as directory:
        test_handoff_preserves_worker_and_continues_evaluations(Path(directory))
    with tempfile.TemporaryDirectory(prefix='foreground-test-') as directory:
        test_handoff_preserves_worker_and_continues_evaluations(Path(directory), foreground=True)
    print('Overnight handoff checks passed')
