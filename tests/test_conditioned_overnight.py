"""Exercise serial queue success, failure and wall-clock shutdown on CPU."""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_conditioned_overnight import run_task


def task(root, code, seconds=10):
    state = {'tasks': []}
    environment = dict(os.environ, QUEUE_RUNTIME=str(root))
    result = run_task([sys.executable, '-c', code], 'fixture', root,
                      environment, state, time.time() + seconds)
    return result, state


def test_queue_task_success(tmp_path):
    result, state = task(tmp_path, 'print("done")')
    assert result and state['stage'] == 'between_tasks'
    assert state['tasks'][0]['returncode'] == 0
    assert (tmp_path / 'fixture.log').read_text().strip() == 'done'


def test_queue_task_failure_stops_chain(tmp_path):
    try:
        task(tmp_path, 'raise SystemExit(3)')
    except RuntimeError:
        state = json.loads((tmp_path / 'status.json').read_text())
        assert state['stage'] == 'failed'
        assert state['tasks'][0]['returncode'] == 3
    else:
        raise AssertionError('failed task must not be followed by another task')


def test_queue_deadline_stops_child(tmp_path):
    result, state = task(tmp_path, 'import time; time.sleep(10)', seconds=.2)
    assert not result and state['stage'] == 'deadline_reached'
    record = state['tasks'][0]
    assert record['deadline_interrupted']
    assert record['returncode'] != 0
    try:
        os.kill(record['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('deadline left a child running')
