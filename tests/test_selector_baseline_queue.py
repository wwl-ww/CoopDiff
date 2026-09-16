"""The frozen random-baseline queue must not leak caller-relative paths."""

from pathlib import Path
import importlib.util
import json
import os
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'run_selector_baseline_queue', ROOT / 'scripts/run_selector_baseline_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def test_queue_resolves_paths_before_frozen_runtime_subprocess(tmp_path):
    selector = tmp_path / 'selector'; selector.mkdir()
    (selector / 'status.json').write_text(json.dumps({'stage': 'complete'}))
    original = Path.cwd()
    os.chdir(tmp_path)
    captured = {}
    try:
        def run(command, **kwargs):
            captured['command'] = command
            return type('Result', (), {'returncode': 0})()
        with patch.object(queue.subprocess, 'run', side_effect=run):
            queue.main(['--selector-run', 'selector', '--out', 'output'])
    finally:
        os.chdir(original)
    command = captured['command']
    labels = Path(command[command.index('--labels') + 1])
    destination = Path(command[command.index('--out') + 1])
    assert labels.is_absolute() and labels == selector / 'val8_labels'
    assert destination.is_absolute() and destination == tmp_path / 'output/random_baselines'
    assert json.loads((tmp_path / 'output/status.json').read_text())['stage'] == 'complete'


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test_queue_resolves_paths_before_frozen_runtime_subprocess(Path(directory))
