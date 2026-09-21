"""Deferred data-scale jobs cannot skip data or GPU readiness."""
from pathlib import Path
import sys
from unittest.mock import patch
from subprocess import CompletedProcess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_rank_data_scale_queue import gpu_idle, missing_inputs, preflight_command


def test_queue_readiness_and_preflight(tmp_path):
    existing = tmp_path / 'label.json'; existing.write_text('{}')
    missing = tmp_path / 'absent.json'
    assert missing_inputs({'required_paths': [str(existing), str(missing)]}) == [str(missing)]
    command = ['python', 'train.py', '--out', 'training', '--epochs', '20']
    assert preflight_command(command, 'audit') == ['python', 'train.py', '--out', 'audit', '--epochs', '20', '--audit-only']
    assert command[3] == 'training'
    for code, stdout, expected in [(0, '', True), (0, '123\n', False), (1, '', False)]:
        with patch('scripts.run_rank_data_scale_queue.subprocess.run', return_value=CompletedProcess([], code, stdout, '')):
            assert gpu_idle(0) == expected
