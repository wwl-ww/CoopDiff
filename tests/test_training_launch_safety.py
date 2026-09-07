"""Protect experiments and epoch-boundary stochastic continuation."""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_entry import check_output_directory
from check_dual_training_data import inspect
from tricoopdiff.reproducibility import capture_rng_state, restore_rng_state


def test_output_protection(tmp_path):
    check_output_directory(tmp_path)
    checkpoint = tmp_path / 'best_model.pt'
    checkpoint.touch()
    with pytest.raises(ValueError, match='nonempty'):
        check_output_directory(tmp_path)
    with pytest.raises(ValueError, match='does not exist'):
        check_output_directory(tmp_path, tmp_path / 'missing.pt')
    check_output_directory(tmp_path, checkpoint)
    other = tmp_path / 'another'
    other.mkdir()
    (other / 'config.json').touch()
    with pytest.raises(ValueError, match='nonempty'):
        check_output_directory(other, checkpoint)


def test_rng_roundtrip_weights_only():
    before = capture_rng_state()
    try:
        payload = io.BytesIO()
        torch.save(before, payload)
        expected = (random.random(), np.random.rand(4), torch.rand(4))
        payload.seek(0)
        restore_rng_state(torch.load(payload, weights_only=True))
        assert random.random() == expected[0]
        assert np.array_equal(np.random.rand(4), expected[1])
        assert torch.equal(torch.rand(4), expected[2])
    finally:
        restore_rng_state(before)


def test_empty_training_indexes_cannot_pass(tmp_path):
    paths = [tmp_path / name for name in ('train.json', 'val.json')]
    for path in paths:
        path.write_text(json.dumps({'entries': [], 'clusters': []}))
    report = inspect(*paths)
    assert not report['data_pass']
    assert not report['model_training_ready']
    assert any('empty' in error for error in report['errors'])
