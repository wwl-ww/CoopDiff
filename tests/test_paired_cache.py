"""Strict paired training uses one writer and a read-only cache follower."""

from argparse import Namespace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'featurize_common', ROOT / 'scripts/featurize_common.py')
featurize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(featurize)


def test_read_only_featurization_follower_does_not_generate_missing_case():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = []
        for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
            path = root / name; path.write_text(name); inputs.append(path.resolve())
        cache = root / 'cache'; cache.mkdir()
        try:
            featurize.featurize_complex(
                'demo', *inputs, Namespace(receptor_radius=30), 'cpu',
                cache_dir=cache, cache_read_only=True, cache_wait_seconds=0.)
        except TimeoutError as error:
            assert 'read-only featurization follower' in str(error)
        else:
            raise AssertionError('read-only follower attempted cold featurization')
        assert not list(cache.iterdir())


def test_failed_featurization_publish_preserves_cache_and_removes_partial_file(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    cache = tmp_path / 'cache'
    cache.mkdir()
    destination = cache / 'demo.pt'
    torch.save({'cache_version': -1}, destination)
    original = destination.read_bytes()
    data = {'coords': torch.zeros(1, 3, 3), 'features': torch.ones(1, 3, 2)}

    def fail_save(payload, path):
        Path(path).write_bytes(b'partial cache')
        raise OSError('intentional interrupted write')

    with patch.object(featurize, 'generate_esm_embeddings', return_value={path: [] for path in inputs}), \
            patch.object(featurize, 'featurize_subgraph', return_value=SimpleNamespace(success=[True])), \
            patch.object(featurize, 'extract_receptor', side_effect=lambda *args: dict(data)), \
            patch.object(featurize, 'extract_ligand', return_value=dict(data)), \
            patch.object(featurize, '_true_relative_offsets', return_value=(np.zeros(3), np.zeros(3))), \
            patch('torch.save', side_effect=fail_save):
        try:
            featurize.featurize_complex('demo', *inputs, Namespace(remove_hs=True), 'cpu', cache_dir=cache)
        except OSError as error:
            assert 'intentional' in str(error)
        else:
            raise AssertionError('cache write failure was hidden')
    assert destination.read_bytes() == original, 'failed write replaced the original cache'
    assert list(cache.iterdir()) == [destination], 'failed write left a partial cache behind'


if __name__ == '__main__':
    test_read_only_featurization_follower_does_not_generate_missing_case()
