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
            patch.object(featurize, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)), \
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


def test_v8_cache_enrichment_preserves_embeddings_without_esm(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    args = Namespace(remove_hs=True)
    data = {'coords': torch.zeros(3, 3), 'features': torch.ones(3, 2)}
    payload = {'cache_version': 8,
               'source_signature': featurize._featurize_signature(*inputs, args, cache_version=None),
               'e3_data': dict(data), 'poi_data': dict(data), 'ligand_data': dict(data),
               'receptor_feat_dim': 2}
    torch.save(payload, tmp_path / 'demo.pt')
    with patch.object(featurize, 'generate_esm_embeddings', side_effect=AssertionError('ESM was rebuilt')), \
            patch.object(featurize, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)) as geometry:
        e3, poi, _, _ = featurize.featurize_complex('demo', *inputs, args, 'cpu', cache_dir=tmp_path)
        assert geometry.call_count == 2
        assert e3['local_geometry'].shape == poi['local_geometry'].shape == (1, 3, 32)
        again = featurize.featurize_complex('demo', *inputs, args, 'cpu', cache_dir=tmp_path)
        assert geometry.call_count == 2 and torch.equal(again[0]['features'], e3['features'])
    upgraded = torch.load(tmp_path / 'demo.pt', weights_only=False)
    assert upgraded['cache_version'] == 9
    assert upgraded['source_signature'] != payload['source_signature']
    assert upgraded['e3_data']['local_geometry'].shape == (3, 32)


def test_cache_only_never_rebuilds_missing_or_stale_features(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    with patch.object(featurize, 'generate_esm_embeddings', side_effect=AssertionError('ESM must not run')):
        for stale in (False, True):
            if stale:
                torch.save({'cache_version': -1}, tmp_path / 'demo.pt')
            try:
                featurize.featurize_complex('demo', *inputs, Namespace(remove_hs=True), 'cpu',
                                           cache_dir=tmp_path, cache_only=True)
            except FileNotFoundError as error:
                assert 'ESM rebuilding disabled' in str(error)
            else:
                raise AssertionError('missing/stale cache was accepted')


def test_v7_cache_only_corrects_center_without_rebuilding_esm(tmp_path):
    inputs = []
    for name in ('e3.pdb', 'poi.pdb', 'ligand.sdf'):
        path = tmp_path / name
        path.write_text(name)
        inputs.append(str(path.resolve()))
    args = Namespace(remove_hs=True)
    data = {'coords': torch.arange(9).float().reshape(3, 3), 'features': torch.ones(3, 2)}
    torch.save({'cache_version': 7,
        'source_signature': featurize._featurize_signature(*inputs, args, cache_version=None),
        'e3_data': dict(data), 'poi_data': dict(data), 'ligand_data': dict(data), 'receptor_feat_dim': 2}, tmp_path / 'demo.pt')
    with patch.object(featurize, 'generate_esm_embeddings', side_effect=AssertionError('ESM must not run')), \
            patch.object(featurize, 'protein_local_geometry', return_value=torch.ones(1, 3, 32)), \
            patch.object(featurize, '_ligand_centroid', return_value=np.array([4., 5., 6.])), \
            patch.object(featurize, '_protein_ca_centroid', return_value=np.array([1., 1., 1.])):
        e3, _, ligand, _ = featurize.featurize_complex('demo', *inputs, args, 'cpu',
                                                     cache_dir=tmp_path, cache_only=True)
    assert torch.equal(e3['features'][0], data['features'])
    assert torch.equal(ligand['coords'].mean(1)[0], torch.tensor([3., 4., 5.]))
    assert torch.equal(torch.cdist(ligand['coords'][0], ligand['coords'][0]), torch.cdist(data['coords'], data['coords']))


if __name__ == '__main__':
    test_read_only_featurization_follower_does_not_generate_missing_case()
