"""Strict paired training uses one writer and a read-only cache follower."""

from argparse import Namespace
import importlib.util
from pathlib import Path
import tempfile

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


if __name__ == '__main__':
    test_read_only_featurization_follower_does_not_generate_missing_case()
