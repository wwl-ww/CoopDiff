"""Shared-layer compatibility and isolation from the historical FiLM model."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_shared_layer_imports_remain_compatible():
    from tricoopdiff import consensus_layers, geometry_encoders, trifuse
    from tricoopdiff.expert_time import _set_time_embeddings
    from tricoopdiff.model import _set_time_embeddings as legacy_time

    for module, names in (
        (geometry_encoders, ('LigandFeatureEncoder', 'BondContextEncoder',
                             'TernaryGeometryEncoder')),
        (consensus_layers, ('TokenNegotiator', 'JudgeHead', 'StateHead',
                           'TorsionCoordinator')),
    ):
        for name in names:
            assert getattr(trifuse, name) is getattr(module, name), name
    assert legacy_time is _set_time_embeddings


def test_consensus_import_does_not_load_legacy_model():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT)!r}); '
        'import tricoopdiff.consensus; '
        'assert "tricoopdiff.model" not in sys.modules; '
        'assert "tricoopdiff.trifuse" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    test_shared_layer_imports_remain_compatible()
    test_consensus_import_does_not_load_legacy_model()
    print('Module boundary checks passed')
