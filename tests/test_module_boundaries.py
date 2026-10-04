"""Shared-layer compatibility and isolation from the historical FiLM model."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_shared_layer_imports_remain_compatible():
    from tricoopdiff.models import consensus
    from tricoopdiff.legacy import trifuse
    from tricoopdiff.models.expert import _set_time_embeddings
    from tricoopdiff.legacy.model import _set_time_embeddings as legacy_time

    for module, names in (
        (consensus, ('LigandFeatureEncoder', 'BondContextEncoder',
                             'TernaryGeometryEncoder')),
        (consensus, ('TokenNegotiator', 'JudgeHead', 'StateHead',
                           'TorsionCoordinator')),
    ):
        for name in names:
            assert getattr(trifuse, name) is getattr(module, name), name
    assert legacy_time is _set_time_embeddings


def test_consensus_import_does_not_load_legacy_model():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT)!r}); '
        'import tricoopdiff.models.consensus; '
        'assert "tricoopdiff.legacy.model" not in sys.modules; '
        'assert "tricoopdiff.legacy.diffusion" not in sys.modules; '
        'assert "tricoopdiff.legacy.trifuse" not in sys.modules; '
        'assert not any(name.startswith(("tricoopdiff.experiments", '
        '"tricoopdiff.ranking", "tricoopdiff.conditioning")) for name in sys.modules)'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_flow_import_is_independent_of_model_and_legacy_diffusion():
    code = (
        'import sys; '
        f'sys.path.insert(0, {str(ROOT)!r}); '
        'from tricoopdiff import SDPIFlow; '
        'assert SDPIFlow().inference_steps == 20; '
        'assert "tricoopdiff.models.consensus" not in sys.modules; '
        'assert "tricoopdiff.legacy.diffusion" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    test_shared_layer_imports_remain_compatible()
    test_consensus_import_does_not_load_legacy_model()
    test_flow_import_is_independent_of_model_and_legacy_diffusion()
    print('Module boundary checks passed')
