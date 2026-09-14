"""CPU tests for cached mixed-initialization contracts."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.dual_init import (  # noqa: E402
    CachedEndpointStore, DUAL_INIT_CACHE_VERSION, validate_endpoint_cache,
)
from tricoopdiff.online_endpoints import (  # noqa: E402
    LazyTrainingEndpointStore, ONLINE_TRAINING_ENDPOINT_PROTOCOL,
)


def payload(source='bitseed', fallback=None):
    record = {
        'case': 'demo', 'actual_source': source,
        'candidate_id': ['bitseed', 0, 1, 2], 'conformer_seed': 17,
        'binary_pose_ids': [1, 2], 'fallback': fallback,
        'native_used_in_candidate_generation': False,
        'native_used_in_candidate_selection': False,
        'native_used_for_training_projection': True,
        'endpoint': {
            'ligand': np.zeros((4, 3)), 'poi': np.zeros((5, 3)),
            'body_frame': np.eye(3),
        },
        'training_endpoint': {
            'ligand': np.zeros((4, 3)), 'poi': np.zeros((5, 3)),
            'body_frame': np.eye(3),
        },
        'coupling': {
            'angles': np.zeros(1), 'omegas': np.zeros((2, 3)),
            'centers': np.zeros((2, 3)), 'bonds': np.array([[1, 2]]),
            'masks': np.array([[False, False, True, True]]),
        },
        'projection': {'contract': 'torsion_rigid_endpoint_projection_v1'},
    }
    return {
        'cache_version': DUAL_INIT_CACHE_VERSION,
        'generation_protocol': {'frozen_expert_checkpoint_sha256': 'abc123'},
        'entries': [record],
    }


def test_cache_roundtrip_preserves_deterministic_coupling():
    value = payload()
    validate_endpoint_cache(value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'endpoints.pt'
        torch.save(value, path)
        store = CachedEndpointStore(path)
        first, record = store.coupling('demo', occurrence=4)
        first['angles'][0] = 99
        second, _ = store.coupling('demo', occurrence=4)
        assert second['angles'][0] == 0
        assert record['actual_source'] == 'bitseed'


def test_fallback_cannot_be_relabelled_as_bitseed():
    value = payload(fallback='sdpi_v1')
    try:
        validate_endpoint_cache(value)
    except ValueError as error:
        assert 'fallback' in str(error)
    else:
        raise AssertionError('fallback endpoint must retain its actual SDPI source')


def test_ood_expert_failure_is_a_valid_sdpi_fallback_record():
    value = payload(source='sdpi', fallback='sdpi_v1')
    validate_endpoint_cache(value)


def test_lazy_training_protocol_is_exactly_one_pose_per_expert():
    expert = type('Expert', (), {'provenance': {
        'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc',
        'no_torsion': False,
    }})()
    with tempfile.TemporaryDirectory() as directory:
        store = LazyTrainingEndpointStore(directory, expert)
        assert store.protocol == {
            'version': ONLINE_TRAINING_ENDPOINT_PROTOCOL,
            'conformers': 1, 'poses_per_side': 1, 'binary_steps': 20,
            'binary_retry_limit': 1, 'selection': 'single_pair_no_ranking',
            'fallback': 'sdpi_v1_on_any_invalid_expert_side',
            'base_seed': 41000,
            'frozen_expert': {
                'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc',
                'no_torsion': False,
            },
        }


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
