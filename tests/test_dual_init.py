"""CPU tests for cached mixed-initialization contracts."""

import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.initialization.endpoints import (  # noqa: E402
    CachedEndpointStore, DUAL_INIT_CACHE_VERSION, validate_endpoint_cache,
)
from tricoopdiff.initialization.endpoints import (  # noqa: E402
    LazyTrainingEndpointStore, RESAMPLED_TRAINING_ENDPOINT_PROTOCOL,
    endpoint_input_signature,
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
        record['coupling']['angles'][0] = 88
        record['actual_source'] = 'sdpi'
        third, fresh_record = store.coupling('demo', occurrence=4)
        assert third['angles'][0] == 0 and fresh_record['actual_source'] == 'bitseed', (
            'returned metadata must not expose mutable cached training state')


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
            'version': RESAMPLED_TRAINING_ENDPOINT_PROTOCOL,
            'conformers': 1, 'poses_per_side': 1, 'binary_steps': 20,
            'binary_retry_limit': 1, 'selection': 'single_pair_no_ranking',
            'fallback': 'sdpi_v1_on_any_invalid_expert_side',
            'base_seed': 41000,
            'expert_runtime': store.protocol['expert_runtime'],
            'generation_version': 'raw_target_projection_v3',
            'frozen_expert': {
                'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc',
                'no_torsion': False,
            },
        }


def _input_batch():
    return {
        'e3_data': {'coords': torch.zeros(1, 3, 3), 'features': torch.ones(1, 3, 2)},
        'poi_data': {'coords': torch.zeros(1, 5, 3)},
        'ligand_data': {
            'coords': torch.arange(12).float().reshape(1, 4, 3),
            'mol': b'ordered chemistry', 'edge_index': torch.tensor([[1], [2]]),
            'mask_rotate': np.array([[False, False, True, True]]),
        },
    }


def _lazy_store(directory):
    expert = type('Expert', (), {'provenance': {
        'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc', 'no_torsion': False,
    }})()
    return LazyTrainingEndpointStore(directory, expert)


def test_resampled_endpoints_use_occurrence_and_keep_validation_fixed():
    from unittest.mock import patch
    from tricoopdiff.initialization.endpoints import RESAMPLED_TRAINING_ENDPOINT_PROTOCOL

    with tempfile.TemporaryDirectory() as directory:
        expert = _lazy_store(directory).expert
        store = LazyTrainingEndpointStore(directory, expert)
        assert store.protocol['version'] == RESAMPLED_TRAINING_ENDPOINT_PROTOCOL

        def generate(case, batch, signature, occurrence):
            record = deepcopy(payload()['entries'][0])
            record['endpoint']['ligand'] += occurrence
            record['target_ligand'] = np.zeros((4, 3))
            return record

        with patch.object(store, '_generate', side_effect=generate) as generation:
            a, _ = store.coupling('demo', occurrence=0, batch=_input_batch())
            repeated, _ = store.coupling('demo', occurrence=0, batch=_input_batch())
            b, _ = store.coupling('demo', occurrence=1, batch=_input_batch())
            validation, _ = store.coupling('demo', occurrence=0, batch=_input_batch())
        assert generation.call_count == 3
        assert np.array_equal(a['initial_state']['ligand'], repeated['initial_state']['ligand'])
        assert not np.array_equal(a['initial_state']['ligand'], b['initial_state']['ligand'])
        assert np.array_equal(a['initial_state']['ligand'], validation['initial_state']['ligand'])
        assert store._path('demo', 0) != store._path('demo', 1)
        assert len(store._records) == 1, 'resampling must not accumulate every occurrence in RAM'


def test_cache_contract_includes_and_rechecks_actual_expert_policy():
    from types import SimpleNamespace
    from tricoopdiff.models.expert import configure_expert_policy

    with tempfile.TemporaryDirectory() as directory:
        expert = _lazy_store(directory).expert
        expert.model = SimpleNamespace()
        configure_expert_policy(expert, 100., 'irrep_norm')
        a = LazyTrainingEndpointStore(directory, expert)
        configure_expert_policy(expert, 1., 'componentwise')
        b = LazyTrainingEndpointStore(directory, expert)
        assert a.protocol != b.protocol
        try:
            a.coupling('demo', batch=_input_batch())
        except ValueError as error:
            assert 'runtime changed' in str(error)
        else:
            raise AssertionError('a cache must not outlive a change in expert policy')


def test_endpoint_signature_tracks_geometry_chemistry_and_atom_order():
    batch = _input_batch()
    signature = endpoint_input_signature(batch)
    assert endpoint_input_signature(deepcopy(batch)) == signature
    for component, key, replacement in (
        ('ligand_data', 'coords', torch.zeros(1, 6, 3)),
        ('ligand_data', 'coords', batch['ligand_data']['coords'].flip(1)),
        ('ligand_data', 'mol', b'different chemistry'),
        ('ligand_data', 'edge_index', torch.tensor([[2], [1]])),
        ('ligand_data', 'mask_rotate', np.ones((1, 4), dtype=bool)),
        ('e3_data', 'coords', torch.ones(1, 3, 3)),
        ('e3_data', 'features', torch.zeros(1, 3, 2)),
        ('poi_data', 'coords', torch.ones(1, 5, 3)),
    ):
        changed = deepcopy(batch)
        changed[component][key] = replacement
        assert endpoint_input_signature(changed) != signature, (component, key)


def test_lazy_cache_checks_disk_and_memory_inputs_without_overwriting():
    with tempfile.TemporaryDirectory() as directory:
        store = _lazy_store(directory)
        batch = _input_batch()
        path = store._path('demo')
        record = deepcopy(payload()['entries'][0])
        record['occurrence'] = 0
        record['target_ligand'] = np.zeros((4, 3))
        torch.save({'case': 'demo', 'occurrence': 0, 'protocol': store.protocol,
                    'record': record, 'input_signature': endpoint_input_signature(batch)}, path)
        original = path.read_bytes()
        first, record = store.coupling('demo', batch=batch)
        first['initial_state']['ligand'][0, 0] = 99
        record['endpoint']['ligand'][0, 0] = 88
        record['target_ligand'][0, 0] = 77
        record['actual_source'] = 'sdpi'
        again, fresh_record = store.coupling('demo', batch=deepcopy(batch))
        assert again['initial_state']['ligand'][0, 0] == 0
        assert again['target_ligand'][0, 0] == 0 and fresh_record['actual_source'] == 'bitseed', (
            'returned metadata changed deterministic cached inputs, labels or source')
        changed = deepcopy(batch)
        changed['ligand_data']['coords'] = torch.ones(1, 6, 3)
        for reader in (store, _lazy_store(directory)):
            try:
                reader.coupling('demo', batch=changed)
            except ValueError as error:
                assert 'input signature' in str(error)
            else:
                raise AssertionError('changed input must not reuse the cached coupling')
        assert path.read_bytes() == original, 'rejection must leave old cache untouched'


def test_read_only_endpoint_follower_never_generates_missing_case():
    with tempfile.TemporaryDirectory() as directory:
        store = LazyTrainingEndpointStore(
            directory, _lazy_store(directory).expert, read_only=True, wait_seconds=0.)
        try:
            store.coupling('missing', batch=_input_batch())
        except TimeoutError as error:
            assert 'read-only endpoint follower' in str(error)
        else:
            raise AssertionError('read-only follower generated a missing endpoint')


def test_lazy_legacy_cache_and_missing_batch_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        store = _lazy_store(directory)
        path = store._path('demo')
        torch.save({'case': 'demo', 'protocol': store.protocol,
                    'record': payload()['entries'][0]}, path)
        for batch in (None, _input_batch()):
            try:
                store.coupling('demo', batch=batch)
            except ValueError as error:
                assert 'batch is required' in str(error) or 'signature missing' in str(error)
            else:
                raise AssertionError('unverified legacy cache must be rejected')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
