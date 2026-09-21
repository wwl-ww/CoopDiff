"""Three real source routes share optimizer/EMA and preserve explicit fallbacks."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_trainer import trainer, update
from test_initialization import ligand_input
from tricoopdiff.anchored_trainer import AnchoredTrainer, validate_anchored_checkpoint, build_anchored_model
from tricoopdiff.bianchor_source import install_bianchor_source
from tricoopdiff.mixed_source_training import prepare_free_batch, FORMAT, evaluate_free_sources
from tricoopdiff.sdpi_flow import SDPIFlow


def mixed_trainer():
    model = trainer().model
    install_bianchor_source(model)
    return AnchoredTrainer(model, parent_checkpoint_sha256='a' * 64,
                           training_index_sha256='b' * 64, three_source=True)


def fixture():
    ligand = ligand_input('CCCC')
    ligand.update(features=torch.randn(1, 4, 16), edge_attr=torch.ones(6, 1))
    proteins = []
    for offset in (-3., 3.):
        proteins.append(dict(coords=torch.randn(1, 5, 3) + offset,
                             features=torch.randn(1, 5, 16),
                             edge_index=torch.stack((torch.arange(5), torch.arange(5).roll(1)))))
    return dict(case='fixture', e3_data=proteins[0], poi_data=proteins[1], ligand_data=ligand)


def test_real_free_flows_update_distinct_source_rows_and_share_optimizer():
    value = mixed_trainer()
    batch = fixture()
    optimizer = value.optimizer
    for source in ('sdpi', 'bitseed'):
        coupling = SDPIFlow.sample_coupling(batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
        store = SimpleNamespace(coupling=lambda *a, **k: (coupling, {'actual_source': source}))
        prepared, audit = prepare_free_batch(batch, source, store, seed=7315)
        result = value.step_free(requested_source=source, **prepared, t_override=.5)
        assert result['loss_total'] >= 0 and audit['actual_source'] == source
        index = ('sdpi', 'bitseed').index(source)
        grad = value.model.trifuse.source_embedding.weight.grad
        assert grad[index].abs().sum() > 0
        assert grad[[i for i in range(3) if i != index]].count_nonzero() == 0
        assert value.optimizer is optimizer and not hasattr(value.free_objective, 'optimizer')
        assert all(p.grad is None for p in value.model.expert.parameters())
    update(value)
    assert value.source_training['updates'] == dict(sdpi=1, bitseed=1, bianchor=1)
    assert value.next_source() == 'sdpi'
    payload = value.checkpoint(epoch=0, next_case=3, seed=7315, scope='implementation_smoke')
    assert payload['format'] == FORMAT
    rebuilt = build_anchored_model(payload, value.model.expert, allow_smoke=True)
    assert rebuilt.trifuse.source_embedding.num_embeddings == 3
    restored = mixed_trainer()
    restored.restore(payload)
    assert restored.source_training == value.source_training
    assert restored.next_source() == value.next_source()
    for key, tensor in value.ema.items():
        assert torch.equal(restored.ema[key], tensor)
    prepared, _ = prepare_free_batch(batch, 'sdpi', None, seed=7315)
    rng = torch.get_rng_state().clone()
    value.step_free(requested_source='sdpi', **prepared, t_override=.5)
    torch.set_rng_state(rng)
    restored.step_free(requested_source='sdpi', **prepared, t_override=.5)
    for key, tensor in value.model.trifuse.state_dict().items():
        assert torch.equal(restored.model.trifuse.state_dict()[key], tensor), key
    for key, tensor in value.ema.items():
        assert torch.equal(restored.ema[key], tensor), key


def test_fallback_is_counted_as_sdpi_not_bitseed():
    value = mixed_trainer()
    batch = fixture()
    coupling = SDPIFlow.sample_coupling(batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
    store = SimpleNamespace(coupling=lambda *a, **k: (coupling, {'actual_source': 'sdpi'}))
    prepared, audit = prepare_free_batch(batch, 'bitseed', store, seed=7315)
    assert audit['fallback'] and prepared['source'] == 'sdpi'
    value.step_free(requested_source='bitseed', **prepared, t_override=.5)
    assert value.source_training['updates'] == dict(sdpi=1, bitseed=0, bianchor=0)
    assert value.source_training['fallbacks'] == 1 and value.next_source() == 'bitseed'


def test_checkpoint_rejects_missing_or_false_source_coverage_and_legacy_resume():
    value = mixed_trainer()
    payload = value.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    altered = copy.deepcopy(payload)
    altered['source_training']['updates']['bitseed'] = 1
    try:
        validate_anchored_checkpoint(altered)
    except ValueError:
        pass
    else:
        raise AssertionError('invented BiT coverage accepted')
    from test_bianchor_source import new_trainer
    try:
        new_trainer().restore(payload)
    except ValueError as error:
        assert 'source mix' in str(error)
    else:
        raise AssertionError('mixed training silently resumed as BiAnchor-only')
    del payload['source_training']
    try:
        validate_anchored_checkpoint(payload)
    except ValueError:
        pass
    else:
        raise AssertionError('mixed checkpoint accepted without source contract')


def test_free_validation_is_repeatable_and_preserves_rng():
    value = mixed_trainer()
    batch = fixture()
    coupling = SDPIFlow.sample_coupling(batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
    store = SimpleNamespace(coupling=lambda *a, **k: (coupling, {'actual_source': 'bitseed'}))
    load = lambda entry: (batch['e3_data'], batch['poi_data'], batch['ligand_data'], 16)
    rng, numpy_rng = torch.get_rng_state().clone(), np.random.get_state()
    first = evaluate_free_sources(value, [{'case': 'fixture'}], load, store, seed=7315)
    second = evaluate_free_sources(value, [{'case': 'fixture'}], load, store, seed=7315)
    assert first == second and torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert first['evaluated_by_source'] == dict(sdpi=1, bitseed=1)


def test_mixed_validation_cannot_omit_free_source_reports():
    value = mixed_trainer()
    try:
        value.record_validation({})
    except ValueError as error:
        assert 'separate real SDPI and BiT' in str(error)
    else:
        raise AssertionError('BiAnchor-only validation masqueraded as three-source validation')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print('PASS', name)
