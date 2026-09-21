"""Three initialization identities, legacy-preserving migration and resume."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_consensus import fixture
from test_anchored_trainer import trainer, update
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.bianchor_source import install_bianchor_source, anchored_source, FORMAT
from tricoopdiff.anchored_trainer import AnchoredTrainer, build_anchored_model, validate_anchored_checkpoint
from tricoopdiff.anchored_training import TRAINING_CONTRACT
from tricoopdiff.anchored_fragments import FRAGMENT_POLICY
from tricoopdiff.anchored_flow import FLOW_VERSION


def new_trainer():
    model = trainer().model
    install_bianchor_source(model)
    return AnchoredTrainer(model, parent_checkpoint_sha256='a' * 64,
                           training_index_sha256='b' * 64)


def test_migration_preserves_outputs_rng_and_routes_gradient():
    base, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**{**base.config, 'source_conditioning': True}).eval()
    model = SimpleNamespace(trifuse=fuse)
    before = [fuse(graph, proposals, state, source=i)['rank_features'].detach() for i in (0, 1)]
    rng = torch.get_rng_state().clone()
    original = copy.deepcopy(fuse.state_dict())
    install_bianchor_source(model)
    assert torch.equal(rng, torch.get_rng_state()), 'migration changed sampling RNG'
    assert anchored_source(model) == 'bianchor'
    for key, value in original.items():
        target = fuse.state_dict()[key]
        assert torch.equal(value, target[:2] if key == 'source_embedding.weight' else target)
    for index in (0, 1):
        assert torch.equal(before[index], fuse(graph, proposals, state, source=index)['rank_features'])
    result = fuse(graph, proposals, state, source='bianchor')
    assert torch.equal(before[0], result['rank_features']), 'migration must preserve legacy anchored behavior'
    result['rank_features'].square().sum().backward()
    gradient = fuse.source_embedding.weight.grad
    assert gradient[2].abs().sum() > 0
    assert gradient[:2].count_nonzero() == 0


def test_checkpoint_roundtrip_and_contract_rejection():
    value = new_trainer()
    update(value)
    payload = value.checkpoint(epoch=0, next_case=1, seed=7315, scope='implementation_smoke')
    assert payload['format'] == FORMAT
    restored = build_anchored_model(payload, value.model.expert, allow_smoke=True)
    assert anchored_source(restored) == 'bianchor'
    resumed = new_trainer()
    resumed.restore(payload)
    rng = torch.get_rng_state().clone()
    update(value)
    torch.set_rng_state(rng)
    update(resumed)
    for key, expected in value.model.trifuse.state_dict().items():
        assert torch.equal(expected, resumed.model.trifuse.state_dict()[key]), key
    for field, replacement in [('source_policy', 'anchored_only_reuses_source_embedding_zero_v1'),
                               ('format', 'anchored_consensus_training_v9'),
                               ('model_config', {**payload['model_config'], 'source_names': ['sdpi', 'bitseed']})]:
        altered = copy.deepcopy(payload)
        altered[field] = replacement
        try:
            validate_anchored_checkpoint(altered)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted contradictory {field}')
    legacy = trainer()
    assert anchored_source(legacy.model) == 0
    try:
        legacy.restore(payload)
    except ValueError:
        pass
    else:
        raise AssertionError('resume silently changed source semantics')


def test_training_and_validation_use_bianchor_row():
    value = new_trainer()
    sample = SimpleNamespace(audit={'training_contract': TRAINING_CONTRACT,
                                    'fragment_augmentation': {'policy': FRAGMENT_POLICY}},
                             flow=SimpleNamespace(version=FLOW_VERSION), prior_angles=None,
                             conditions={'sides': torch.ones(2, 5)})
    calls = []

    def objective(model, *args, source, **kwargs):
        calls.append(source)
        assert source == 'bianchor'
        return {'loss_total': model.trifuse.source_embedding.weight[2].square().mean()}

    coords = {'coords': torch.zeros(1, 1, 3)}
    with patch('tricoopdiff.anchored_trainer.anchored_target_loss', side_effect=objective):
        value.step(sample, None, coords, coords, time=.5, dt=.1)
        assert value.model.trifuse.source_embedding.weight.grad[2].abs().sum() > 0
        with value.ema_mode():
            value.evaluate(sample, None, coords, coords, time=.5, dt=.1)
    assert calls == ['bianchor', 'bianchor']


if __name__ == '__main__':
    test_migration_preserves_outputs_rng_and_routes_gradient()
    test_checkpoint_roundtrip_and_contract_rejection()
    test_training_and_validation_use_bianchor_row()
    print('BiAnchor source checks passed')
