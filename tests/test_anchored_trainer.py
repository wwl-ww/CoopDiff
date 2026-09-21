"""Checkpoint isolation, immutable publication and deterministic optimizer resume."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_consensus import Expert
from test_consensus import fixture
from tricoopdiff.anchored_flow import FLOW_VERSION
from tricoopdiff.anchored_fragments import FRAGMENT_POLICY
from tricoopdiff.anchored_training import TRAINING_CONTRACT
from tricoopdiff.anchored_trainer import AnchoredTrainer, build_anchored_model, validate_anchored_checkpoint
from tricoopdiff.anchored_trainer import warm_start_anchored_weights
from tricoopdiff.conditional_consensus import ConditionedConsensusCoopDiff, ConditionedConsensusFuse
from tricoopdiff.conditioned_consensus_train import validate_conditioned_checkpoint


def trainer(validation_contract=None, binary_bounds=None, body_aware=False, coupled=False, **kwargs):
    base, _, _, _ = fixture()
    expert = Expert()
    expert.provenance = {'diffdock_commit': 'fixture', 'score_checkpoint_sha256': 'c' * 64,
                         'no_torsion': False}
    model = ConditionedConsensusCoopDiff(
        expert, ConditionedConsensusFuse(**{**base.config, 'source_conditioning': True}))
    if body_aware or coupled:
        from tricoopdiff.anchored_action_adapter import install_action_adapter, ADAPTER_POLICY
        from tricoopdiff.coupled_linker_action import COUPLED_ADAPTER_POLICY
        install_action_adapter(model, policy=COUPLED_ADAPTER_POLICY if coupled else ADAPTER_POLICY)
    return AnchoredTrainer(model, parent_checkpoint_sha256='a' * 64, training_index_sha256='b' * 64,
                           validation_contract=validation_contract, binary_bounds=binary_bounds, **kwargs)


def test_coupled_checkpoint_migrates_v6_and_cannot_silently_drop_joint_head():
    from tricoopdiff.anchored_validation import VALIDATION_POLICY, VALIDATION_TIMES
    contract = dict(policy=VALIDATION_POLICY, times=list(VALIDATION_TIMES), seed=7315, index_sha256='e' * 64)
    parent = trainer(contract, body_aware=True)
    parent.record_validation({**contract, 'global_step': 0, 'mean_loss': .5,
                              'cohort_sha256': 'f' * 64, 'evaluated': 1})
    source = parent.checkpoint(epoch=0, next_case=0, seed=7315)
    target = trainer(coupled=True)
    try:
        warm_start_anchored_weights(target.model, source)
    except ValueError as error:
        assert 'reset_action_adapter' in str(error)
    else:
        raise AssertionError('stacked action replacement was not explicitly acknowledged')
    warm_start_anchored_weights(target.model, source, reset_action_adapter=True)
    for key, value in source['ema_trifuse_state_dict'].items():
        if key.startswith('anchored_action_adapter.'):
            continue
        assert torch.equal(target.model.trifuse.state_dict()[key], value)
    assert target.model.trifuse.anchored_action_adapter.coupled_output.weight.count_nonzero() == 0
    payload = target.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    assert payload['format'] == 'anchored_consensus_training_v8'
    restored = build_anchored_model(payload, target.model.expert, allow_smoke=True)
    for key, value in payload['ema_trifuse_state_dict'].items():
        assert torch.equal(restored.trifuse.state_dict()[key], value)
    try:
        trainer(body_aware=True).restore(payload)
    except ValueError as error:
        assert 'adapter' in str(error)
    else:
        raise AssertionError('joint head was silently removed on resume')


def test_body_aware_checkpoint_roundtrip_and_explicit_backbone_migration():
    from tricoopdiff.anchored_validation import VALIDATION_POLICY, VALIDATION_TIMES
    contract = dict(policy=VALIDATION_POLICY, times=list(VALIDATION_TIMES), seed=7315, index_sha256='e' * 64)
    parent = trainer(contract)
    parent.record_validation({**contract, 'global_step': 0, 'mean_loss': .5,
                              'cohort_sha256': 'f' * 64, 'evaluated': 1})
    source = parent.checkpoint(epoch=0, next_case=0, seed=7315)
    target = trainer(body_aware=True)
    warm_start_anchored_weights(target.model, source)
    for key, value in source['ema_trifuse_state_dict'].items():
        assert torch.equal(target.model.trifuse.state_dict()[key], value)
    assert torch.count_nonzero(target.model.trifuse.anchored_action_adapter.net[-1].weight) == 0
    payload = target.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    assert payload['format'] == 'anchored_consensus_training_v6'
    model = build_anchored_model(payload, target.model.expert, allow_smoke=True)
    assert hasattr(model.trifuse, 'anchored_action_adapter')
    for key, value in payload['ema_trifuse_state_dict'].items():
        assert torch.equal(model.trifuse.state_dict()[key], value)
    try:
        trainer().restore(payload)
    except ValueError as error:
        assert 'adapter' in str(error)
    else:
        raise AssertionError('resume silently removed the body-aware head')


def test_checkpoint_prevents_silent_task_or_prior_changes():
    from tricoopdiff.chemical_prior_contract import chemical_prior_config
    bounds = {'translation': [.5, .5], 'rotation': [.2, .2]}
    value = trainer(binary_bounds=bounds, conditioning_task='single_anchor', prior_config=chemical_prior_config())
    payload = value.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    validate_anchored_checkpoint(payload)
    restored = trainer(binary_bounds=bounds, conditioning_task='single_anchor', prior_config=chemical_prior_config())
    restored.restore(payload)
    for target in (trainer(binary_bounds=bounds),
                   trainer(binary_bounds=bounds, conditioning_task='single_anchor')):
        try:
            target.restore(payload)
        except ValueError as error:
            assert 'task or chemical prior' in str(error)
        else:
            raise AssertionError('checkpoint silently changed task/prior')
    altered = copy.deepcopy(payload)
    altered.pop('conditioning_task')
    try:
        validate_anchored_checkpoint(altered)
    except ValueError:
        pass
    else:
        raise AssertionError('single-anchor checkpoint was interpreted as dual-binary')


def test_physical_metric_and_cluster_schedule_require_new_checkpoint_contract():
    from tricoopdiff.anchored_data_schedule import CLUSTER_ORDER
    from tricoopdiff.anchored_training_metric import PHYSICAL_METRIC
    value = trainer(binary_metric=PHYSICAL_METRIC, data_schedule=CLUSTER_ORDER)
    payload = value.checkpoint(epoch=2, next_case=3, seed=7315, scope='implementation_smoke')
    assert payload['format'] == 'anchored_consensus_training_v5'
    restored = trainer(binary_metric=PHYSICAL_METRIC, data_schedule=CLUSTER_ORDER)
    assert restored.restore(payload) == {'epoch': 2, 'next_case': 3, 'seed': 7315}
    try:
        trainer().restore(payload)
    except ValueError as error:
        assert 'metric or data schedule' in str(error)
    else:
        raise AssertionError('resume silently changed the velocity metric or data visits')
    for field in ('binary_metric', 'data_schedule'):
        modified = copy.deepcopy(payload)
        modified[field] = 'unknown_policy'
        try:
            validate_anchored_checkpoint(modified)
        except ValueError:
            pass
        else:
            raise AssertionError('unknown training metric/data policy accepted')


def update(value):
    sample = SimpleNamespace(audit={'training_contract': TRAINING_CONTRACT,
                                    'fragment_augmentation': {'policy': FRAGMENT_POLICY}},
                             flow=SimpleNamespace(version=FLOW_VERSION), prior_angles=None,
                             conditions={'sides': torch.ones(2, 5)})

    def objective(model, *args, **kwargs):
        loss = (model.trifuse.torsion_coordinator.net[-1].weight - 1).square().mean() * torch.rand(())
        return {'loss_total': loss}

    with patch('tricoopdiff.anchored_trainer.anchored_target_loss', side_effect=objective):
        return value.step(sample, None, {'coords': torch.zeros(1, 1, 3)},
                          {'coords': torch.zeros(1, 1, 3)}, time=.5, dt=.1)


def test_resume_restores_optimizer_ema_cursor_and_random_stream(tmp_path):
    first = trainer()
    update(first)
    path = tmp_path / 'step1.pt'
    first.save(path, epoch=0, next_case=1, seed=7315, scope='implementation_smoke')
    payload = torch.load(path, weights_only=False)
    expected_loss = update(first)
    second = trainer()
    assert second.restore(payload) == {'epoch': 0, 'next_case': 1, 'seed': 7315}
    assert update(second) == expected_loss
    for key, value in first.model.trifuse.state_dict().items():
        assert torch.equal(value, second.model.trifuse.state_dict()[key]), key
        assert torch.equal(first.ema[key], second.ema[key]), key
    before = path.read_bytes()
    try:
        first.save(path, epoch=0, next_case=2, seed=7315, scope='implementation_smoke')
    except FileExistsError:
        pass
    else:
        raise AssertionError('checkpoint publication overwrote an existing artifact')
    assert path.read_bytes() == before


def test_checkpoint_cannot_masquerade_as_old_flow_or_trained_smoke():
    value = trainer()
    payload = value.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    for action in (lambda: validate_conditioned_checkpoint(payload),
                   lambda: build_anchored_model(payload, value.model.expert)):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible format/smoke gate was bypassed')
    restored = build_anchored_model(payload, value.model.expert, allow_smoke=True)
    for key, tensor in restored.trifuse.state_dict().items():
        assert torch.equal(tensor, value.ema[key])
    for key, bad in [('source_policy', 'sdpi'), ('generative_process', 'sdpi_v1'),
                     ('action_policy', 'torsion_only'), ('format', 'anchored_consensus_training_v2')]:
        modified = copy.deepcopy(payload)
        modified[key] = bad
        try:
            validate_anchored_checkpoint(modified)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid {key} accepted')
    modified = copy.deepcopy(payload)
    modified['training_index_sha256'] = 'd' * 64
    try:
        value.restore(modified)
    except ValueError as error:
        assert 'training_index_sha256' in str(error)
    else:
        raise AssertionError('resume silently changed training dataset')


def test_bounded_checkpoint_cannot_change_uncertainty_or_become_strict():
    bounds = {'translation': [.5, .5], 'rotation': [.2, .2]}
    value = trainer(binary_bounds=bounds)
    payload = value.checkpoint(epoch=0, next_case=0, seed=7315, scope='implementation_smoke')
    restored = trainer(binary_bounds=bounds)
    restored.restore(payload)
    assert restored.binary_bounds == bounds
    try:
        trainer().restore(payload)
    except ValueError as error:
        assert 'bounds' in str(error)
    else:
        raise AssertionError('bounded checkpoint silently became a strict model')
    for bad in (None, {'translation': [.5, .5], 'rotation': [4., 4.]}):
        modified = copy.deepcopy(payload)
        modified['binary_bounds'] = bad
        try:
            validate_anchored_checkpoint(modified)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid or missing uncertainty declaration accepted')


def test_explicit_weight_transfer_does_not_restore_task_state():
    from tricoopdiff.anchored_validation import VALIDATION_POLICY, VALIDATION_TIMES

    contract = dict(policy=VALIDATION_POLICY, times=list(VALIDATION_TIMES), seed=7315,
                    index_sha256='e' * 64)
    source = trainer(contract)
    update(source)
    source.record_validation({**contract, 'global_step': source.global_step, 'mean_loss': .5,
                              'cohort_sha256': 'f' * 64, 'evaluated': 1})
    payload = source.checkpoint(epoch=0, next_case=1, seed=7315)
    target = trainer(binary_bounds={'translation': [.5, .5], 'rotation': [.2, .2]})
    warm_start_anchored_weights(target.model, payload)
    for key, value in target.model.trifuse.state_dict().items():
        assert torch.equal(value, source.ema[key])
    assert target.global_step == 0 and not target.optimizer.state
    assert target.validation_state['best_loss'] is None
    invalid = copy.deepcopy(payload)
    invalid['scope'] = 'implementation_smoke'
    try:
        warm_start_anchored_weights(target.model, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('smoke artifact accepted as a trained weight parent')
