"""CPU contracts for differentiable short-horizon trajectory experiments."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.experiments.trajectory_experiment import (
    DISTANCE_STAGED_POI_CONTRACT_MODE,
    PRIOR_REPLAY_CONTRACT_MODE,
    PRIOR_REPLAY_EVAL_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE,
    STAGED_POI_CONTRACT_MODE,
    TRAJECTORY_CONTRACT_MODE,
    DeterministicPriorReplayTrajectoryTrainer,
    PriorReplayTrajectoryTrainer,
    PriorReplayJointTranslationTrainer,
    PriorReplayJointTranslationReplaceTrainer,
    PriorReplayJointTranslationWindowReplaceTrainer,
    PriorReplayJointTranslationSoftTimeTrainer,
    PriorReplayJointTranslationDirectionTrainer,
    PriorReplayJointTranslationAgreementTrainer,
    StagedPOITrajectoryTrainer,
    TrajectoryConsistencyTrainer,
    differentiable_sdpi_step,
    remaining_centroid_loss,
    joint_endpoint_translation_loss,
    prior_replay_schedule,
    staged_poi_state_loss,
)


def example(dtype=torch.float64):
    ligand = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
          [2., 1., .5], [2.5, 1.5, 1.2]]], dtype=dtype)
    poi = torch.tensor(
        [[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]],
        dtype=dtype)
    body = torch.eye(3, dtype=dtype).reshape(1, 3, 3)
    ligand_data = {
        'edge_index': torch.tensor([[0, 1], [1, 0]]),
        'edge_mask': torch.tensor([True, False]),
        'mask_rotate': np.array([[False, True, True, True, True]]),
    }
    velocities = {
        'e3_ligand': {
            'tr': torch.tensor([[.4, -.2, .3]], dtype=dtype),
            'rot': torch.tensor([[.2, .1, -.3]], dtype=dtype),
        },
        'poi_ligand': {
            'tr': torch.tensor([[-.1, .5, .2]], dtype=dtype),
            'rot': torch.tensor([[.3, -.2, .1]], dtype=dtype),
        },
        'torsion': torch.tensor([[.7]], dtype=dtype),
    }
    return ligand, poi, body, ligand_data, velocities


def test_differentiable_step_matches_production_solver():
    ligand, poi, body, ligand_data, velocities = example()
    dt = torch.tensor([.05], dtype=ligand.dtype)
    expected = SDPIFlow.step(
        ligand, poi, body, velocities, dt, ligand_data)
    actual = differentiable_sdpi_step(
        ligand, poi, body, velocities, dt, ligand_data)
    for value, reference in zip(actual, expected):
        assert torch.allclose(value, reference, atol=2e-7, rtol=2e-7), (
            value - reference).abs().max()


def test_remaining_centroid_loss_uses_exact_independent_motion_and_gradients():
    from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE

    ligand, poi, body, data, velocities = example()
    velocities['rigid_output_mode'] = INDEPENDENT_RIGID_MODE
    time = ligand.new_tensor([.4])
    scales = ligand.new_tensor([10., 15.])
    target_ligand, target_poi, _ = SDPIFlow.step(ligand, poi, body, velocities, time, data)
    loss = remaining_centroid_loss(ligand, poi, velocities, time, target_ligand, target_poi, scales)
    assert loss < 1e-20, 'rotation and torsion must not change the supervised centroid'
    rotation = torch.linalg.matrix_exp(torch.tensor([[0., -.3, .2], [.3, 0., -.1], [-.2, .1, 0.]], dtype=ligand.dtype))
    moved = {**velocities, **{b: {**velocities[b], 'tr': velocities[b]['tr'] @ rotation.T}
                            for b in ('e3_ligand', 'poi_ligand')}}
    reference = remaining_centroid_loss(ligand, poi, velocities, time, ligand, poi, scales)
    transformed = remaining_centroid_loss(ligand @ rotation.T + 10, poi @ rotation.T + 10,
                                         moved, time, ligand @ rotation.T + 10,
                                         poi @ rotation.T + 10, scales)
    assert torch.allclose(reference, transformed, atol=1e-12)
    for branch in ('e3_ligand', 'poi_ligand'):
        velocities[branch]['tr'] = velocities[branch]['tr'].clone().requires_grad_()
    loss = remaining_centroid_loss(ligand, poi, velocities, time, ligand, poi, scales)
    loss.backward()
    for branch in ('e3_ligand', 'poi_ligand'):
        grad = velocities[branch]['tr'].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    zero = remaining_centroid_loss(ligand, poi, velocities, time * 0, ligand, poi, scales)
    assert zero == 0 and torch.isfinite(zero), 'no inverse-time singularity is allowed'


def test_remaining_centroid_loss_rejects_legacy_motion():
    ligand, poi, _, _, velocities = example()
    try:
        remaining_centroid_loss(ligand, poi, velocities, ligand.new_tensor([.5]),
                                ligand, poi, ligand.new_tensor([10., 15.]))
    except ValueError as error:
        assert 'independent' in str(error)
    else:
        raise AssertionError('dual-interface velocities do not independently move the POI centroid')


def test_gradients_cross_two_differentiable_steps():
    ligand, poi, body, ligand_data, velocities = example()
    leaves = []
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            value = velocities[branch][component].clone().requires_grad_()
            velocities[branch][component] = value
            leaves.append(value)
    velocities['torsion'] = velocities['torsion'].clone().requires_grad_()
    leaves.append(velocities['torsion'])
    dt = torch.tensor([.05], dtype=ligand.dtype)
    for _ in range(2):
        ligand, poi, body = differentiable_sdpi_step(
            ligand, poi, body, velocities, dt, ligand_data)
    loss = ligand.square().mean() + poi.square().mean() + body.square().mean()
    loss.backward()
    for value in leaves:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert float(value.grad.abs().sum()) > 0


def test_differentiable_step_supports_no_torsion_graphs():
    ligand, poi, body, ligand_data, velocities = example()
    ligand_data = {
        **ligand_data,
        'edge_mask': torch.tensor([False, False]),
        'mask_rotate': np.zeros((0, ligand.shape[1]), dtype=bool),
    }
    velocities = {**velocities, 'torsion': ligand.new_empty((1, 0))}
    dt = torch.tensor([.05], dtype=ligand.dtype)
    expected = SDPIFlow.step(ligand, poi, body, velocities, dt, ligand_data)
    actual = differentiable_sdpi_step(
        ligand, poi, body, velocities, dt, ligand_data)
    for value, reference in zip(actual, expected):
        assert torch.allclose(value, reference, atol=1e-12, rtol=1e-12)


def test_no_torsion_output_can_disable_existing_rotatable_bonds():
    ligand, poi, body, ligand_data, velocities = example()
    assert bool(ligand_data['edge_mask'].any())
    velocities['torsion'] = ligand.new_empty((1, 0))
    dt = ligand.new_tensor([.05])
    actual = SDPIFlow.differentiable_step(ligand, poi, body, velocities, dt, ligand_data)
    expected = SDPIFlow.step(ligand, poi, body, velocities, dt, ligand_data)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(actual, expected))


def test_trajectory_contract_has_checkpoint_identity():
    trainer = TrajectoryConsistencyTrainer.__new__(TrajectoryConsistencyTrainer)
    trainer.trajectory_contract_mode = TRAJECTORY_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = .1
    trainer.trajectory_min_time = .25
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == TRAJECTORY_CONTRACT_MODE
    assert contract['trajectory_steps'] == 3
    assert contract['trajectory_weight'] == .1
    assert contract['trajectory_min_time'] == .25


def test_staged_poi_loss_jointly_penalizes_center_and_orientation():
    poi = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]],
        dtype=torch.float64, requires_grad=True)
    target = poi.detach() + torch.tensor([[[2., -1., 3.]]], dtype=torch.float64)
    native = target + torch.tensor([[[20., 0., 0.]]], dtype=torch.float64)
    body = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).requires_grad_()
    target_body = body.detach().clone()
    target_body[:, 0, 0] = 0
    loss, parts = staged_poi_state_loss(
        poi, target, native, body, target_body, poi.new_tensor(10.),
        distance_weighted=True, distance_weight_cap=3.)
    assert parts['center'] > 0 and parts['orientation'] > 0
    assert 1 < parts['distance_weight'] <= 3
    loss.backward()
    assert float(poi.grad.abs().sum()) > 0
    assert float(body.grad.abs().sum()) > 0


def test_staged_poi_contracts_are_distinct_and_complete():
    for mode, weighted in (
            (STAGED_POI_CONTRACT_MODE, False),
            (DISTANCE_STAGED_POI_CONTRACT_MODE, True)):
        trainer = StagedPOITrajectoryTrainer.__new__(StagedPOITrajectoryTrainer)
        trainer.trajectory_contract_mode = mode
        trainer.trajectory_steps = 5
        trainer.trajectory_weight = 1.
        trainer.trajectory_min_time = .25
        trainer.trajectory_max_time = .65
        trainer.distance_weighted = weighted
        trainer.distance_weight_cap = 3.
        trainer.activation_clamp = 100.
        trainer.activation_clamp_mode = 'irrep_norm'
        trainer.residual_weight = 1e-3
        trainer.ema_decay = .999
        trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
        contract = trainer.runtime_contract()
        assert contract['trajectory_objective'] == mode
        assert contract['trajectory_max_time'] == .65
        assert contract['trajectory_distance_weighted'] is weighted
        assert contract['trajectory_distance_weight_cap'] == 3.


def test_prior_replay_schedule_reaches_requested_time():
    time = torch.tensor([.43], dtype=torch.float64)
    count, step = prior_replay_schedule(
        time, inference_steps=20, max_steps=15)
    assert count == 12
    assert torch.allclose(torch.ones_like(time) - count * step, time)
    capped_count, capped_step = prior_replay_schedule(
        time, inference_steps=20, max_steps=4)
    assert capped_count == 4
    assert torch.allclose(torch.ones_like(time) - capped_count * capped_step, time)
    endpoint_count, endpoint_step = prior_replay_schedule(
        torch.ones_like(time), inference_steps=20, max_steps=15)
    assert endpoint_count == 0
    assert torch.equal(endpoint_step, torch.zeros_like(time))


def test_prior_replay_contract_records_exposure_policy():
    trainer = PriorReplayTrajectoryTrainer.__new__(PriorReplayTrajectoryTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .25
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_CONTRACT_MODE
    assert contract['trajectory_replay_start_time'] == 1.
    assert contract['trajectory_replay_max_steps'] == 15
    assert contract['trajectory_replay_gradients'] is False


def test_deterministic_prior_replay_uses_eval_and_restores_training():
    trainer = DeterministicPriorReplayTrajectoryTrainer.__new__(
        DeterministicPriorReplayTrajectoryTrainer)
    trainer.model = torch.nn.Sequential(torch.nn.Dropout(.5))
    trainer.model.train()
    with trainer._replay_context():
        assert not trainer.model.training
    assert trainer.model.training

    trainer.trajectory_contract_mode = PRIOR_REPLAY_EVAL_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .25
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_EVAL_CONTRACT_MODE
    assert contract['trajectory_replay_model_mode'] == 'eval'


def test_joint_endpoint_translation_loss_only_differentiates_translations():
    _, _, _, _, velocities = example()
    targets = {}
    leaves = []
    for branch in ('e3_ligand', 'poi_ligand'):
        targets[branch] = {}
        for component in ('tr', 'rot'):
            value = velocities[branch][component].clone().requires_grad_()
            velocities[branch][component] = value
            leaves.append((component, value))
            targets[branch][component] = torch.zeros_like(value)
    loss = joint_endpoint_translation_loss(
        velocities, targets,
        (torch.tensor(2., dtype=torch.float64),
         torch.tensor(4., dtype=torch.float64)))
    loss.backward()
    for component, value in leaves:
        if component == 'tr':
            assert value.grad is not None
            assert float(value.grad.abs().sum()) > 0
        else:
            assert value.grad is None


def test_prior_replay_translation_contract_is_explicit():
    trainer = PriorReplayJointTranslationTrainer.__new__(
        PriorReplayJointTranslationTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = .1
    trainer.trajectory_min_time = .3
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE
    assert contract['trajectory_min_time'] == .3
    assert contract['trajectory_auxiliary_branches'] == [
        'e3_ligand.tr', 'poi_ligand.tr']
    assert contract['trajectory_auxiliary_state_gradients'] is False
    assert contract['trajectory_auxiliary_native_reading'] == 'training_only'
    assert contract['trajectory_rotation_torsion_target'] == 'analytic_teacher_only'


def test_prior_replay_translation_replacement_contract_is_explicit():
    trainer = PriorReplayJointTranslationReplaceTrainer.__new__(
        PriorReplayJointTranslationReplaceTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .3
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == (
        PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE)
    assert contract['trajectory_translation_teacher_policy'] == (
        'replace_when_prior_replay_target_available')


def test_prior_replay_translation_window_contract_is_explicit():
    trainer = PriorReplayJointTranslationWindowReplaceTrainer.__new__(
        PriorReplayJointTranslationWindowReplaceTrainer)
    trainer.trajectory_contract_mode = (
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE)
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .3
    trainer.trajectory_max_time = .65
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == (
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE)
    assert contract['trajectory_min_time'] == .3
    assert contract['trajectory_max_time'] == .65
    assert contract['trajectory_translation_teacher_policy'] == (
        'replace_only_inside_prior_replay_time_window')


def test_prior_replay_translation_window_keeps_early_teacher_target():
    trainer = PriorReplayJointTranslationWindowReplaceTrainer.__new__(
        PriorReplayJointTranslationWindowReplaceTrainer)
    trainer.trajectory_max_time = .65
    trainer._translation_auxiliary_applied = True
    time = torch.tensor([.8])
    trainer._trajectory_context = (None, None, None, None, time, None, None)
    loss = trainer._trajectory_loss()
    assert loss.item() == 0
    assert not loss.requires_grad
    assert trainer._translation_auxiliary_applied is False


def test_soft_time_translation_mix_has_expected_endpoints():
    trainer = PriorReplayJointTranslationSoftTimeTrainer.__new__(
        PriorReplayJointTranslationSoftTimeTrainer)
    trainer.endpoint_zero_time = .8
    trainer.endpoint_full_time = .3
    for time, expected in ((.8, 0.), (.55, .5), (.3, 1.)):
        trainer._trajectory_context = (
            None, None, None, None, torch.tensor([time]), None, None)
        assert torch.allclose(
            trainer._translation_replacement_mix(), torch.tensor(expected))


def test_direction_translation_uses_teacher_magnitude():
    trainer = PriorReplayJointTranslationDirectionTrainer.__new__(
        PriorReplayJointTranslationDirectionTrainer)
    endpoint = torch.tensor([[0., 2., 0.]])
    analytic = torch.tensor([[3., 0., 0.]])
    prediction = torch.tensor([[0., 3., 0.]])
    branches = ('e3_ligand', 'poi_ligand')
    velocities = {branch: {'tr': prediction} for branch in branches}
    endpoint_targets = {branch: {'tr': endpoint} for branch in branches}
    analytic_targets = {branch: {'tr': analytic} for branch in branches}
    loss = trainer._translation_loss(
        velocities, endpoint_targets, (torch.ones(1), torch.ones(1)),
        analytic_targets)
    assert torch.allclose(loss, torch.zeros(()))


def test_agreement_translation_falls_back_for_opposite_target():
    trainer = PriorReplayJointTranslationAgreementTrainer.__new__(
        PriorReplayJointTranslationAgreementTrainer)
    endpoint = torch.tensor([[-2., 0., 0.]])
    analytic = torch.tensor([[3., 0., 0.]])
    branches = ('e3_ligand', 'poi_ligand')
    velocities = {branch: {'tr': analytic} for branch in branches}
    endpoint_targets = {branch: {'tr': endpoint} for branch in branches}
    analytic_targets = {branch: {'tr': analytic} for branch in branches}
    loss = trainer._translation_loss(
        velocities, endpoint_targets, (torch.ones(1), torch.ones(1)),
        analytic_targets)
    assert torch.allclose(loss, torch.zeros(()))


def test_new_translation_contract_names_are_distinct():
    assert len({
        PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE,
    }) == 5


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
