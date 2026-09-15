"""CPU-only contracts for experimental off-path body-frame transport."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.offpath_experiments import (
    BoundedJointEndpointRolloutTrainer,
    BodyFrameTransportRolloutTrainer,
    GAUGE_CORRECT_CONTRACT_MODE,
    GAUGE_CORRECT_TORSION_CONTRACT,
    GaugeCorrectBodyFrameTransportRolloutTrainer,
    JOINT_ENDPOINT_CONTRACT_MODE,
    JOINT_ENDPOINT_TORSION_CONTRACT,
    OFFPATH_CONTRACT_MODE,
    TORSION_GAUGE_CONTRACT,
    _state_torsion_gauge,
    transport_poi_body_frame_targets,
    transport_targets_with_recomputed_gauge,
)
from tricoopdiff.sdpi_flow import _rotation_exp


def rotation(vector):
    return _rotation_exp(torch.as_tensor(vector, dtype=torch.float64).reshape(1, 3))


def targets(dtype=torch.float64):
    return {
        'e3_ligand': {
            'tr': torch.tensor([[7., 8., 9.]], dtype=dtype),
            'rot': torch.tensor([[1., 4., 2.]], dtype=dtype),
        },
        'poi_ligand': {
            'tr': torch.tensor([[2., -3., 5.]], dtype=dtype),
            'rot': torch.tensor([[-4., 1., 6.]], dtype=dtype),
        },
        'torsion': torch.tensor([[.2, -.7]], dtype=dtype),
    }


def test_identity_when_teacher_and_rollout_bodies_equal():
    body = rotation([.3, -.5, .7])
    original = targets()
    result = transport_poi_body_frame_targets(original, body, body)
    assert torch.allclose(result['poi_ligand']['tr'], original['poi_ligand']['tr'], atol=1e-12)
    assert torch.allclose(result['poi_ligand']['rot'], original['poi_ligand']['rot'], atol=1e-12)
    assert result['e3_ligand'] is original['e3_ligand']
    assert result['torsion'] is original['torsion']


def test_transport_is_globally_rotation_equivariant():
    teacher_body = rotation([.2, -.4, .1])
    rollout_body = rotation([-.7, .3, .5])
    global_rotation = rotation([.6, .1, -.2])
    original = targets()
    result = transport_poi_body_frame_targets(original, teacher_body, rollout_body)
    rotated_targets = {
        **original,
        'poi_ligand': {
            **original['poi_ligand'],
            'tr': torch.einsum('...i,...ij->...j', original['poi_ligand']['tr'],
                               global_rotation.transpose(-1, -2)),
            'rot': torch.einsum('...i,...ij->...j', original['poi_ligand']['rot'],
                                global_rotation.transpose(-1, -2)),
        },
    }
    rotated = transport_poi_body_frame_targets(
        rotated_targets, global_rotation @ teacher_body, global_rotation @ rollout_body)
    for component in ('tr', 'rot'):
        expected = torch.einsum('...i,...ij->...j', result['poi_ligand'][component],
                                global_rotation.transpose(-1, -2))
        assert torch.allclose(rotated['poi_ligand'][component], expected, atol=1e-12)


def test_transport_preserves_rigid_label_norms():
    original = targets()
    result = transport_poi_body_frame_targets(
        original, rotation([.8, -.2, .4]), rotation([-.1, .9, -.3]))
    for component in ('tr', 'rot'):
        before = torch.linalg.vector_norm(original['poi_ligand'][component], dim=-1)
        after = torch.linalg.vector_norm(result['poi_ligand'][component], dim=-1)
        assert torch.allclose(after, before, atol=1e-12)


def torsion_example():
    teacher = torch.tensor([[[0., 0., 0.], [1., 0., 0.],
                             [1., 1., 0.], [2., 1., .5]]], dtype=torch.float64)
    rollout = teacher.clone()
    rollout[0, 2:] = rollout[0, 2:] @ rotation([.6, 0., 0.])[0].T
    coupling = {
        'bonds': torch.tensor([[0, 1]]).numpy(),
        'masks': torch.tensor([[False, False, True, True]]).numpy(),
    }
    return teacher, rollout, coupling


def test_gauge_correct_transport_removes_and_recomputes_common_spin():
    teacher_ligand, rollout_ligand, coupling = torsion_example()
    teacher_body, rollout_body = rotation([.3, -.2, .1]), rotation([-.4, .2, .5])
    original = targets()
    teacher_gauge = _state_torsion_gauge(
        teacher_ligand, coupling, original['torsion'][:, :1])
    rollout_gauge = _state_torsion_gauge(
        rollout_ligand, coupling, original['torsion'][:, :1])
    # Match the one active bond in this synthetic graph.
    one_torsion = {**original, 'torsion': original['torsion'][:, :1]}
    result = transport_targets_with_recomputed_gauge(
        one_torsion, teacher_body, rollout_body,
        teacher_ligand, rollout_ligand, coupling)
    assert torch.allclose(result['e3_ligand']['rot'],
                          original['e3_ligand']['rot'] - teacher_gauge + rollout_gauge,
                          atol=1e-12)
    old_global = torch.einsum('...i,...ij->...j',
                              original['poi_ligand']['rot'], teacher_body)
    new_global = torch.einsum('...i,...ij->...j',
                              result['poi_ligand']['rot'], rollout_body)
    assert torch.allclose(new_global, old_global - teacher_gauge + rollout_gauge,
                          atol=1e-12)


def test_gauge_correct_contract_has_distinct_checkpoint_identity():
    trainer = GaugeCorrectBodyFrameTransportRolloutTrainer.__new__(
        GaugeCorrectBodyFrameTransportRolloutTrainer)
    trainer.offpath_contract_mode = GAUGE_CORRECT_CONTRACT_MODE
    trainer.rollout_probability = .5
    trainer.rollout_steps = 5
    trainer.rollout_min_time = .25
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_contract_mode'] == GAUGE_CORRECT_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == GAUGE_CORRECT_TORSION_CONTRACT


def test_bounded_joint_endpoint_changes_only_rigid_targets_within_caps():
    native_ligand, rollout_ligand, coupling = torsion_example()
    native_poi = torch.tensor(
        [[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]],
        dtype=torch.float64)
    poi_rotation = rotation([-.2, .3, .1])
    rollout_poi = native_poi @ poi_rotation.transpose(-1, -2) + torch.tensor(
        [[[2., -1., .5]]], dtype=torch.float64)
    teacher_body = rotation([.3, -.2, .1])
    rollout_body = rotation([-.4, .2, .5])
    original = targets()
    original = {**original, 'torsion': original['torsion'][:, :1]}
    teacher = (native_ligand, native_poi, teacher_body, original)

    trainer = BoundedJointEndpointRolloutTrainer.__new__(
        BoundedJointEndpointRolloutTrainer)
    trainer.rollout_min_time = .25
    trainer.endpoint_blend = .25
    trainer.endpoint_translation_cap = 2.
    trainer.endpoint_rotation_cap = .4
    trainer.joint_endpoint_applied_states = 0
    result = trainer._map_targets(
        teacher, rollout_ligand, rollout_poi, rollout_body, coupling,
        ligand_data={'coords': native_ligand},
        poi_data={'coords': native_poi},
        time=torch.tensor([.5], dtype=torch.float64))
    base = transport_targets_with_recomputed_gauge(
        original, teacher_body, rollout_body,
        native_ligand, rollout_ligand, coupling)

    assert trainer.joint_endpoint_applied_states == 1
    assert result['torsion'] is base['torsion']
    assert torch.equal(result['torsion'], original['torsion'])
    for branch in ('e3_ligand', 'poi_ligand'):
        for component, cap in (('tr', 2.), ('rot', .4)):
            change = torch.linalg.vector_norm(
                result[branch][component] - base[branch][component], dim=-1)
            assert bool((change <= .25 * cap + 1e-12).all()), (
                branch, component, change)


def test_bounded_joint_endpoint_contract_has_distinct_checkpoint_identity():
    trainer = BoundedJointEndpointRolloutTrainer.__new__(
        BoundedJointEndpointRolloutTrainer)
    trainer.offpath_contract_mode = JOINT_ENDPOINT_CONTRACT_MODE
    trainer.rollout_probability = .5
    trainer.rollout_steps = 5
    trainer.rollout_min_time = .25
    trainer.endpoint_blend = .25
    trainer.endpoint_translation_cap = 10.
    trainer.endpoint_rotation_cap = .5
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_target'] == JOINT_ENDPOINT_CONTRACT_MODE
    assert contract['offpath_contract_mode'] == JOINT_ENDPOINT_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == JOINT_ENDPOINT_TORSION_CONTRACT
    assert contract['endpoint_blend'] == .25
    assert contract['endpoint_translation_cap'] == 10.
    assert contract['endpoint_rotation_cap'] == .5


class FakeFlow:
    def __init__(self, teacher_body, rollout_body, teacher_targets):
        self.teacher_body = teacher_body
        self.rollout_body = rollout_body
        self.teacher_targets = teacher_targets
        self.path_calls = 0

    def path(self, ligand_data, poi_coords, coupling, time):
        self.path_calls += 1
        body = (self.teacher_body if torch.equal(time, torch.tensor([.5], dtype=time.dtype))
                else rotation([-.1, .2, .3]))
        ligand = ligand_data['coords'] + self.path_calls
        poi = poi_coords + self.path_calls
        return ligand, poi, body, self.teacher_targets

    def normalized_predictions(self, output, time):
        return output

    def velocities(self, normalized, e3_coords, poi_coords):
        return normalized

    def step(self, ligand, poi, body, velocities, dt, ligand_data):
        return ligand + 10, poi + 20, self.rollout_body


def test_training_state_maps_only_generated_offpath_poi_targets():
    teacher_body = rotation([.4, -.3, .2])
    rollout_body = rotation([-.6, .1, .5])
    original = targets()
    trainer = BodyFrameTransportRolloutTrainer.__new__(BodyFrameTransportRolloutTrainer)
    trainer.diffusion = FakeFlow(teacher_body, rollout_body, original)
    trainer.rollout_probability = 1.0
    trainer.rollout_steps = 1
    trainer.rollout_min_time = 0.0
    trainer.rollout_augmented_states = 41  # Must not be used to identify this state.
    trainer._forward_state = lambda *args, **kwargs: {'deterministic': True}
    coords = torch.zeros(1, 2, 3, dtype=torch.float64)
    state = trainer._training_state(
        {'coords': coords}, {'coords': coords}, {'coords': coords}, {},
        torch.tensor([.5], dtype=torch.float64),
        torch.tensor([.05], dtype=torch.float64))
    expected = transport_poi_body_frame_targets(original, teacher_body, rollout_body)
    assert trainer.rollout_augmented_states == 42
    assert torch.equal(state[2], rollout_body)
    for component in ('tr', 'rot'):
        assert torch.allclose(state[3]['poi_ligand'][component],
                              expected['poi_ligand'][component], atol=1e-12)
    assert state[3]['e3_ligand'] is original['e3_ligand']
    assert state[3]['torsion'] is original['torsion']


def test_teacher_state_is_not_mapped_and_contract_records_mode():
    original = targets()
    trainer = BodyFrameTransportRolloutTrainer.__new__(BodyFrameTransportRolloutTrainer)
    trainer.diffusion = FakeFlow(rotation([.1, .2, .3]), rotation([.7, -.2, .4]), original)
    trainer.rollout_probability = 0.0
    trainer.rollout_steps = 1
    trainer.rollout_min_time = 0.0
    trainer.rollout_augmented_states = 99
    coords = torch.zeros(1, 2, 3, dtype=torch.float64)
    state = trainer._training_state(
        {'coords': coords}, {'coords': coords}, {'coords': coords}, {},
        torch.tensor([.5], dtype=torch.float64),
        torch.tensor([.05], dtype=torch.float64))
    assert state[3] is original
    assert trainer.rollout_augmented_states == 99

    trainer.offpath_contract_mode = OFFPATH_CONTRACT_MODE
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_target'] == OFFPATH_CONTRACT_MODE
    assert contract['offpath_contract_mode'] == OFFPATH_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == TORSION_GAUGE_CONTRACT
    assert contract['rollout_probability'] == 0.0


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
