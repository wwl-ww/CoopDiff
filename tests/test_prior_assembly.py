"""Evidence-driven actions and soft assembly have explicit, distinct contracts."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_training import fixture, prepare
from tricoopdiff.assembly_flow import AssemblyFlow
from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
from tricoopdiff.bounded_joint_motion import bounded_interface_jacobian
from tricoopdiff.prior_joint_action import PriorJointActionAdapter, prior_energies, prior_state_features
from tricoopdiff.chemical_prior_contract import chemical_prior_config, apply_pose_prior


def setup():
    e3, poi, ligand = fixture()
    sample = prepare(e3, poi, ligand)
    flow = AssemblyFlow(sample.flow, *sample.audit['selected_atom_maps'])
    return e3, poi, ligand, sample, flow


def test_soft_assembly_has_no_translation_bound_and_reachable_target():
    e3, poi, ligand, sample, flow = setup()
    state, errors = flow.encode_target(ligand['coords'][0], poi['coords'][0], ligand['edge_index'])
    assert max(errors.values()) < 1e-3
    moved = state.clone()
    moved[flow.torsion_count] += 3
    assert (flow.decode(moved)[0] - flow.decode(state)[0]).norm(dim=-1).min() > 29.9
    recovered, errors = flow.encode_target(*flow.decode(moved)[:2], ligand['edge_index'])
    assert max(errors.values()) < 1e-3
    assert torch.isfinite(bounded_interface_jacobian(flow, moved, [5., 7.])).all()


def test_assembly_pose_prior_retains_and_corrupts_binary_templates():
    _, _, _, _, flow = setup()
    config = chemical_prior_config(assembly=True)
    values = []
    for seed in range(16):
        state = flow.strict.ligand.new_zeros(flow.size)
        apply_pose_prior(flow, state, generator=torch.Generator().manual_seed(seed), config=config)
        values.append(float(state[flow.torsion_count:].norm()))
    assert any(value == 0 for value in values) and any(value > .5 for value in values)


def test_joint_prior_action_reaches_both_pose_and_torsion_outputs():
    e3, poi, ligand, sample, free = setup()
    flow = BoundedBinaryTorsionFlow(sample.flow, *sample.audit['selected_atom_maps'],
                                    translation_bounds=(.5, .5), rotation_bounds=(.2, .2))
    state = flow.sample_prior(generator=torch.Generator().manual_seed(17))
    energies, gradient = prior_state_features(flow, state, e3['coords'][0], sample.conditions)
    assert gradient.shape == (3, flow.size) and torch.isfinite(gradient).all()
    assert gradient[:, flow.torsion_count:].abs().sum() > 0
    hidden = 8
    context = {'global': torch.randn(1, hidden),
               'bond': torch.randn(1, ligand['edge_index'].shape[1], hidden),
               'ligand_directions': torch.randn_like(ligand['coords'][0]),
               'poi_directions': torch.randn_like(poi['coords'][0]),
               'edge_index': ligand['edge_index'], 'e3_coords': e3['coords'][0],
               'conditions': sample.conditions}
    model = PriorJointActionAdapter(hidden).to(state)
    velocity = torch.randn_like(state)
    result = model(flow, state, context, velocity, [5., 7.], .5)
    assert result.shape == state.shape and result.count_nonzero() == 0
    result.sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
    result = model(flow, state, context, velocity, [5., 7.], .5)
    assert result[:flow.torsion_count].abs().sum() > 0
    assert result[flow.torsion_count:].abs().sum() > 0


def test_prior_energy_has_no_global_frame_dependence():
    e3, poi, ligand, sample, flow = setup()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(11))
    xyz, protein, _ = flow.decode(state)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=xyz.dtype)
    before = prior_energies(xyz, protein, e3['coords'][0], sample.conditions)
    after = prior_energies(xyz @ rotation + 7, protein @ rotation + 7,
                          e3['coords'][0] @ rotation + 7, sample.conditions)
    assert torch.allclose(before, after, atol=1e-5)


def test_free_assembly_jacobian_matches_actual_decoder():
    import math
    from test_bounded_binary_flow import bounded
    from test_anchored_joint_motion import observed
    original = bounded()
    flow = AssemblyFlow(original.strict, original.end1, original.end2)
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    scales = state.new_tensor([4., 7.])
    matrix = bounded_interface_jacobian(flow, state, scales)
    ligand, poi, _ = flow.decode(state)
    for index in range(flow.size):
        direction = torch.eye(flow.size, dtype=state.dtype)[index]
        if index < flow.torsion_count:
            direction *= math.pi
        vl, vp = flow.cartesian_velocity(state, direction)
        assert torch.allclose(matrix[:, index], observed(ligand, poi, vl, vp, scales), atol=1e-8)


def test_joint_actions_change_when_only_binary_evidence_changes():
    e3, poi, ligand, sample, flow = setup()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(8))
    model = PriorJointActionAdapter(8)
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
    context = {'global': torch.randn(1, 8), 'bond': torch.randn(1, ligand['edge_index'].shape[1], 8),
               'ligand_directions': torch.randn_like(ligand['coords'][0]),
               'poi_directions': torch.randn_like(poi['coords'][0]),
               'edge_index': ligand['edge_index'], 'e3_coords': e3['coords'][0],
               'conditions': sample.conditions}
    velocity = torch.randn_like(state)
    before = model(flow, state, context, velocity, [5., 7.], .5)
    changed = {key: value.clone() for key, value in sample.conditions.items()}
    changed['binary1'][..., :2] += 2
    after = model(flow, state, {**context, 'conditions': changed}, velocity, [5., 7.], .5)
    assert not torch.allclose(before[:flow.torsion_count], after[:flow.torsion_count])
    assert not torch.allclose(before[flow.torsion_count:], after[flow.torsion_count:])
