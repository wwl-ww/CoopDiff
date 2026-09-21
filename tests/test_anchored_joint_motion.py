"""Independent derivative, frame, compatibility and gradient checks."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.anchored_joint_motion import interface_jacobian, joint_anchored_velocity


def proposals(flow, rigid, torsion):
    full = torsion.new_zeros((1, len(flow.plan.original_indices) + len(flow.plan.rejected)))
    full[0, list(flow.plan.original_indices)] = torsion
    return {'e3_ligand': {'tr': rigid[:3][None], 'rot': rigid[3:6][None]},
            'poi_ligand': {'tr': rigid[6:9][None], 'rot': rigid[9:][None]},
            'torsion': full}


def observed(ligand, poi, v_ligand, v_poi, scales):
    def rigid(coords, velocity):
        x = coords - coords.mean(0)
        # Independent least-squares rigid fit, not the inertia implementation.
        basis = torch.eye(3, dtype=x.dtype)
        spins = torch.stack([torch.linalg.cross(b.expand_as(x), x).flatten() for b in basis], 1)
        spin = torch.linalg.lstsq(spins, (velocity - velocity.mean(0)).flatten()).solution
        return velocity.mean(0), spin
    tl, wl = rigid(ligand, v_ligand)
    tp, wp = rigid(poi, v_poi)
    return torch.cat((tl / scales[0], wl / math.pi,
                      (tl - tp - torch.linalg.cross(wp, ligand.mean(0) - poi.mean(0))) / scales[1],
                      (wl - wp) / math.pi))


def test_joint_jacobian_matches_decoder_derivative():
    from dataclasses import replace
    state = fixture()
    for order in ((0, 1, 2), (2, 0, 1)):
        plan = state[3]
        plan = replace(plan, **{key: tuple(getattr(plan, key)[i] for i in order)
                               for key in ('bonds', 'moving_atoms', 'moves_poi', 'original_indices')})
        flow = AnchoredTorsionFlow(*state[:3], plan)
        angles = flow.ligand.new_tensor([.7, -.3, 1.2])
        scales = angles.new_tensor([4., 7.])
        matrix = interface_jacobian(flow, angles, scales)
        ligand, poi, _ = flow.decode(angles)
        for index in range(3):
            velocity = torch.eye(3, dtype=angles.dtype)[index] * math.pi
            vl, vp = flow.cartesian_velocity(angles, velocity)
            expected = observed(ligand, poi, vl, vp, scales)
            assert torch.allclose(matrix[:, index], expected, atol=1e-10)


def test_joint_solve_recovers_compatible_motion_and_trains_all_proposals():
    flow = AnchoredTorsionFlow(*fixture())
    angles = flow.ligand.new_tensor([.4, -.6, .8])
    scales = angles.new_tensor([4., 7.])
    matrix = interface_jacobian(flow, angles, scales)
    target = angles.new_tensor([.2, -.4, .3])
    velocity, audit = joint_anchored_velocity(flow, angles, proposals(flow, matrix @ target, target), scales)
    assert torch.allclose(velocity, target * math.pi, atol=1e-10)
    assert audit['rigid_conflict_rms'] < 1e-10
    rigid = torch.randn(12, dtype=angles.dtype, requires_grad=True)
    torsion = torch.randn(3, dtype=angles.dtype, requires_grad=True)
    velocity, _ = joint_anchored_velocity(flow, angles, proposals(flow, rigid, torsion), scales)
    velocity.square().sum().backward()
    assert torsion.grad.abs().sum() > 0
    for group in rigid.grad.reshape(4, 3):
        assert torch.isfinite(group).all() and group.abs().sum() > 0


def test_joint_solve_is_rigid_frame_invariant_and_preserves_binary_geometry():
    ligand, poi, body, plan = fixture()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=ligand.dtype)
    shift = ligand.new_tensor([100., -30., 50.])
    flow = AnchoredTorsionFlow(ligand, poi, body, plan)
    transformed = AnchoredTorsionFlow(ligand @ rotation.T + shift, poi @ rotation.T + shift,
                                      rotation @ body, plan)
    angles, scales = ligand.new_tensor([.7, -.2, 1.]), ligand.new_tensor([4., 7.])
    rigid, torsion = torch.randn(12, dtype=ligand.dtype), torch.randn(3, dtype=ligand.dtype)
    velocity, _ = joint_anchored_velocity(flow, angles, proposals(flow, rigid, torsion), scales)
    other, _ = joint_anchored_velocity(transformed, angles, proposals(
        flow, (rigid.reshape(4, 3) @ rotation.T).flatten(), torsion), scales)
    assert torch.allclose(velocity, other, atol=1e-10)
    _, (moved, receptor, _) = flow.step(angles, velocity, .1)
    assert torch.equal(moved[:3], flow.ligand[:3])
    assert torch.allclose(torch.cdist(moved[5:], receptor),
                          torch.cdist(flow.ligand[5:], flow.poi), atol=1e-10)


def test_interface_observables_match_existing_sdpi_rigid_step():
    from tricoopdiff.sdpi_flow import SDPIFlow

    ligand, poi, body, _ = fixture()
    scales = ligand.new_tensor([4., 7.])
    raw = torch.randn(12, dtype=ligand.dtype)
    velocities = {'e3_ligand': {'tr': raw[:3][None] * scales[0],
                                'rot': raw[3:6][None] * math.pi},
                  'poi_ligand': {'tr': raw[6:9][None] * scales[1],
                                 'rot': raw[9:][None] * math.pi},
                  'torsion': ligand.new_empty((1, 0))}
    epsilon = 1e-7
    moved, receptor, _ = SDPIFlow.step(
        ligand[None], poi[None], body[None], velocities, ligand.new_tensor([epsilon]),
        {'edge_mask': torch.zeros(1, dtype=torch.bool)})
    measured = observed(ligand, poi, (ligand - moved[0]) / epsilon,
                         (poi - receptor[0]) / epsilon, scales)
    assert torch.allclose(measured, raw, atol=1e-5), \
        'relative interface center/spin convention disagrees with existing integrator'
