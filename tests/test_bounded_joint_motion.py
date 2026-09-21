"""Bounded offset Jacobians and gradients through the executed joint action."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_bounded_binary_flow import bounded
from test_anchored_joint_motion import observed, proposals
from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
from tricoopdiff.bounded_joint_motion import bounded_interface_jacobian, bounded_joint_velocity
from tricoopdiff.anchored_joint_motion import joint_anchored_velocity


def test_bounded_interface_jacobian_matches_all_decoder_directions():
    flow = bounded()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    scales = state.new_tensor([4., 7.])
    matrix = bounded_interface_jacobian(flow, state, scales)
    ligand, poi, _ = flow.decode(state)
    for index in range(flow.size):
        velocity = torch.eye(flow.size, dtype=state.dtype)[index]
        if index < flow.torsion_count:
            velocity = velocity * math.pi
        vl, vp = flow.cartesian_velocity(state, velocity)
        expected = observed(ligand, poi, vl, vp, scales)
        assert torch.allclose(matrix[:, index], expected, atol=1e-9), (index, matrix[:, index], expected)


def test_bounded_joint_gradient_and_zero_bound_reduction():
    flow = bounded()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    rigid = torch.randn(12, dtype=state.dtype, requires_grad=True)
    torsion = torch.randn(flow.torsion_count, dtype=state.dtype, requires_grad=True)
    normalized = proposals(flow.strict, rigid, torsion)
    scales = state.new_tensor([4., 7.])
    velocity, _ = bounded_joint_velocity(flow, state, normalized, scales)
    velocity.square().sum().backward()
    assert torch.isfinite(rigid.grad).all() and rigid.grad.abs().sum() > 0
    assert torch.isfinite(torsion.grad).all() and torsion.grad.abs().sum() > 0
    assert velocity[flow.torsion_count:].abs().sum() > 0
    exact = BoundedBinaryTorsionFlow(flow.strict, flow.end1, flow.end2,
                                    translation_bounds=(0., 0.), rotation_bounds=(0., 0.))
    bounded_velocity, _ = bounded_joint_velocity(exact, state, normalized, scales)
    strict_velocity, _ = joint_anchored_velocity(flow.strict, state[:flow.torsion_count], normalized, scales)
    assert torch.allclose(bounded_velocity[:flow.torsion_count], strict_velocity, atol=1e-10)
    assert torch.count_nonzero(bounded_velocity[flow.torsion_count:]) == 0
def test_analytic_offsets_match_autograd_at_zero_and_large_latents():
    from tricoopdiff.bounded_joint_motion import _offset_derivatives
    from tricoopdiff.bounded_binary_flow import ball_map, rotation_exp
    for values in ([[0., 0., 0.], [0., 0., 0.]],
                   [[.2, -.7, 1.1], [.3, -.8, .6]],
                   [[1e6, -1e5, 1e4], [-1e3, 2e3, 4e3]]):
        latent = torch.tensor(values, dtype=torch.float64)
        actual_t, actual_r = _offset_derivatives(latent, .5, .2)
        expected_t = torch.autograd.functional.jacobian(lambda z: ball_map(z, .5), latent[0])
        matrix = rotation_exp(ball_map(latent[1], .2))
        derivative = torch.autograd.functional.jacobian(lambda z: rotation_exp(ball_map(z, .2)), latent[1])
        spin = torch.einsum('ijk,lj->ilk', derivative, matrix)
        expected_r = torch.stack((spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0],
                                  spin[1, 0] - spin[0, 1])) / 2
        assert torch.allclose(actual_t, expected_t, atol=1e-14, rtol=1e-9)
        assert torch.allclose(actual_r, expected_r, atol=1e-14, rtol=1e-9)
