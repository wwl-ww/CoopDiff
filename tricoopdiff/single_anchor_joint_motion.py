"""Joint action conversion for one known interface and a freely moving POI."""

import math

import torch

from .anchored_joint_motion import interface_jacobian
from .bounded_binary_flow import ball_map, rotation_exp
from .bounded_joint_motion import _offset_derivatives


SINGLE_ACTION_POLICY = 'single_anchor_joint_interface_least_squares_v1'


def single_interface_jacobian(flow, state, scales):
    torsion = interface_jacobian(flow, state, scales)
    with torch.no_grad():
        ligand, poi, _ = flow.decode(state)
        ligand, poi = ligand.double(), poi.double()
        k = flow.torsion_count
        offset = state[k:k + 6].double().reshape(2, 3)
        pivot = (flow.strict.ligand[list(flow.anchor)].double().mean(0)
                 + ball_map(offset[0], flow.translation_bound))
        scales = torch.as_tensor(scales, dtype=torch.float64, device=state.device).reshape(2)
        matrix = torch.zeros((12, 12), dtype=torch.float64, device=state.device)
    translation, spin = _offset_derivatives(offset, flow.translation_bound, flow.rotation_bound)
    ligand_velocity = torch.cat((translation, torch.linalg.cross(
        spin.T, (ligand.mean(0) - pivot).expand(3, 3)).T), 1)
    matrix[:3, :6] = ligand_velocity / scales[0]
    matrix[3:6, 3:6] = spin / math.pi
    matrix[6:9, :6] = ligand_velocity / scales[1]
    matrix[9:, 3:6] = spin / math.pi
    matrix[6:9, 6:9] = -torch.eye(3, dtype=matrix.dtype, device=matrix.device) * flow.scale / scales[1]
    coordinate = state[k + 9:].double().detach()
    rotation = rotation_exp(coordinate * math.pi)
    derivative = torch.autograd.functional.jacobian(lambda z: rotation_exp(z * math.pi), coordinate)
    skew = torch.einsum('ijk,lj->ilk', derivative, rotation)
    angular = torch.stack((skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0],
                            skew[1, 0] - skew[0, 1])) / 2
    matrix[6:9, 9:] = -torch.linalg.cross(
        angular.T, (ligand.mean(0) - poi.mean(0)).expand(3, 3)).T / scales[1]
    matrix[9:, 9:] = -angular / math.pi
    return torch.cat((torsion, matrix), 1).detach()


def single_joint_velocity(flow, state, normalized, scales):
    torsion = flow.consensus_velocity(normalized['torsion']) / math.pi
    values = [normalized[b][c] for b in ('e3_ligand', 'poi_ligand') for c in ('tr', 'rot')]
    if any(v.shape != (1, 3) or not torch.isfinite(v).all()
           or v.dtype != state.dtype or v.device != state.device for v in values):
        raise ValueError('single-anchor proposals must be finite [1,3] on state dtype/device')
    target = torch.cat([v[0] for v in values]).double()
    matrix = single_interface_jacobian(flow, state, scales)
    prior = torch.cat((torsion.double(), state.new_zeros(12).double()))
    weights = state.new_full((flow.size,), 1e-6, dtype=torch.float64)
    weights[:flow.torsion_count] = 1 / max(flow.torsion_count, 1) + 1e-6
    weights[flow.torsion_count:flow.torsion_count + 6] = 1e-4
    solution = torch.linalg.solve(matrix.T @ matrix / 3 + torch.diag(weights),
                                  matrix.T @ target / 3 + weights * prior)
    if not torch.isfinite(solution).all():
        raise FloatingPointError('nonfinite single-anchor velocity')
    velocity = torch.cat((solution[:flow.torsion_count] * math.pi, solution[flow.torsion_count:]))
    return velocity.to(state), {'policy': SINGLE_ACTION_POLICY,
        'rigid_conflict_rms': float((matrix @ solution - target).detach().square().mean().sqrt())}
