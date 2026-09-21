"""Joint interface conversion with explicit, bounded binary-pose uncertainty."""

import math

import torch

from .anchored_joint_motion import interface_jacobian
from .bounded_binary_flow import ball_map, rotation_exp


BOUNDED_ACTION_POLICY = 'bounded_joint_interface_least_squares_v2'


def _offset_derivatives(latent, translation_radius, rotation_radius):
    """Exact ball derivative and SO(3) left Jacobian; no repeated autograd calls."""
    identity = torch.eye(3, dtype=latent.dtype, device=latent.device)

    def ball_derivative(z, radius):
        # Same overflow-safe rescaling as ball_map; the scale cancels exactly.
        scale = z.abs().max().clamp_min(1.)
        unit = z / scale
        denominator = (scale.reciprocal().square() + unit.square().sum()).sqrt()
        return (radius / scale / denominator) * (
            identity - torch.outer(unit, unit) / denominator.square())

    translation = ball_derivative(latent[0], translation_radius)
    vector = ball_map(latent[1], rotation_radius)
    x, y, z = vector.unbind()
    zero = x.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta = vector.norm()
    squared = theta.square()
    a = .5 * torch.sinc(theta / (2 * math.pi)).square()
    b = torch.where(theta < 1e-4, 1 / 6 - squared / 120 + squared.square() / 5040,
                    (1 - torch.sinc(theta / math.pi)) / squared.clamp_min(1e-16))
    left = identity + a * skew + b * (skew @ skew)
    angular = left @ ball_derivative(latent[1], rotation_radius)
    return translation.detach(), angular.detach()


def bounded_interface_jacobian(flow, state, scales):
    """[12,K+12] Jacobian; torsions use pi units, offset latents unit scale."""
    torsion = interface_jacobian(flow, state, scales)
    with torch.no_grad():
        ligand, poi, _ = flow.decode(state)
        ligand, poi = ligand.double(), poi.double()
        offsets = state[flow.torsion_count:].double().reshape(2, 2, 3)
        scales = torch.as_tensor(scales, dtype=torch.float64, device=state.device).reshape(2)
        base_ligand, _, base_body = (x.double() for x in flow.strict.decode(state[:flow.torsion_count]))
        re = rotation_exp(flow.offset_map(offsets[0, 1], flow.rotation_bounds[0]))
        transported = re @ base_body @ flow.strict.body.double().T
        te = flow.offset_map(offsets[0, 0], flow.translation_bounds[0])
        tp = transported @ flow.offset_map(offsets[1, 0], flow.translation_bounds[1])
        e3_pivot = flow.strict.ligand[list(flow.end1)].double().mean(0) + te
        poi_pivot = ligand[list(flow.end2)].mean(0) + tp
        matrix = torch.zeros((12, 12), dtype=torch.float64, device=state.device)
    dte, dwe = flow.offset_derivatives(offsets[0], flow.translation_bounds[0], flow.rotation_bounds[0])
    dtp, dwp = flow.offset_derivatives(offsets[1], flow.translation_bounds[1], flow.rotation_bounds[1])
    dtp, dwp = transported @ dtp, transported @ dwp
    matrix[:3, :3] = dte / scales[0]
    matrix[:3, 3:6] = torch.linalg.cross(dwe.T, (ligand.mean(0) - e3_pivot).expand(3, 3)).T / scales[0]
    matrix[3:6, 3:6] = dwe / math.pi
    matrix[6:9, 6:9] = -dtp / scales[1]
    matrix[6:9, 9:] = -torch.linalg.cross(dwp.T, (ligand.mean(0) - poi_pivot).expand(3, 3)).T / scales[1]
    matrix[9:, 9:] = -dwp / math.pi
    return torch.cat((torsion, matrix), 1).detach()


def bounded_joint_velocity(flow, state, normalized, scales):
    torsion = flow.consensus_velocity(normalized['torsion']) / math.pi
    rigid = []
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            value = normalized[branch][component]
            if (value.shape != (1, 3) or not torch.isfinite(value).all()
                    or value.dtype != state.dtype or value.device != state.device):
                raise ValueError('bounded joint proposals must be finite [1,3] on state dtype/device')
            rigid.append(value[0])
    target = torch.cat(rigid).double()
    matrix = bounded_interface_jacobian(flow, state, scales)
    # Offset latents are not measured in normalized rigid-velocity units. A
    # unit-sized penalty here would suppress sub-Angstrom corrections and force
    # the rigid heads to emit huge proposals. Use only a small numerical ridge;
    # total template displacement is bounded by the decoder, not this penalty.
    prior = torch.cat((torsion.double(), state.new_zeros(12).double()))
    weights = state.new_full((flow.size,), 1e-4, dtype=torch.float64)
    weights[:flow.torsion_count] = 1 / max(flow.torsion_count, 1) + 1e-6
    solution = torch.linalg.solve(matrix.T @ matrix / 3 + torch.diag(weights),
                                  matrix.T @ target / 3 + weights * prior)
    if not torch.isfinite(solution).all():
        raise FloatingPointError('nonfinite bounded joint velocity')
    physical = torch.cat((solution[:flow.torsion_count] * math.pi, solution[flow.torsion_count:]))
    policy = ('unbounded_joint_interface_v1' if flow.version == 'unbounded_binary_pose_assembly_chart_v1'
              else BOUNDED_ACTION_POLICY)
    return physical.to(state), {'policy': policy,
        'rigid_conflict_rms': float((matrix @ solution - target).detach().square().mean().sqrt()),
        'offset_velocity_rms': float(solution[flow.torsion_count:].detach().square().mean().sqrt())}
