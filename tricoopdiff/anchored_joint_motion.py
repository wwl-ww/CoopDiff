"""Convert both interface rigid proposals and torsions into one feasible velocity.

Interface B describes ligand motion relative to the moving POI, expressed in
the global frame, NOT an independent global POI velocity. Rotations use the
instantaneous ligand centroid, matching SDPI's horizontal torsion gauge.
"""

import math

import torch


ACTION_POLICY = 'anchored_joint_interface_least_squares_v1'


def interface_jacobian(flow, angles, scales):
    """Return [12,K] normalized interface velocities per normalized angle speed.

    Bridge cuts rooted at the fixed end form nested/disjoint moving subtrees.
    Their final axes therefore give the exact instantaneous decoder Jacobian,
    irrespective of the stored bond order. Geometry is not a learned variable.
    """
    with torch.no_grad():
        ligand, poi, _ = flow.decode(angles)
        ligand, poi = ligand.double(), poi.double()
        scales = torch.as_tensor(scales, device=ligand.device, dtype=torch.float64).reshape(-1)
        if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('joint motion requires two finite positive interface scales')
        count = len(flow.plan.bonds)
        if not count:
            return ligand.new_zeros((12, 0))
        ligand_columns, poi_translation, poi_rotation = [], [], []
        for (u, v), moving, follows in zip(
                flow.plan.bonds, flow.plan.moving_atoms, flow.plan.moves_poi):
            axis = ligand[u] - ligand[v]
            if axis.norm() < 1e-8:
                raise ValueError('degenerate joint-motion axis')
            axis = axis / axis.norm()
            velocity = torch.zeros_like(ligand)
            indices = list(moving)
            velocity[indices] = torch.linalg.cross(axis.expand(len(indices), 3),
                                                   ligand[indices] - ligand[v])
            ligand_columns.append(velocity)
            poi_rotation.append(axis if follows else torch.zeros_like(axis))
            poi_translation.append(torch.linalg.cross(axis, poi.mean(0) - ligand[v])
                                   if follows else torch.zeros_like(axis))
        columns = torch.stack(ligand_columns)
        center = ligand.mean(0)
        x = ligand - center
        inertia = torch.eye(3, device=x.device, dtype=x.dtype) * x.square().sum() - x.T @ x
        torque = torch.linalg.cross(x.expand_as(columns), columns).sum(1)
        spin = torque @ torch.linalg.pinv(inertia, hermitian=True, rtol=1e-12).T
        translation = columns.mean(1)
        poi_spin, poi_center_velocity = torch.stack(poi_rotation), torch.stack(poi_translation)
        # Relative ligand-center velocity subtracts the POI velocity AT that
        # center, not at the POI centroid. Both branch vectors are global.
        relative_translation = translation - poi_center_velocity - torch.linalg.cross(
            poi_spin, (center - poi.mean(0)).expand_as(poi_spin))
        relative_spin = spin - poi_spin
        return torch.cat((translation / scales[0], spin / math.pi,
                          relative_translation / scales[1], relative_spin / math.pi), 1).T * math.pi


def joint_anchored_velocity(flow, angles, normalized, scales):
    """Differentiable weighted solve; only the decoder executes the result.

    Each normalized 3-vector has unit aggregate weight, as does the complete
    torsion vector. A tiny additional torsion-centered ridge stabilizes the
    solve without biasing exactly compatible proposals toward zero.
    """
    torsion = flow.consensus_velocity(normalized['torsion']) / math.pi
    rigid = []
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            value = normalized[branch][component]
            if (value.shape != (1, 3) or not torch.isfinite(value).all()
                    or value.dtype != angles.dtype or value.device != angles.device):
                raise ValueError('joint rigid proposals must match finite [1,3] angle dtype/device')
            rigid.append(value[0])
    target = torch.cat(rigid).double()
    matrix = interface_jacobian(flow, angles, scales)
    if not torsion.numel():
        return torsion, {'policy': ACTION_POLICY, 'rigid_conflict_rms': float(target.square().mean().sqrt())}
    weight = 1 / torsion.numel() + 1e-6
    normal = matrix.T @ matrix / 3 + weight * torch.eye(
        torsion.numel(), dtype=matrix.dtype, device=matrix.device)
    rhs = matrix.T @ target / 3 + weight * torsion.double()
    solution = torch.linalg.solve(normal, rhs)
    if not torch.isfinite(solution).all():
        raise FloatingPointError('nonfinite joint anchored velocity')
    diagnostics = {'policy': ACTION_POLICY,
                   'rigid_conflict_rms': float((matrix @ solution - target).detach().square().mean().sqrt()),
                   'torsion_conflict_rms': float((solution - torsion).detach().square().mean().sqrt())}
    return solution.to(angles) * math.pi, diagnostics
