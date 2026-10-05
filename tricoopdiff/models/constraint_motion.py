"""Terminal-geometry repair velocity solve for the two-stage expert flow.

All unknowns are normalized *forward-clock* velocities. The existing reverse
integrator applies -dt. Proteins remain rigid; shared ligand torsions use the
v9 symmetric horizontal tangent. No target coordinates enter this module.
"""

import math

import torch
from torch import nn


CONSTRAINT_SOLVER_CONTRACT = 'finite_displacement_detached_goal_v4'
TARGET_GEOMETRY_LOSS_WEIGHT = 0.1
DISTANCE_SCALE = 10.0  # Angstrom: distance and Jacobian normalization.
SOLVER_RIDGE = 0.01    # Isotropic in the normalized rigid/torsion coordinates.


def skew(vector):
    """Matrix [v]x such that [v]x @ u = v cross u; accepts [...,3]."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def horizontal_torsion_jacobian(coords, edge_index, edge_mask, masks):
    """[N,3,K] normalized torsion tangent of symmetric_horizontal_v1 at dt=0.

    Includes pi scaling, mean/spin removal and relative ridge of the updater.
    Reversing a bond together with its fragment leaves this tangent unchanged.
    """
    bonds = edge_index[:, edge_mask].T.to(device=coords.device)
    if not len(bonds):
        return coords.new_zeros(len(coords), 3, 0)
    if isinstance(masks, (tuple, list)):
        masks = masks[0]
    masks = torch.as_tensor(masks, device=coords.device, dtype=torch.bool)
    if masks.shape != (len(bonds), len(coords)):
        raise ValueError('constraint torsion masks must be [K,N]')
    axis = coords[bonds[:, 0]] - coords[bonds[:, 1]]
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if bool((norm <= 1e-10).any()):
        raise ValueError('degenerate constraint torsion axis')
    axis = axis / norm
    pivot = coords[bonds].mean(1)
    tangent = torch.linalg.cross(axis[:, None], coords[None] - pivot[:, None], dim=-1)
    tangent = tangent * (masks.to(coords.dtype) - .5)[..., None]
    centered = (coords - coords.mean(0)).double()
    identity = torch.eye(3, device=coords.device, dtype=torch.float64)
    inertia = centered.square().sum() * identity - centered.T @ centered
    torque = torch.linalg.cross(centered[None], tangent.double(), dim=-1).sum(1)
    ridge = inertia.trace().clamp_min(torch.finfo(torch.float64).tiny) * 1e-6
    spin = torch.linalg.solve(inertia + ridge * identity, torque.T).T
    tangent = tangent.double() - tangent.double().mean(1, keepdim=True)
    tangent = tangent - torch.linalg.cross(spin[:, None], centered[None], dim=-1)
    return tangent.to(coords.dtype).permute(1, 2, 0) * math.pi


def point_jacobian(points, component, centers, scales, width):
    """[N,3,D] rigid point velocities; E3=0, POI=1, ligand=2.

    Unknown ordering is ligand tr/rot, POI tr/rot, then shared torsions.
    """
    result = points.new_zeros(len(points), 3, width)
    if component == 0:
        return result
    offset, scale = (0, scales[0]) if component == 2 else (6, scales[1])
    result[:, :, offset:offset + 3] = torch.eye(3, device=points.device, dtype=points.dtype) * scale
    result[:, :, offset + 3:offset + 6] = -math.pi * skew(points - centers[component])
    return result


def binary_rigid_maps(centers, scales, width):
    """Normalized interface velocities for independent ligand/POI motion.

    B translation is at the ligand centroid, as in native DiffDock. This is
    the inverse of the established dual-interface -> independent conversion.
    """
    a = centers.new_zeros(6, width)
    a[:, :6] = torch.eye(6, device=centers.device, dtype=centers.dtype)
    b = torch.zeros_like(a)
    eye = torch.eye(3, device=centers.device, dtype=centers.dtype)
    b[:3, :3] = scales[0] / scales[1] * eye
    b[:3, 6:9] = -eye
    b[:3, 9:12] = -math.pi / scales[1] * skew(centers[1] - centers[2])
    b[3:, 3:6], b[3:, 9:12] = eye, -eye
    return a, b


def solve_velocity(matrix, target, precision, ridge=SOLVER_RIDGE):
    """Differentiable strictly convex WLS; finite under rank-deficient contacts.

    Float64 normal equations and positive damping avoid pseudoinverse/SVD
    discontinuities. Weights are residual precisions, not calibrated confidence.
    """
    if (matrix.ndim != 2 or target.shape != (len(matrix),)
            or precision.shape != target.shape):
        raise ValueError('invalid constraint solve dimensions or ridge')
    if (not torch.isfinite(matrix).all() or not torch.isfinite(target).all()
            or not torch.isfinite(precision).all() or bool((precision < 0).any())):
        raise ValueError('constraint solve requires finite evidence and nonnegative precision')
    a, b, w = matrix.double(), target.double(), precision.double()
    ridge = torch.as_tensor(ridge, device=a.device, dtype=a.dtype)
    if ridge.ndim != 0 or not torch.isfinite(ridge) or ridge <= 0:
        raise ValueError('constraint ridge must be a finite positive scalar')
    normal = a.T @ (w[:, None] * a)
    normal = normal + ridge * torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
    rhs = a.T @ (w * b)
    solution = torch.linalg.solve(normal, rhs)
    if not torch.isfinite(solution).all():
        raise FloatingPointError('non-finite joint constraint velocity')
    diagnostics = {
        'constraint_weighted_rms': ((w * (a @ solution - b).square()).sum()
                                    / w.sum().clamp_min(1e-12)).sqrt().detach(),
        'solver_normal_residual': torch.linalg.vector_norm(normal @ solution - rhs).detach(),
    }
    return solution.to(matrix.dtype), diagnostics


def solve_terminal_velocity(matrix, error, precision, dt):
    """Finite Gauss--Newton displacement expressed in forward-clock units.

    Damping is relative to the observable geometry; no distance-rate tanh or
    artificial remaining-time floor. dt=0 is an exact no-op. This is a
    step-size-conditioned discrete flow, not a step-size-independent ODE.
    """
    from tricoopdiff.models.operator_flow import limit_step
    duration = torch.as_tensor(dt, device=matrix.device, dtype=matrix.dtype).reshape(())
    if not torch.isfinite(duration) or duration < 0:
        raise ValueError('repair duration must be finite and nonnegative')
    # Relative isotropic damping retains SE(3) covariance and a positive floor
    # for completely unobserved modes. Keep its geometry derivative explicit.
    spectral_scale = (matrix.double().square() * precision.double()[:, None]).sum() / matrix.shape[1]
    ridge = 1e-3 * spectral_scale + 1e-8
    displacement, diagnostics = solve_velocity(matrix, error, precision, ridge=ridge)
    diagnostics['repair_ridge'] = ridge.detach()
    if not bool(duration.detach() > 0):
        return displacement * 0., diagnostics
    velocity = limit_step(displacement / duration, duration)
    return velocity, diagnostics


def terminal_geometry_loss(predictions, e3, poi, ligand_candidates):
    """Supervise one observed assembly, allowing equivalent ligand atom labels.

    Targets and current material-anchor assignments are stopped gradients.
    All interfaces use one shared symmetry choice. Learned reliability cannot
    hide prediction errors. Nothing here is called by model inference.
    """
    losses = []
    proteins = (e3, poi)
    for item in predictions:
        with torch.no_grad():
            anchor = item['anchor_weights'].detach() @ proteins[item['anchor_type']]
            left = ligand_candidates if item['receiver_type'] == 2 else poi[None]
            delta = anchor[None, None] - left[:, :, None]
            target = torch.sqrt(delta.square().sum(-1) + .01) / DISTANCE_SCALE
        predicted = item['distance'][None].expand_as(target)
        loss = torch.nn.functional.smooth_l1_loss(
            torch.log1p(predicted), torch.log1p(target), reduction='none')
        losses.append(loss.mean((-1, -2)))
    return torch.stack(torch.broadcast_tensors(*losses)).mean(0).min()


class ConstraintMotionHead(nn.Module):
    """Predict terminal geometry and solve a repair field, without expert fusion.

    All three interfaces have equal total potential precision, independent of
    residue counts. Geometry-selected hard contacts are deliberately avoided.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        def head():
            return nn.Sequential(nn.LayerNorm(4 * hidden_dim + 4),
                nn.Linear(4 * hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        # Independent parameters: endpoint control must not change the terminal
        # distance readout to compensate for insufficient corrective motion.
        self.distance_head = head()
        self.precision_head = head()
        for network in (self.distance_head, self.precision_head):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        with torch.no_grad():
            self.precision_head[-1].bias.fill_(-2.)

    def forward(self, joint, graph, proposals, context, diffusion_state):
        coords, types = graph['node_coords'], graph['node_types']
        centers = torch.stack([coords[types == i].mean(0) for i in range(3)])
        scales = diffusion_state[0, [0, 3]].exp()
        tor_a, tor_b = (proposals[name]['tor'][0] for name in ('e3_ligand', 'poi_ligand'))
        if tor_a.shape != tor_b.shape:
            raise ValueError('binary torsion suggestions must share the same bonds')
        count, width = len(tor_a), 12 + len(tor_a)
        ligand = coords[types == 2]
        torsion = (horizontal_torsion_jacobian(ligand, graph['lig_edge_index'],
                    graph['lig_edge_mask'], graph['lig_mask_rotate']) if count else None)
        if torsion is not None and torsion.shape[-1] != count:
            raise ValueError('expert torsions must match the ligand tangent')
        matrices, targets, weights, geometry_predictions = [], [], [], []
        points, latent_types = joint['points'], joint['latent_types']
        # Each interface is visited once; moving receivers retain full detail.
        for receiver_type, anchor_type in ((2, 0), (2, 1), (1, 0)):
            receiver = (types == receiver_type).nonzero().flatten()
            anchor = (latent_types == anchor_type).nonzero().flatten()
            left, right = coords[receiver], points[anchor]
            ja = point_jacobian(left, receiver_type, centers, scales, width)
            if receiver_type == 2 and count:
                ja[:, :, 12:] = torsion
            jb = point_jacobian(right, anchor_type, centers, scales, width)
            # Anchors are instantaneous material points of a rigid protein.
            # Their learned assignment is fixed for this tangent, not detached
            # from outer training gradients.
            delta = right[None] - left[:, None]
            direction = delta / torch.sqrt(delta.square().sum(-1, keepdim=True) + .01)
            jacobian = torch.einsum('ijc,ijcd->ijd', direction, jb[None] - ja[:, None]) / DISTANCE_SCALE
            shape = (len(left), len(right), context.shape[-1])
            inputs = torch.cat((joint['features'][receiver, None].expand(shape),
                                joint['latent_features'][None, anchor].expand(shape),
                                joint['relation'][receiver][:, anchor], context[:, None].expand(shape),
                                joint['geometry'][receiver][:, anchor]), -1)
            predicted_distance = torch.nn.functional.softplus(self.distance_head(inputs)[..., 0])
            precision_logits = self.precision_head(inputs)[..., 0]
            distance = torch.sqrt(delta.square().sum(-1) + .01) / DISTANCE_SCALE
            geometry_predictions.append({
                'distance': predicted_distance, 'receiver_type': receiver_type,
                'anchor_type': anchor_type,
                'anchor_weights': joint['material_weights'][anchor_type].detach()})
            matrices.append(jacobian.reshape(-1, width))
            targets.append((distance - predicted_distance.detach()).reshape(-1))
            weights.append((.001 + precision_logits.sigmoid()).reshape(-1) / (len(left) * len(right)))
        relative_weights = torch.cat(weights)
        total_weight = relative_weights.sum()
        relative_weights = relative_weights / total_weight
        velocity, diagnostics = solve_terminal_velocity(
            torch.cat(matrices), torch.cat(targets), relative_weights, diffusion_state[0, 5])
        from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE, SYMMETRIC_TORSION_MODE
        predictions = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                       'torsion_update_mode': SYMMETRIC_TORSION_MODE,
                       'torsion': velocity[None, 12:]}
        for index, name in enumerate(('e3_ligand', 'poi_ligand')):
            offset = index * 6
            predictions[name] = {'tr': velocity[None, offset:offset + 3],
                                 'rot': velocity[None, offset + 3:offset + 6],
                                 'tor': proposals[name]['tor']}
        return {'normalized_velocity': predictions, 'rank_features': context,
                'target_geometry': geometry_predictions,
                'residual_penalty': context.new_zeros(()),
                'consensus_diagnostics': {**diagnostics, 'constraint_solver': CONSTRAINT_SOLVER_CONTRACT,
                    'interface_precisions': (torch.stack([w.sum() for w in weights[:3]]) / total_weight).detach(),
                    'solved_dofs': width,
                    'constraint_rows': sum(len(value) for value in targets)}}
