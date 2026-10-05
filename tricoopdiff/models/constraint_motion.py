"""Joint binary-evidence / ternary-constraint velocity solve (model-4.5).

All unknowns are normalized *forward-clock* velocities. The existing reverse
integrator applies -dt. Proteins remain rigid; shared ligand torsions use the
v9 symmetric horizontal tangent. No target coordinates enter this module.
"""

import math

import torch
from torch import nn


CONSTRAINT_SOLVER_CONTRACT = 'supervised_terminal_distance_v2'
TARGET_GEOMETRY_LOSS_WEIGHT = 0.1
DISTANCE_SCALE = 10.0  # Angstrom: distance-rate residual normalization.
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
            or precision.shape != target.shape or not math.isfinite(ridge) or ridge <= 0):
        raise ValueError('invalid constraint solve dimensions or ridge')
    if (not torch.isfinite(matrix).all() or not torch.isfinite(target).all()
            or not torch.isfinite(precision).all() or bool((precision < 0).any())):
        raise ValueError('constraint solve requires finite evidence and nonnegative precision')
    a, b, w = matrix.double(), target.double(), precision.double()
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


def terminal_distance_rate(distance, predicted_distance, time):
    """Forward-clock rate: reverse integration moves toward the predicted goal.

    Distances are normalized by 10 A. A clock floor bounds the final correction;
    the limit of two normalized distance units also bounds far-start requests.
    """
    return 2. * torch.tanh((distance - predicted_distance) / (2. * time.clamp_min(.2)))


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
    """Predict terminal distances and relative weights, then solve all DOFs.

    All three interfaces have equal total potential precision, independent of
    residue counts. Geometry-selected hard contacts are deliberately avoided.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.contact_head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim + 4), nn.Linear(4 * hidden_dim + 4, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.expert_precision = nn.Linear(hidden_dim + 14, 4)
        self.torsion_precision = nn.Sequential(nn.Linear(hidden_dim + 2, hidden_dim),
                                               nn.SiLU(), nn.Linear(hidden_dim, 2))
        for head in (self.contact_head[-1], self.expert_precision, self.torsion_precision[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.contact_head[-1].bias[1] = -2.
            self.expert_precision.bias.fill_(-2.)
            self.torsion_precision[-1].bias.fill_(-2.)

    def forward(self, joint, graph, proposals, context, evidence, bonds, diffusion_state):
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
            values = self.contact_head(inputs)
            predicted_distance = torch.nn.functional.softplus(values[..., 0])
            distance = torch.sqrt(delta.square().sum(-1) + .01) / DISTANCE_SCALE
            geometry_predictions.append({
                'distance': predicted_distance, 'receiver_type': receiver_type,
                'anchor_type': anchor_type,
                'anchor_weights': joint['material_weights'][anchor_type].detach()})
            matrices.append(jacobian.reshape(-1, width))
            targets.append(terminal_distance_rate(distance, predicted_distance,
                                                  diffusion_state[0, -1]).reshape(-1))
            weights.append((.001 + values[..., 1].sigmoid()).reshape(-1) / (len(left) * len(right)))
        a_map, b_map = binary_rigid_maps(centers, scales, width)
        precision = .001 + self.expert_precision(torch.cat((context, evidence), -1))[0].sigmoid()
        from tricoopdiff.models.consensus import bounded_vector
        for index, (name, mapping) in enumerate(zip(('e3_ligand', 'poi_ligand'), (a_map, b_map))):
            matrices.append(mapping)
            targets.append(torch.cat([bounded_vector(proposals[name][key])[0] for key in ('tr', 'rot')]))
            weights.append(precision[index * 2:index * 2 + 2].repeat_interleave(3) / 3.)
        if count:
            values = self.torsion_precision(torch.cat((bonds[0], torch.stack((tor_a.tanh(), tor_b.tanh()), -1)), -1))
            mapping = coords.new_zeros(count, width)
            mapping[:, 12:] = torch.eye(count, device=coords.device, dtype=coords.dtype)
            for index, target in enumerate((tor_a.tanh(), tor_b.tanh())):
                matrices.append(mapping)
                targets.append(target)
                weights.append((.001 + values[:, index].sigmoid()) / count)
        relative_weights = torch.cat(weights)
        total_weight = relative_weights.sum()
        relative_weights = relative_weights / total_weight
        velocity, diagnostics = solve_velocity(torch.cat(matrices), torch.cat(targets), relative_weights)
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
                    'binary_precisions': (precision / total_weight).detach(),
                    'interface_precisions': (torch.stack([w.sum() for w in weights[:3]]) / total_weight).detach(),
                    'solved_dofs': width,
                    'constraint_rows': sum(len(value) for value in targets)}}
