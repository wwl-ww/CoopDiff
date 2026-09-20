"""Differentiable Cartesian tangents for the shared SDPI rigid/torsion gauge."""

import torch


def torsion_tangents(ligand, edge_index, edge_mask, mask_rotate):
    """Return [T,N,3] tangents, removing the Kabsch rigid gauge analytically."""
    bonds = edge_index[:, edge_mask].T
    masks = torch.as_tensor(mask_rotate, device=ligand.device, dtype=ligand.dtype)
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.shape != (len(bonds), len(ligand)):
        raise ValueError('torsion masks do not match bonds and ligand atoms')
    if not len(bonds):
        return ligand.new_zeros((0, len(ligand), 3))
    start, end = bonds.T
    # Vendored torsion updates rotate about v -> u, not u -> v.
    axis = ligand[start] - ligand[end]
    length = axis.norm(dim=-1, keepdim=True)
    if (length < 1e-8).any():
        raise ValueError('degenerate torsion axis')
    lever = ligand[None] - ligand[end, None]
    tangent = torch.cross((axis / length)[:, None].expand_as(lever), lever, dim=-1)
    tangent = tangent * masks[..., None]
    tangent = tangent - tangent.mean(1, keepdim=True)
    centered = ligand - ligand.mean(0)
    inertia = torch.eye(3, device=ligand.device, dtype=ligand.dtype) * centered.square().sum()
    inertia = inertia - centered.T @ centered
    torque = torch.cross(centered[None].expand_as(tangent), tangent, dim=-1).sum(1)
    spin = torque @ torch.linalg.pinv(inertia).T
    return tangent - torch.cross(spin[:, None].expand_as(tangent),
                                 centered[None].expand_as(tangent), dim=-1)


def cartesian_velocities(ligand, poi, velocities, ligand_data):
    """Forward-time velocities; reverse integration negates these tangents."""
    center = ligand.mean(0)
    left, right = velocities['e3_ligand'], velocities['poi_ligand']
    ligand_velocity = left['tr'] + torch.cross(
        left['rot'].expand_as(ligand), ligand - center, dim=-1)
    if velocities['torsion'].numel():
        tangent = torsion_tangents(ligand, ligand_data['edge_index'],
                                   ligand_data['edge_mask'], ligand_data['mask_rotate'])
        ligand_velocity = ligand_velocity + torch.einsum('t,tnc->nc',
                                                        velocities['torsion'][0], tangent)
    poi_velocity = left['tr'] - right['tr'] + torch.cross(
        (left['rot'] - right['rot']).expand_as(poi), poi - center, dim=-1)
    return ligand_velocity, poi_velocity


def structural_velocity_loss(e3, ligand, poi, predicted, target, ligand_data,
                             conditions, scale):
    """Match local distance rates plus Cartesian linker motion to the flow target.

    Uses the same noisy state and coupling as the velocity loss. A non-native
    intermediate state is not incorrectly forced to satisfy endpoint contacts.
    """
    pv_l, pv_p = cartesian_velocities(ligand, poi, predicted, ligand_data)
    tv_l, tv_p = cartesian_velocities(ligand, poi, target, ligand_data)
    delta_l, delta_p = (pv_l - tv_l) / scale, (pv_p - tv_p) / scale
    result = .5 * (delta_l.square().mean() + delta_p.square().mean())
    # Interface distance-rate supervision uses current contacts, with extra
    # emphasis on coarse overlaps. The target remains the same flow tangent.
    for left, right, vl, vr, threshold in (
            (e3, poi, torch.zeros_like(e3), delta_p, 3.),
            (e3, ligand, torch.zeros_like(e3), delta_l, 2.5),
            (poi, ligand, delta_p, delta_l, 2.5)):
        displacement = right[None] - left[:, None]
        distance = displacement.norm(dim=-1)
        error = ((vr[None] - vl[:, None]) * displacement
                 / distance.clamp_min(1e-6)[..., None]).sum(-1)
        weight = torch.exp(-distance / 8.) * (1. + torch.relu(1. - distance / threshold))
        result = result + (weight * error.square()).sum() / weight.sum().clamp_min(1.) / 3.
    if conditions is None:
        return result
    from .binary_geometry import select_binary_mapping
    for side, protein in enumerate((e3, poi), start=1):
        displacement = ligand[:, None] - protein[None]
        distance = displacement.norm(dim=-1)
        alternative = torch.as_tensor(conditions[f'binary{side}'], device=ligand.device,
                                      dtype=ligand.dtype)
        chosen, _ = select_binary_mapping(distance, alternative)
        weight = chosen[..., 2] * torch.exp(-chosen[..., 0] / 10.)
        relative = delta_l[:, None] - (delta_p[None] if side == 2 else 0.)
        rate_error = (relative * displacement / distance.clamp_min(1e-6)[..., None]).sum(-1)
        result = result + (weight * rate_error.square()).sum() / weight.sum().clamp_min(1.)
    return result
