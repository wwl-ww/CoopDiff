"""Chart-origin-independent linker geometry for coordinated torsion actions."""

import torch


@torch.no_grad()
def linker_geometry_features(coordinates, plan, edge_index, length_scale):
    """Return [K,9] local Fourier geometry and [K,K,4] joint geometry.

    Average over every valid bonded neighbor quartet, rather than choosing an
    arbitrary lowest atom index. Three harmonics retain symmetric rotor modes.
    No absolute coordinates, reference q origin, or native structure is used.
    """
    if coordinates.shape != (plan.atom_count, 3) or not torch.isfinite(coordinates).all():
        raise ValueError('invalid linker geometry coordinates')
    if (edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long
            or bool((edge_index < 0).any()) or bool((edge_index >= plan.atom_count).any())):
        raise ValueError('invalid linker geometry bonds')
    scale = torch.as_tensor(length_scale, device=coordinates.device, dtype=coordinates.dtype)
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError('linker geometry needs a finite positive length scale')
    count = len(plan.bonds)
    if not count:
        return coordinates.new_zeros((0, 9)), coordinates.new_zeros((0, 0, 4))
    neighbors = [set() for _ in range(plan.atom_count)]
    for left, right in edge_index.detach().cpu().T.tolist():
        neighbors[left].add(right)
        neighbors[right].add(left)
    features, axes, centers, masks = [], [], [], []
    for (u, v), moving in zip(plan.bonds, plan.moving_atoms):
        axis = coordinates[u] - coordinates[v]
        if axis.norm() < 1e-8:
            raise ValueError('degenerate linker action axis')
        axis = axis / axis.norm()
        axes.append(axis)
        centers.append((coordinates[u] + coordinates[v]) / 2)
        mask = coordinates.new_zeros(plan.atom_count)
        mask[list(moving)] = 1
        masks.append(mask)
        left = sorted(neighbors[u] - {v} - set(moving))
        right = sorted((neighbors[v] - {u}) & set(moving))
        quartets = []
        for a in left:
            for b in right:
                av, bv = coordinates[a] - coordinates[u], coordinates[b] - coordinates[v]
                ap, bp = av - av.dot(axis) * axis, bv - bv.dot(axis) * axis
                if min(float(ap.norm()), float(bp.norm())) < 1e-8:
                    continue
                angle = torch.atan2(torch.linalg.cross(ap, bp).dot(axis), ap.dot(bp))
                harmonics = torch.arange(1, 4, dtype=coordinates.dtype, device=coordinates.device) * angle
                quartets.append(torch.cat((harmonics.sin(), harmonics.cos(),
                    (av.dot(axis) / av.norm()).reshape(1), (bv.dot(axis) / bv.norm()).reshape(1))))
        local = torch.stack(quartets).mean(0) if quartets else coordinates.new_zeros(8)
        valid = len(quartets) / max(1, len(left) * len(right))
        features.append(torch.cat((local, coordinates.new_tensor([valid]))))
    axes, centers, masks = torch.stack(axes), torch.stack(centers), torch.stack(masks)
    distance = torch.cdist(centers, centers) / scale
    intersection = masks @ masks.T
    union = masks.sum(1)[:, None] + masks.sum(1)[None, :] - intersection
    follows = coordinates.new_tensor(plan.moves_poi)
    pairs = torch.stack((axes @ axes.T, distance / (1 + distance),
                         intersection / union.clamp_min(1.), follows[:, None] * follows[None, :]), -1)
    return torch.stack(features), pairs
