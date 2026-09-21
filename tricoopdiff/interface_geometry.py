"""Typed local interface messages and equivariant coarse-grained directions."""

import torch
from torch import nn


class InterfaceGeometryEncoder(nn.Module):
    """Read current interfaces without reference contacts or native coordinates.

    Proteins use C-alpha nodes: overlap channels are coarse geometry cues, not
    atomistic van der Waals energies. Near-contact edges include every pair
    within 12 A plus per-node nearest neighbors for separated components.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0., 12., 8))
        self.pair_type = nn.Embedding(3, hidden_dim)
        self.message = nn.Sequential(nn.Linear(2 * hidden_dim + 9, hidden_dim),
                                     nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.direction = nn.Linear(hidden_dim, 1)

    def forward(self, graph, chemical_nodes, *, return_full_directions=False):
        coords, types = graph['node_coords'], graph['node_types']
        nodes = torch.zeros_like(chemical_nodes)
        force = torch.zeros_like(coords)
        degree = coords.new_zeros(len(coords))
        ligand_ids = torch.where(types == 2)[0]
        ligand_forces = coords.new_zeros((2, len(ligand_ids), 3))
        for pair_index, (left_type, right_type) in enumerate(((0, 2), (1, 2), (0, 1))):
            left, right = torch.where(types == left_type)[0], torch.where(types == right_type)[0]
            distance = torch.cdist(coords[left], coords[right])
            selected = distance < 12.
            selected.scatter_(1, distance.topk(min(4, len(right)), largest=False, dim=1).indices, True)
            selected.scatter_(0, distance.topk(min(4, len(left)), largest=False, dim=0).indices, True)
            i, j = selected.nonzero(as_tuple=True)
            a, b = left[i], right[j]
            d = distance[i, j]
            threshold = 3. if pair_index == 2 else 2.5
            overlap = torch.relu(1. - d / threshold)
            rbf = torch.exp(-.5 * ((d[:, None] - self.centers) / 2.).square())
            message = self.message(torch.cat((chemical_nodes[a], chemical_nodes[b],
                                              rbf, overlap[:, None]), dim=-1))
            message = message + self.pair_type.weight[pair_index]
            weight = torch.exp(-d / 12.)
            weighted = message * weight[:, None]
            nodes = nodes.index_add(0, a, weighted).index_add(0, b, weighted)
            degree = degree.index_add(0, a, weight).index_add(0, b, weight)
            # Equal and opposite pair directions. Learned coefficients allow
            # attractive/contact corrections as well as overlap resolution.
            coefficient = overlap + torch.tanh(self.direction(message).squeeze(-1))
            vector = (coords[b] - coords[a]) / d.clamp_min(1e-6)[:, None]
            vector = vector * (coefficient * weight)[:, None]
            force = force.index_add(0, a, -vector).index_add(0, b, vector)
            if pair_index < 2:
                ligand_forces[pair_index] = torch.zeros_like(coords[ligand_ids]).index_add(0, j, vector)
        nodes = nodes / degree.clamp_min(1.)[:, None]
        force = force / degree.clamp_min(1.)[:, None]
        ligand_forces = ligand_forces / degree[ligand_ids].clamp_min(1.)[None, :, None]
        center = coords[ligand_ids].mean(0)
        basis = []
        for group in (2, 1):
            mask = types == group
            basis.append(force[mask].mean(0))
            basis.append(torch.cross((coords[mask] - center) / 10., force[mask], dim=-1).mean(0))
        result = (nodes, torch.stack(basis).unsqueeze(0), ligand_forces)
        return (*result, force) if return_full_directions else result
