"""Dynamic binary-pose messages in the current ternary coordinate frame."""

import torch
from torch import nn


def select_binary_mapping(distance, alternatives):
    """Select one complete legal map using only the current predicted geometry.

    All atom/residue pairs vote for one assignment, never independent pairwise
    minima. Builders canonically sort maps so equal scores have stable ties.
    Selection is discrete; gradients flow through the chosen map's errors.
    """
    low, high, confidence = alternatives.unbind(-1)
    if (alternatives.ndim != 4 or not len(alternatives)
            or not torch.isfinite(alternatives).all() or (low < 0).any()
            or (high != low).any() or (confidence < 0).any() or (confidence > 1).any()):
        raise ValueError('binary alternatives must contain finite exact mapped distances')
    available = confidence.sum((1, 2)) > 0
    error = distance.unsqueeze(0) - low
    weight = confidence * torch.exp(-low / 10.)
    cost = (torch.log1p(error.square()) * weight).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1e-6)
    if available.any():
        cost = cost.masked_fill(~available, float('inf'))
    index = cost.detach().argmin()
    return alternatives[index], index


class BinaryGeometryEncoder(nn.Module):
    """Encode receptor-indexed distance errors and propagate through linker bonds.

    Targets are independent protein-local distances, never cross-protein native
    geometry. Symmetric atoms retain complete alternative assignments.
    Vectors follow the current frame; scalars are invariant under rigid motion.
    """

    def __init__(self, hidden_dim, layers=3):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.bonds = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim, bias=False),
                          nn.SiLU(), nn.Linear(hidden_dim, hidden_dim, bias=False))
            for _ in range(layers)])

    def forward(self, graph, features):
        coords, types = graph['node_coords'], graph['node_types']
        ligand = coords[types == 2]
        nodes = coords.new_zeros((len(coords), self.pair[-1].out_features))
        ligand_messages = torch.zeros_like(nodes[types == 2])
        vectors = []
        forces = []
        for side in (0, 1):
            protein = coords[types == side]
            displacement = ligand[:, None] - protein[None]
            distance = displacement.norm(dim=-1)
            target, _ = select_binary_mapping(distance, features[f'binary{side + 1}'])
            low, high, confidence = target.unbind(-1)
            if ((low < 0).any() or (high < low).any()
                    or (confidence < 0).any() or (confidence > 1).any()):
                raise ValueError('invalid binary distance interval or confidence')
            error = distance - distance.clamp(min=low, max=high)
            # Retain nearby constraints in the supplied binary even when the
            # current candidate is far away. No candidate-distance cutoff.
            weight = confidence * torch.exp(-low / 10.)
            inputs = torch.stack((torch.log1p(distance), torch.log1p(low),
                                  torch.log1p(high), torch.tanh(error / 10.),
                                  confidence), dim=-1)
            messages = self.pair(inputs) * weight[..., None]
            ligand_messages = ligand_messages + messages.sum(1) / weight.sum(1).clamp_min(1.)[:, None]
            nodes[types == side] = messages.sum(0) / weight.sum(0).clamp_min(1.)[:, None]
            # Bounded restoring directions add geometry unavailable in expert
            # proposals. Learned coefficients choose their flow sign and scale.
            force = (-torch.tanh(error / 10.)[..., None] * displacement
                     / distance.clamp_min(1e-6)[..., None] * weight[..., None]).sum(1)
            force = force / weight.sum(1).clamp_min(1.)[:, None]
            forces.append(force)
            center = ligand.mean(0)
            torque = torch.cross((ligand - center) / 10., force, dim=-1).mean(0)
            vectors.extend((force.mean(0), torque))
        edge = graph['lig_edge_index']
        if edge is not None and edge.numel():
            # Symmetrize and deduplicate: callers may supply one or both bond directions.
            edge = torch.unique(torch.cat((edge, edge.flip(0)), dim=1), dim=1)
            source, destination = edge
            lengths = torch.log1p((ligand[source] - ligand[destination]).norm(dim=-1))[:, None]
            available = (features['binary1'][..., 2].sum()
                         + features['binary2'][..., 2].sum() > 0).to(coords)
            for layer in self.bonds:
                message = layer(torch.cat((ligand_messages[source],
                                           ligand_messages[destination], lengths), -1)) * available
                update = torch.zeros_like(ligand_messages).index_add(0, destination, message)
                degree = coords.new_zeros(len(ligand)).index_add(
                    0, destination, coords.new_ones(len(destination)))
                ligand_messages = ligand_messages + update / degree.clamp_min(1.)[:, None]
        nodes[types == 2] = ligand_messages
        return nodes, torch.stack(vectors).unsqueeze(0), torch.stack(forces)


def binary_torsion_directions(graph, forces):
    """Project both end constraints onto the actual shared rotatable bonds.

    The moving-atom mask and bond orientation follow the sampler contract.
    A single shared torsion must reconcile both protein-local pose constraints.
    """
    ligand = graph['node_coords'][graph['node_types'] == 2]
    from .structure_velocity import torsion_tangents
    tangent = torsion_tangents(ligand, graph['lig_edge_index'], graph['lig_edge_mask'],
                               graph['lig_mask_rotate'])
    projected = torch.einsum('tnc,snc->ts', tangent, forces)
    scale = tangent.square().sum((1, 2)).clamp_min(1.)
    return torch.tanh(projected / scale[:, None])
