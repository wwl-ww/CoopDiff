"""Sparse local interaction propagation over existing ternary node embeddings."""

import torch
from torch import nn


def interface_edges(coords, types, ligand_bonds, *, neighbors=4, cutoff=12.):
    """Directed covalent and cross-component edges; protein nodes remain C-alpha.

    Relation IDs: ligand bond, E3-ligand, POI-ligand, protein interface.
    Nearest contacts are selected in both directions, then deduplicated.
    Exact distance ties at a kNN boundary retain the usual index ambiguity.
    """
    if neighbors < 1 or cutoff <= 0:
        raise ValueError('positive neighbor count and cutoff required')
    ids = [(types == component).nonzero().flatten() for component in range(3)]
    rows = []
    if ligand_bonds.numel():
        bonds = ids[2][ligand_bonds]
        rows.extend((torch.stack((bonds[0], bonds[1], torch.zeros_like(bonds[0])), 1),
                     torch.stack((bonds[1], bonds[0], torch.zeros_like(bonds[0])), 1)))
    for a, b, relation in ((0, 2, 1), (1, 2, 2), (0, 1, 3)):
        distance = torch.cdist(coords[ids[a]], coords[ids[b]])
        for left, right, matrix in ((ids[a], ids[b], distance), (ids[b], ids[a], distance.T)):
            values, nearest = matrix.topk(min(neighbors, len(right)), largest=False, dim=1)
            source = left[:, None].expand_as(nearest)
            target = right[nearest]
            valid = values < cutoff
            source, target = source[valid], target[valid]
            kind = torch.full_like(source, relation)
            rows.extend((torch.stack((source, target, kind), 1), torch.stack((target, source, kind), 1)))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long, device=coords.device), types.new_empty(0)
    edges = torch.unique(torch.cat(rows), dim=0)
    return edges[:, :2].T, edges[:, 2]


class InterfaceMessagePassing(nn.Module):
    """Propagate local chemistry/contact patterns before component pooling.

    This supplements, rather than replaces, the existing handedness-sensitive
    geometry encoder. Distances alone in this branch are reflection invariant.
    """

    def __init__(self, hidden_dim, layers=2, neighbors=4, cutoff=12., readout='pooled'):
        super().__init__()
        if layers < 1 or neighbors < 1 or cutoff <= 0:
            raise ValueError('invalid interface message-passing configuration')
        self.neighbors, self.cutoff = neighbors, float(cutoff)
        if readout not in ('pooled', 'contacts', 'contacts-independent'):
            raise ValueError('unknown interface readout')
        if readout != 'pooled' and hidden_dim % 4:
            raise ValueError('contact attention requires hidden_dim divisible by four')
        self.readout_mode = readout
        self.register_buffer('centers', torch.linspace(0., cutoff, 16))
        self.relation_embedding = nn.Embedding(4, hidden_dim)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(3 * hidden_dim + 16),
                          nn.Linear(3 * hidden_dim + 16, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
                          nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.readout = nn.Sequential(nn.LayerNorm(3 * hidden_dim),
                                     nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU())
        if readout != 'pooled':
            self.contact_project = nn.Sequential(nn.LayerNorm(3 * hidden_dim + 16),
                                                nn.Linear(3 * hidden_dim + 16, hidden_dim), nn.SiLU())
            # Instantiate identically in both contact arms; independent skips interaction.
            self.contact_interaction = nn.TransformerEncoderLayer(
                hidden_dim, 4, dim_feedforward=2 * hidden_dim, dropout=0.,
                batch_first=True, norm_first=True)
            self.contact_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
            self.contact_attention = nn.MultiheadAttention(hidden_dim, 4, dropout=0., batch_first=True)
            self.contact_norm = nn.LayerNorm(hidden_dim)
            self.readout.requires_grad_(False)
            if readout == 'contacts-independent':
                self.contact_interaction.requires_grad_(False)

    def forward(self, features, coords, types, ligand_bonds):
        edges, relations = interface_edges(coords, types, ligand_bonds,
                                           neighbors=self.neighbors, cutoff=self.cutoff)
        source, target = edges
        if not len(source):
            return features.new_zeros(1, features.shape[1])
        distances = torch.linalg.vector_norm(coords[source] - coords[target], dim=-1)
        rbf = torch.exp(-.5 * ((distances[:, None] - self.centers) / (self.cutoff / 15)) ** 2)
        relation = self.relation_embedding(relations)
        degree = torch.bincount(target, minlength=len(features)).to(features.dtype)[:, None]
        active = degree[:, 0] > 0
        current = features
        for message, update in zip(self.messages, self.updates):
            values = message(torch.cat((current[source], current[target], relation, rbf), -1))
            aggregate = torch.zeros_like(current).index_add(0, target, values) / degree.clamp_min(1)
            residual = update(torch.cat((current, aggregate), -1))
            current = current + residual * active[:, None]
        if self.readout_mode != 'pooled':
            selected = []
            # At most 32 undirected contacts per physical interface, no native labels.
            # Bond edges affect messages above but are not interface readout tokens.
            for kind in (1, 2, 3):
                ids = ((relations == kind) & (types[source] < types[target])).nonzero().flatten()
                if len(ids):
                    selected.append(ids[torch.argsort(distances[ids], stable=True)[:32]])
            if not selected:
                return features.new_zeros(1, features.shape[1])
            ids = torch.cat(selected)
            tokens = self.contact_project(torch.cat(
                (current[source[ids]], current[target[ids]], relation[ids], rbf[ids]), -1))[None]
            if self.readout_mode == 'contacts':
                tokens = self.contact_interaction(tokens)
            pooled, _ = self.contact_attention(self.contact_query, tokens, tokens, need_weights=False)
            return self.contact_norm(pooled[:, 0])
        pooled = []
        for component in range(3):
            mask = active & (types == component)
            pooled.append(current[mask].mean(0) if bool(mask.any()) else current.new_zeros(current.shape[1]))
        return self.readout(torch.cat(pooled)[None])
