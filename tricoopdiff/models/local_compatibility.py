"""Compact-support ternary compatibility corrections in rigid/torsion DOFs.

The scalar cost is learned compatibility, not physical energy. Protein pairs
and chemically bridged ligand contacts are the only learned terms.
"""
import torch
from torch import nn


LOCAL_COMPATIBILITY_CONTRACT = 'protected_binary_local_cost_v1'


def _skew(value):
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*value.shape[:-1], 3, 3)


def motion_jacobian(coords, types, graph, scales, no_torsion=False):
    """Actual forward-clock Cartesian tangent [N,3,12+K], E3 held fixed.

    Includes pi/scales and the symmetric horizontal torsion spin subtraction of
    SDPIFlow. The reverse decoder therefore applies minus this tangent.
    """
    ligand = coords[types == 2]
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']].T.to(coords.device)
    if no_torsion:
        bonds = bonds[:0]
    width = 12 + len(bonds)
    result = coords.new_zeros(len(coords), 3, width)
    for component, offset, scale in ((2, 0, scales[0]), (1, 6, scales[1])):
        selected = types == component
        points = coords[selected]
        centered = points - points.mean(0)
        result[selected, :, offset:offset + 3] = torch.eye(3, device=coords.device, dtype=coords.dtype) * scale
        result[selected, :, offset + 3:offset + 6] = -torch.pi * _skew(centered)
    if len(bonds):
        masks = graph['lig_mask_rotate']
        if isinstance(masks, (tuple, list)):
            masks = masks[0]
        masks = torch.as_tensor(masks, device=coords.device, dtype=coords.dtype)
        if masks.shape != (len(bonds), len(ligand)):
            raise ValueError('torsion masks must have shape [K,N_ligand]')
        axis = ligand[bonds[:, 0]] - ligand[bonds[:, 1]]
        norm = axis.norm(dim=-1, keepdim=True)
        if bool((norm <= 1e-10).any()):
            raise ValueError('degenerate local compatibility torsion bond')
        axis = axis / norm
        pivot = ligand[bonds].mean(1)
        tangent = torch.cross(axis[:, None].expand(-1, len(ligand), -1),
                              ligand[None] - pivot[:, None], dim=-1) * (masks - .5)[..., None]
        centered = (ligand - ligand.mean(0)).double()
        identity = torch.eye(3, device=coords.device, dtype=torch.double)
        inertia = centered.square().sum() * identity - centered.T @ centered
        ridge = inertia.trace().clamp_min(torch.finfo(torch.double).tiny) * 1e-6
        torque = torch.cross(centered[None].expand_as(tangent), tangent.double(), dim=-1).sum(1)
        spin = torch.linalg.solve(inertia + ridge * identity, torque.T).T
        tangent = tangent.double() - tangent.double().mean(1, keepdim=True)
        tangent = tangent - torch.cross(spin[:, None].expand_as(tangent),
                                         centered[None].expand_as(tangent), dim=-1)
        result[types == 2, :, 12:] = tangent.to(coords).permute(1, 2, 0) * torch.pi
    return result


class LocalCompatibilityCorrection(nn.Module):
    """Shared local scalar networks followed by fixed geometric mobility.

    Graph keys: node_coords [N,3], node_types (E3=0, POI=1, ligand=2),
    node_features [N,input_dim], protein_local_geometry [N_protein,32], and
    ligand edge/mask fields. No graph identity, global latent or soft anchor.
    """

    def __init__(self, input_dim, hidden_dim=32, cutoff=8., mobility=.1, action_limit=.25):
        super().__init__()
        if cutoff <= 0 or mobility <= 0 or action_limit <= 0:
            raise ValueError('cutoff, mobility and action limit must be positive')
        self.cutoff, self.mobility, self.action_limit = float(cutoff), float(mobility), float(action_limit)
        from tricoopdiff.models.consensus import LigandFeatureEncoder
        from tricoopdiff.diffdock.constants import rec_residue_feature_dims
        self.amino_acid_embedding = nn.Embedding(rec_residue_feature_dims[0][0], hidden_dim)
        self.protein_encoder = nn.Sequential(nn.Linear(input_dim - 1 + 32, hidden_dim), nn.SiLU())
        self.ligand_encoder = LigandFeatureEncoder(hidden_dim)
        self.pair_head = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.triplet_head = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        for head in (self.pair_head, self.triplet_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _radial(self, left, right):
        distance = ((left[:, None] - right[None]).square().sum(-1) + 1e-12).sqrt()
        radius = distance / self.cutoff
        # Wendland C2: value, first and second derivatives vanish at support edge.
        envelope = (1 - radius).clamp_min(0).pow(4) * (1 + 4 * radius)
        return envelope, radius

    def costs(self, graph, coords=None):
        """Return local pair/triplet sums; absent contacts contribute exact zero."""
        coords = graph['node_coords'] if coords is None else coords
        types, raw = graph['node_types'], graph['node_features']
        geometry = graph['protein_local_geometry']
        if geometry.shape != (int((types != 2).sum()), 32):
            raise ValueError('protein local geometry must be [N_protein,32]')
        if raw.is_inference(): raw = raw.clone()
        if geometry.is_inference(): geometry = geometry.clone()
        protein_raw = raw[types != 2]
        amino_acids = protein_raw[:, 0].long().clamp(0, self.amino_acid_embedding.num_embeddings - 1)
        protein = self.amino_acid_embedding(amino_acids) + self.protein_encoder(
            torch.cat((protein_raw[:, 1:], geometry.to(raw)), -1))
        proteins = [protein[types[types != 2] == component] for component in (0, 1)]
        positions = [coords[types == component] for component in (0, 1, 2)]
        contact, radius = self._radial(positions[0], positions[1])
        # Materialize learned pair features only inside compact support. The
        # C2 envelope makes neighbor-list changes continuous through second order.
        left, right = (radius.detach() < 1).nonzero(as_tuple=True)
        first, second = proteins[0][left], proteins[1][right]
        pair_features = torch.cat((first + second, first * second, radius[left, right, None]), -1)
        pair_cost = (contact[left, right] * self.pair_head(pair_features).squeeze(-1).tanh()).sum() / 16.
        local, occupancy = [], []
        for component in (0, 1):
            weight, _ = self._radial(positions[2], positions[component])
            count = weight.sum(-1, keepdim=True)
            local.append(weight @ proteins[component] / (1 + count))
            occupancy.append(count[:, 0] / (1 + count[:, 0]))
        ligand = self.ligand_encoder(raw[types == 2])
        # Fixed covalent communication bridges distinct ligand binding ends.
        # The graph resolvent is block diagonal across disconnected molecules;
        # it cannot manufacture contact evidence on a missing interface.
        count = len(ligand)
        edges = graph['lig_edge_index'].to(device=coords.device, dtype=torch.long)
        adjacency = torch.zeros(count, count, device=coords.device, dtype=torch.double)
        if edges.numel():
            adjacency[edges[0], edges[1]] = 1.
            adjacency[edges[1], edges[0]] = 1.
        adjacency.fill_diagonal_(0.)
        identity = torch.eye(count, device=coords.device, dtype=torch.double)
        laplacian = torch.diag(adjacency.sum(-1)) - adjacency
        communication = torch.linalg.solve(identity + laplacian, identity).to(coords)
        propagated = [communication @ value for value in local]
        propagated_occupancy = [communication @ value for value in occupancy]
        triplet_cost = coords.new_zeros(())
        for own, partner in ((0, 1), (1, 0)):
            features = torch.cat((ligand, local[own] + propagated[partner],
                                  local[own] * propagated[partner]), -1)
            coefficient = self.triplet_head(features).squeeze(-1).tanh()
            triplet_cost = triplet_cost + .5 * (
                occupancy[own] * propagated_occupancy[partner] * coefficient).sum()
        return pair_cost, triplet_cost

    def forward(self, graph, scales, no_torsion=False):
        outer_grad = torch.is_grad_enabled()
        # Evaluation must still differentiate the scalar with respect to geometry.
        # Existing rollout coordinates retain their graph when one is present.
        with torch.inference_mode(False), torch.enable_grad():
            original = graph['node_coords']
            coords = original if original.requires_grad and not original.is_inference() else original.detach().clone().requires_grad_(True)
            pair_cost, triplet_cost = self.costs(graph, coords)
            cost = pair_cost + triplet_cost
            gradient = torch.autograd.grad(cost, coords, create_graph=outer_grad)[0]
            jacobian = motion_jacobian(coords, graph['node_types'], graph, scales, no_torsion)
            matrix, force = jacobian.reshape(-1, jacobian.shape[-1]).double(), gradient.flatten().double()
            # Pull back the Cartesian cost gradient to normalized DOFs with
            # a fixed Euclidean mobility. Dividing by whole-body inertia here
            # would dilute local corrections as distant residues are added.
            # Positive forward-clock gradient gives descent under reverse -dt.
            vector = self.mobility * (matrix.T @ force)
            blocks = vector[:12].reshape(4, 3).square().sum(-1)
            sizes = torch.cat((blocks, vector[12:].square()))
            maximum = (sizes.max() + 1e-12).sqrt()
            vector = vector * (self.action_limit / (self.action_limit + maximum))
            vector = vector.to(original)
            diagnostics = {'pair_cost': pair_cost.detach(), 'triplet_cost': triplet_cost.detach(),
                           'cost_directional_derivative': -(gradient * torch.einsum('ncd,d->nc', jacobian, vector)).sum().detach(),
                           'mobility': self.mobility, 'cutoff': self.cutoff}
        if not outer_grad:
            vector, cost = vector.detach(), cost.detach()
        return {'vector': vector, 'local_cost': cost, 'diagnostics': diagnostics}
