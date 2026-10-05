"""Frozen binary reads and trainable ternary motion.

Production model-4.5 conditions frozen binary reads on the other interface,
preserves ternary relations, and solves joint rigid/torsion velocities.
Historical motion implementations are maintained on the model-v4 worktree.
All vectors use the E3 frame; native targets never enter the model.
"""

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from tricoopdiff.inputs.graphs import build_binary_graph, build_trifuse_graph, crop_receptor_to_ligand
from tricoopdiff.models.expert import _set_time_embeddings
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.diffdock.constants import lig_feature_dims


PACKED_BATCH_LIMITS = {'nodes': 4096, 'pairs': 300000}


def packed_groups(items, limits=PACKED_BATCH_LIMITS):
    """Bound ragged graph residency; one oversized complex runs alone."""
    group, nodes, pairs = [], 0, 0
    for item in items:
        a, b, l = [item[key]['coords'].shape[-2] for key in ('e3_data', 'poi_data', 'ligand_data')]
        n, p = a + b + 2 * l, 2 * a * b + 2 * l * (a + b) + (32 + l) ** 2
        if group and (nodes + n > limits['nodes'] or pairs + p > limits['pairs']):
            yield group
            group, nodes, pairs = [], 0, 0
        group.append(item)
        nodes, pairs = nodes + n, pairs + p
    if group:
        yield group


class LigandFeatureEncoder(nn.Module):
    """Embed DiffDock categorical ligand features without treating IDs as floats."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        categorical_dims, scalar_dim = lig_feature_dims
        if scalar_dim:
            raise ValueError("TriFuse expects categorical-only DiffDock ligand features")
        self.categorical_dims = tuple(categorical_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, hidden_dim) for size in self.categorical_dims
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] < len(self.embeddings):
            raise ValueError(
                f"ligand features require {len(self.embeddings)} categorical columns"
            )
        encoded = features.new_zeros(features.shape[0], self.norm.normalized_shape[0])
        for column, (embedding, size) in enumerate(zip(
            self.embeddings, self.categorical_dims,
        )):
            indices = features[:, column].long().clamp(0, size - 1)
            encoded = encoded + embedding(indices)
        return self.norm(encoded / math.sqrt(len(self.embeddings)))


class BondContextEncoder(nn.Module):
    """Encode directed torsions, with chemistry/fragment context in v2 only.

    Signed phase averages over every valid adjacent atom pair, rather than
    selecting an index-dependent dihedral or an arbitrary world-space axis.
    """

    def __init__(self, hidden_dim: int, n_rbf: int = 16,
                 max_distance: float = 8.0, chemical_context: bool = False,
                 phase_harmonics: int = 3, symmetric: bool = False):
        super().__init__()
        self.chemical_context = bool(chemical_context)
        self.symmetric = symmetric
        if isinstance(phase_harmonics, bool) or not isinstance(phase_harmonics, int) or phase_harmonics < 1:
            raise ValueError('phase_harmonics must be a positive integer')
        self.phase_harmonics = phase_harmonics
        self.max_distance = float(max_distance)
        self.register_buffer(
            "rbf_centers", torch.linspace(0.0, max_distance, n_rbf),
        )
        self.rbf_width = max_distance / max(n_rbf - 1, 1)
        input_dim = (hidden_dim * 8 + n_rbf + 15 + 2 * (phase_harmonics - 1) if self.chemical_context
                     else hidden_dim * 4 + n_rbf + 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, ligand_tokens: torch.Tensor, ligand_coords: torch.Tensor,
                edge_index: torch.Tensor, edge_mask: torch.Tensor,
                mask_rotate, global_context: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        args = (ligand_tokens, ligand_coords, edge_index, edge_mask, mask_rotate,
                global_context, edge_attr)
        token = self._directed_forward(*args)
        if not self.symmetric or edge_mask is None or not bool(edge_mask.any()):
            return token
        masks = torch.as_tensor(mask_rotate, device=ligand_coords.device, dtype=torch.bool)
        opposite = self._directed_forward(ligand_tokens, ligand_coords, edge_index.flip(0),
                                           edge_mask, ~masks, global_context, edge_attr)
        return (token + opposite) * .5

    def _directed_forward(self, ligand_tokens: torch.Tensor, ligand_coords: torch.Tensor,
                edge_index: torch.Tensor, edge_mask: torch.Tensor,
                mask_rotate, global_context: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None or edge_mask is None or not bool(edge_mask.any()):
            return global_context.new_zeros(global_context.shape[0], 0,
                                            global_context.shape[-1])
        bonds = edge_index[:, edge_mask].T.long()
        left = ligand_tokens[bonds[:, 0]]
        right = ligand_tokens[bonds[:, 1]]
        lengths = torch.linalg.vector_norm(
            ligand_coords[bonds[:, 0]] - ligand_coords[bonds[:, 1]], dim=-1,
        )
        distance_rbf = torch.exp(
            -0.5 * ((lengths[:, None] - self.rbf_centers[None, :])
                    / self.rbf_width) ** 2
        )
        if mask_rotate is None:
            if self.chemical_context:
                raise ValueError('chemical torsion context requires directed boolean fragment masks')
            fragment = lengths.new_zeros(len(bonds), 2)
        else:
            rotate = torch.as_tensor(
                mask_rotate, device=lengths.device, dtype=lengths.dtype,
            )
            if rotate.ndim == 3:
                rotate = rotate[0]
            if rotate.shape[0] != len(bonds):
                raise ValueError("rotatable-bond and fragment-mask counts differ")
            fraction = rotate.mean(dim=-1)
            fragment = torch.stack((fraction, 1.0 - fraction), dim=-1)
        global_per_bond = global_context.expand(len(bonds), -1)
        pieces = [
            left, right, (left - right).abs(), global_per_bond,
            distance_rbf, fragment,
        ]
        if self.chemical_context:
            if (edge_attr is None or edge_attr.shape != (edge_index.shape[1], 4)
                    or not torch.isfinite(edge_attr).all()):
                raise ValueError('chemical torsion context requires finite [E,4] bond attributes')
            if isinstance(mask_rotate, (list, tuple)):
                if len(mask_rotate) != 1:
                    raise ValueError('chemical torsion context requires one ligand graph')
                mask_rotate = mask_rotate[0]
            rotate = torch.as_tensor(mask_rotate, device=ligand_tokens.device)
            if rotate.ndim == 3 and rotate.shape[0] == 1:
                rotate = rotate[0]
            if (rotate.dtype != torch.bool or rotate.shape != (len(bonds), len(ligand_tokens))
                    or bool(rotate.gather(1, bonds[:, :1]).any())
                    or not bool(rotate.gather(1, bonds[:, 1:]).all())):
                raise ValueError('chemical torsion context requires directed boolean fragment masks')
            cross_fragment = rotate[:, edge_index[0]] != rotate[:, edge_index[1]]
            axis_edges = ((edge_index[0][None] == bonds[:, :1])
                          & (edge_index[1][None] == bonds[:, 1:]))
            axis_edges |= ((edge_index[1][None] == bonds[:, :1])
                           & (edge_index[0][None] == bonds[:, 1:]))
            if bool((cross_fragment & ~axis_edges).any()):
                raise ValueError('torsion fragment mask cuts a bond other than its axis')
            side_weights = [side.to(ligand_tokens.dtype) for side in (rotate, ~rotate)]
            side_weights = [weight / weight.sum(-1, keepdim=True) for weight in side_weights]
            pieces.extend(weight @ ligand_tokens for weight in side_weights)

            # Indexed molecular graphs contain both directions. Pad only to
            # the maximum chemical degree, never materialize K*N*N pairs.
            order = edge_index[0].argsort(stable=True)
            sources, destinations = edge_index[:, order]
            degree = torch.bincount(sources, minlength=len(ligand_tokens))
            slots = torch.arange(len(sources), device=sources.device) - (degree.cumsum(0) - degree)[sources]
            neighbors_by_atom = sources.new_zeros(len(ligand_tokens), int(degree.max()))
            neighbor_valid = torch.zeros_like(neighbors_by_atom, dtype=torch.bool)
            neighbors_by_atom[sources, slots] = destinations
            neighbor_valid[sources, slots] = True
            axis = ligand_coords[bonds[:, 1]] - ligand_coords[bonds[:, 0]]
            axis = axis / lengths[:, None].clamp_min(.1 if self.phase_harmonics > 1 else 1e-8)
            neighboring_directions, counts, pair_directions, pair_weights, pair_confidence = [], [], [], [], []
            for end in (0, 1):
                indices = neighbors_by_atom[bonds[:, end]]
                neighbors = neighbor_valid[bonds[:, end]] & (indices != bonds[:, 1 - end, None])
                weights = neighbors.to(ligand_tokens.dtype)
                pieces.append((weights[..., None] * ligand_tokens[indices]).sum(1)
                              / weights.sum(-1, keepdim=True).clamp_min(1.))
                delta = ligand_coords[indices] - ligand_coords[bonds[:, end], None]
                transverse = delta - (delta * axis[:, None]).sum(-1, keepdim=True) * axis[:, None]
                if self.phase_harmonics == 1:
                    norm = torch.linalg.vector_norm(transverse, dim=-1)
                    valid = neighbors & (norm > 1e-8)
                    count = valid.sum(-1, keepdim=True)
                    direction = transverse / norm[..., None].clamp_min(1e-8)
                    neighboring_directions.append((direction * valid[..., None]).sum(1)
                                                  / count.clamp_min(1))
                    counts.append(count)
                else:
                    squared = transverse.square().sum(-1)
                    pair_directions.append(transverse / torch.sqrt(squared[..., None] + .01))
                    affinity = (ligand_tokens[indices] * ligand_tokens[bonds[:, end], None]).sum(-1)
                    affinity = affinity / math.sqrt(ligand_tokens.shape[-1])
                    affinity = affinity.masked_fill(~neighbors, -torch.inf)
                    affinity = torch.where(neighbors.any(-1, keepdim=True), affinity, torch.zeros_like(affinity))
                    pair_weights.append(torch.softmax(affinity, -1) * neighbors)
                    pair_confidence.append(squared / (squared + .01))
            if self.phase_harmonics == 1:
                a, b = neighboring_directions
                phase = torch.stack(((torch.linalg.cross(a, b, dim=-1) * axis).sum(-1),
                                     (a * b).sum(-1),
                                     ((counts[0] > 0) & (counts[1] > 0)).squeeze(-1).to(axis.dtype)), -1)
            else:
                # Aggregate powers of each adjacent-pair phase BEFORE pooling.
                # Powers of an averaged direction erase two-/three-fold local
                # symmetry and make distinct torsion angles indistinguishable.
                a, b = pair_directions[0][:, :, None], pair_directions[1][:, None, :]
                coefficients = pair_weights[0][:, :, None] * pair_weights[1][:, None, :]
                cosine = (a * b).sum(-1)
                sine = (torch.linalg.cross(a, b, dim=-1)
                        * axis[:, None, None]).sum(-1)
                powers = []
                real, imaginary = cosine, sine
                for harmonic in range(1, self.phase_harmonics + 1):
                    powers.extend(((imaginary * coefficients).sum((1, 2)),
                                   (real * coefficients).sum((1, 2))))
                    if harmonic == 1:
                        confidence = pair_confidence[0][:, :, None] * pair_confidence[1][:, None, :]
                        powers.append((confidence * coefficients).sum((1, 2)))
                    real, imaginary = real * cosine - imaginary * sine, imaginary * cosine + real * sine
                phase = torch.stack(powers, -1)
            pieces.extend((edge_attr[edge_mask].to(ligand_tokens.dtype), phase))
            midpoint = ligand_coords[bonds].mean(1)
            for weight in side_weights:
                delta = weight @ ligand_coords - midpoint
                axial = (delta * axis).sum(-1)
                transverse = delta - axial[:, None] * axis
                pieces.append(torch.stack((axial, torch.linalg.vector_norm(transverse, dim=-1),
                                           torch.linalg.vector_norm(delta, dim=-1)), -1)
                              / self.max_distance)
        token = self.net(torch.cat(pieces, dim=-1))
        return token.unsqueeze(0)


class TernaryGeometryEncoder(nn.Module):
    """Encode typed components and their chemically labelled local contacts.

    Global distance histograms alone cannot distinguish which residue or atom
    participates in an interface. Legacy mode selects nearest contacts; v2
    softly pools all pairs and uses tapered typed radius messages. Both modes
    fuse endpoint features with an RBF encoding of separation. Scalar geometry enters this
    controller, so the resulting context remains invariant to a common rigid
    transform. Optional unpooled protein contacts retain directions and levers
    for a separate equivariant motion head.
    """

    PAIRS = ((0, 2), (1, 2), (0, 1))  # E3-L, POI-L, E3-POI
    RBF_CHUNK_SIZE = 65536

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_rbf: int = 16, max_distance: float = 40.0,
                 max_pair_tokens: int = 24, contacts_per_ligand: int = 4,
                 joint_geometry: bool = False,
                 global_node_context: bool = False,
                 spatial_geometry: bool = False,
                 global_node_attention: bool = False,
                 continuous_geometry: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.continuous_geometry = bool(continuous_geometry)
        self.n_rbf = n_rbf
        self.max_distance = float(max_distance)
        if max_pair_tokens <= 0:
            raise ValueError("max_pair_tokens must be positive")
        self.max_pair_tokens = int(max_pair_tokens)
        if contacts_per_ligand <= 0:
            raise ValueError("contacts_per_ligand must be positive")
        self.contacts_per_ligand = int(contacts_per_ligand)
        self.protein_feature_proj = nn.Linear(input_dim, hidden_dim)
        self.ligand_feature_encoder = LigandFeatureEncoder(hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.component_queries = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.pair_type_embedding = nn.Embedding(len(self.PAIRS), hidden_dim)
        self.pair_queries = nn.Parameter(
            torch.randn(len(self.PAIRS), hidden_dim) * 0.02
        )
        self.register_buffer(
            "rbf_centers", torch.linspace(0.0, max_distance, n_rbf),
        )
        self.rbf_width = max_distance / max(n_rbf - 1, 1)

        # Per pair: RBF histogram + min/mean/std + contacts at 4/8/12 A.
        # Per component: radius of gyration + log node count.
        geometry_dim = len(self.PAIRS) * (n_rbf + 6) + 3 * 2
        self.geometry_net = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + n_rbf),
            nn.Linear(hidden_dim * 2 + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The same interface projector is reused for E3-L and POI-L so their
        # representations stay comparable while their inputs remain distinct.
        self.interface_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ligand_contact_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 2),
            nn.Linear(hidden_dim * 3 + 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.joint_geometry = bool(joint_geometry)
        self.global_node_context = bool(global_node_context)
        self.spatial_geometry = bool(spatial_geometry)
        self.global_node_attention = bool(global_node_attention)
        self.edge_geometry_dim = n_rbf + 1 + (8 if self.spatial_geometry else 0)
        if self.global_node_context and not self.joint_geometry:
            raise ValueError('global node context requires joint geometry')
        if self.spatial_geometry and not self.joint_geometry:
            raise ValueError('spatial geometry requires joint geometry')
        if self.global_node_attention and not self.joint_geometry:
            raise ValueError('global node attention requires joint geometry')
        if self.joint_geometry:
            self.messages = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * hidden_dim + self.edge_geometry_dim, hidden_dim),
                              nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(2)
            ])
            self.message_updates = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                              nn.LayerNorm(hidden_dim)) for _ in range(2)
            ])
            if self.global_node_attention:
                heads = 4 if hidden_dim % 4 == 0 else 1
                self.global_attention = nn.ModuleList([
                    nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
                    for _ in range(2)
                ])
                self.global_attention_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
                for attention in self.global_attention:
                    nn.init.zeros_(attention.out_proj.weight)
                    nn.init.zeros_(attention.out_proj.bias)
            if self.global_node_context:
                # Fixed E3/POI/ligand slots preserve component identity instead
                # of averaging three interfaces into one indistinguishable token.
                self.global_updates = nn.ModuleList([
                    nn.Sequential(nn.LayerNorm(4 * hidden_dim),
                                  nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim))
                    for _ in range(2)
                ])
                # An explicitly initialized upgrade is identity until learned.
                for update in self.global_updates:
                    nn.init.zeros_(update[-1].weight)
                    nn.init.zeros_(update[-1].bias)

    def _component_context(self, features, types):
        """Differentiable summaries of every node, in fixed component order."""
        tokens = []
        for component in range(3):
            nodes = features[types == component]
            attention = torch.softmax(nodes @ self.component_queries[component], dim=0)
            tokens.append((nodes.mean(0) + (nodes * attention[:, None]).sum(0)) / 2)
        return torch.cat(tokens, dim=-1)

    def _spatial_edge_geometry(self, coords, types, receiver, direction):
        """Eight SE(3)-invariant features anchoring each edge in the full complex.

        Ordered center distances and edge/center angles supply global position;
        the own-center angle exposes radial versus tangential contacts. A signed
        triple product distinguishes handedness without using laboratory axes.
        All centers come from the current prediction, never the native target.
        """
        centers = torch.stack([coords[types == component].mean(0) for component in range(3)])
        offsets = centers[None] - coords[receiver, None]
        distances = torch.linalg.vector_norm(offsets, dim=-1)
        # A node can coincide with a component center. Softening at 1 A avoids
        # the 1/epsilon coordinate derivatives of a normalized zero vector.
        axes = offsets / torch.sqrt(offsets.square().sum(-1, keepdim=True) + 1.0)
        angles = (axes * direction[:, None]).sum(-1)
        own_axis = axes[torch.arange(len(receiver), device=coords.device), types[receiver]]
        radial_angle = (own_axis * direction).sum(-1, keepdim=True)
        handedness = (torch.cross(axes[:, 0], axes[:, 1], dim=-1) * direction).sum(-1, keepdim=True)
        return torch.cat((torch.log1p(distances / self.max_distance),
                          angles, radial_angle, handedness), dim=-1)

    def _joint_contacts(self, features, coords, types):
        """Typed messages: legacy kNN or continuously tapered radius support.

        Edges point from receiver to sender. Distances encode scalar features;
        directions are retained for the equivariant motion readout, not pooled.
        """
        receivers, senders = [], []
        all_indices = torch.arange(len(coords), device=coords.device)
        for component in range(3):
            candidates = all_indices[types == component]
            if not len(candidates):
                raise ValueError('joint geometry requires all three components')
            distances = torch.cdist(coords, coords[candidates], compute_mode='donot_use_mm_for_euclid_dist')
            distances = distances.masked_fill(all_indices[:, None] == candidates[None], float('inf'))
            if self.continuous_geometry:
                radius = torch.where(types[:, None] == component, 12., 24.)
                receiver, local = (distances < radius).nonzero(as_tuple=True)
                receivers.append(receiver)
                senders.append(candidates[local])
            else:
                values, local = distances.topk(min(4, len(candidates)), largest=False, dim=1)
                receiver = all_indices[:, None].expand_as(local)
                valid = torch.isfinite(values)
                receivers.append(receiver[valid])
                senders.append(candidates[local[valid]])
        receiver, sender = torch.cat(receivers), torch.cat(senders)
        delta = coords[sender] - coords[receiver]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        geometry = torch.cat((
            torch.exp(-0.5 * ((distance[:, None] - self.rbf_centers) / self.rbf_width) ** 2),
            torch.log1p(distance[:, None] / self.max_distance)), dim=-1)
        # Sub-angstrom overlap has no reliable contact direction. Bound its
        # coordinate Jacobian at 10/A rather than amplifying it up to 1e8/A.
        direction = (delta / torch.sqrt(distance[:, None].square() + .01)
                     if self.continuous_geometry else delta / distance[:, None].clamp_min(.1))
        if self.spatial_geometry:
            geometry = torch.cat((geometry, self._spatial_edge_geometry(
                coords, types, receiver, direction)), dim=-1)
        weight = ((1. - distance / torch.where(types[receiver] == types[sender], 12., 24.)).square()[:, None]
                  .to(features.dtype) if self.continuous_geometry else features.new_ones(len(receiver), 1))
        counts = features.new_zeros(len(coords), 1).index_add(0, receiver, weight)
        # A zero-message prior keeps normalization smooth even when an edge
        # enters/leaves support; clamp_min(1) has a derivative kink at one.
        counts = counts + 1. if self.continuous_geometry else counts.clamp_min(1)
        for layer, (message, update) in enumerate(zip(self.messages, self.message_updates)):
            if self.continuous_geometry:
                aggregate = torch.zeros_like(features)
                for start in range(0, len(receiver), 8192):
                    section = slice(start, start + 8192)

                    def summarize(nodes, edge_geometry, envelope, receiving, sending, message=message):
                        inputs = torch.cat((nodes[receiving], nodes[sending], edge_geometry), -1)
                        values = message(inputs) * envelope
                        return torch.zeros_like(nodes).index_add(0, receiving, values)

                    arguments = (features, geometry[section], weight[section],
                                 receiver[section], sender[section])
                    aggregate = aggregate + (checkpoint(summarize, *arguments, use_reentrant=False)
                                              if torch.is_grad_enabled() else summarize(*arguments))
                aggregate = aggregate / counts
            else:
                edge_features = message(torch.cat((features[receiver], features[sender], geometry), -1))
                aggregate = torch.zeros_like(features).index_add(0, receiver, edge_features) / counts
            features = features + update(torch.cat((features, aggregate), -1))
            if self.global_node_attention:
                # No pooled bottleneck: each node reads the detailed features of
                # every other node. need_weights=False enables PyTorch's SDPA
                # kernels instead of returning an N*N attention-weight tensor.
                nodes = self.global_attention_norms[layer](features).unsqueeze(0)
                attention, _ = self.global_attention[layer](nodes, nodes, nodes, need_weights=False)
                features = features + attention[0]
            if self.global_node_context:
                context = self._component_context(features, types).expand(len(features), -1)
                features = features + self.global_updates[layer](torch.cat((features, context), -1))
        return features, {'receiver': receiver, 'sender': sender,
                          'geometry': geometry,
                          'direction': direction,
                          'weight': weight,
                          'features': features}

    def _distance_rbf_sum(self, distances):
        offsets = distances[:, None] - self.rbf_centers[None, :]
        return torch.exp(-0.5 * (offsets / self.rbf_width) ** 2).sum(0)

    def _distance_histogram(self, distances):
        """Exact all-pair histogram without retaining all N*M*RBF activations.

        Checkpoint each chunk independently so backward also recomputes bounded
        chunks, rather than rebuilding the full histogram activation at once.
        Evaluation uses ordinary chunked operations and needs no checkpoint.
        """
        if distances.numel() <= self.RBF_CHUNK_SIZE:
            return self._distance_rbf_sum(distances) / distances.numel()
        total = distances.new_zeros(self.n_rbf)
        for chunk in distances.split(self.RBF_CHUNK_SIZE):
            if torch.is_grad_enabled() and chunk.requires_grad:
                total = total + checkpoint(self._distance_rbf_sum, chunk,
                                           use_reentrant=False)
            else:
                total = total + self._distance_rbf_sum(chunk)
        return total / distances.numel()

    def _pair_descriptor(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.reshape(-1)
        rbf = self._distance_histogram(distances)
        scale = self.max_distance
        nearest = ((distances * torch.softmax(-distances / 8., 0)).sum()
                   if self.continuous_geometry else distances.min())
        contacts = [(torch.sigmoid(threshold - distances).mean() if self.continuous_geometry
                     else (distances < threshold).to(distances.dtype).mean())
                    for threshold in (4., 8., 12.)]
        statistics = torch.stack((
            nearest / scale,
            distances.mean() / scale,
            distances.std(unbiased=False) / scale,
            *contacts,
        ))
        return torch.cat((rbf, statistics), dim=0)

    def _weighted_rbf(self, distances, weights, per_row=False):
        """Soft all-pair geometry with bounded, recomputed RBF activations."""
        total = distances.new_zeros((distances.shape[0], self.n_rbf)
                                    if per_row else (self.n_rbf,))
        flat_distances, flat_weights = distances.reshape(-1), weights.reshape(-1)
        for start in range(0, flat_distances.numel(), 8192):
            indexes = torch.arange(start, min(start + 8192, flat_distances.numel()),
                                   device=distances.device)
            rows = torch.div(indexes, distances.shape[1], rounding_mode='floor') if per_row else None

            def summarize(values, coefficients, rows=rows):
                rbf = torch.exp(-.5 * ((values[:, None] - self.rbf_centers) / self.rbf_width).square())
                weighted = rbf * coefficients[:, None]
                if rows is None:
                    return weighted.sum(0)
                return values.new_zeros(distances.shape[0], self.n_rbf).index_add(0, rows, weighted)

            values, coefficients = flat_distances[indexes], flat_weights[indexes]
            total = total + (checkpoint(summarize, values, coefficients, use_reentrant=False)
                             if torch.is_grad_enabled() and (values.requires_grad or coefficients.requires_grad)
                             else summarize(values, coefficients))
        return total

    def _local_pair_token(
        self,
        features_a: torch.Tensor,
        coords_a: torch.Tensor,
        features_b: torch.Tensor,
        coords_b: torch.Tensor,
        pair_index: int,
        distances: torch.Tensor,
        return_contacts: bool = False,
    ):
        """Pool feature-labelled interfaces with legacy hard or v2 soft contacts."""
        if self.continuous_geometry:
            query = self.pair_queries[pair_index]
            scores = -distances / 8. + (features_a @ query)[:, None] + (features_b @ query)[None]
            weights = torch.softmax(scores.reshape(-1), 0).reshape_as(distances)
            marginal_a, marginal_b = weights.sum(1), weights.sum(0)
            token = self.pair_net(torch.cat(((features_a * marginal_a[:, None]).sum(0),
                                            (features_b * marginal_b[:, None]).sum(0),
                                            self._weighted_rbf(distances, weights))))
            token = token + self.pair_type_embedding.weight[pair_index]
            if return_contacts:
                return (token, token[None], (coords_a * marginal_a[:, None]).sum(0, keepdim=True),
                        (coords_b * marginal_b[:, None]).sum(0, keepdim=True))
            return token
        n_pairs = distances.numel()
        k = min(self.max_pair_tokens, n_pairs)
        values, flat_indices = torch.topk(
            distances.reshape(-1), k=k, largest=False, sorted=True,
        )
        index_a = torch.div(flat_indices, coords_b.shape[0], rounding_mode="floor")
        index_b = flat_indices.remainder(coords_b.shape[0])
        offsets = values[:, None] - self.rbf_centers[None, :]
        distance_rbf = torch.exp(
            -0.5 * (offsets / self.rbf_width) ** 2
        )
        pair_features = self.pair_net(torch.cat((
            features_a[index_a], features_b[index_b], distance_rbf,
        ), dim=-1))
        pair_features = pair_features + self.pair_type_embedding.weight[pair_index]
        attention = torch.softmax(
            pair_features @ self.pair_queries[pair_index], dim=0,
        )
        pooled = (
            (pair_features * attention[:, None]).sum(dim=0)
            + pair_features.mean(dim=0)
            + pair_features.max(dim=0).values
        ) / 3.0
        if return_contacts:
            return pooled, pair_features, coords_a[index_a], coords_b[index_b]
        return pooled

    def _ligand_centric_contacts(
        self,
        protein_features: torch.Tensor,
        ligand_features: torch.Tensor,
        pair_index: int,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Return one contact token per ligand atom with guaranteed coverage."""
        if self.continuous_geometry:
            scores = -distances / 8. + (protein_features @ self.pair_queries[pair_index])[None]
            weights = torch.softmax(scores, 1)
            contacts = self.pair_net(torch.cat((ligand_features, weights @ protein_features,
                                                self._weighted_rbf(distances, weights, per_row=True)), -1))
            return contacts + self.pair_type_embedding.weight[pair_index]
        k = min(self.contacts_per_ligand, protein_features.shape[0])
        values, protein_indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True,
        )
        ligand_indices = torch.arange(
            ligand_features.shape[0], device=ligand_features.device,
        )[:, None].expand(-1, k)
        offsets = values[..., None] - self.rbf_centers
        distance_rbf = torch.exp(-0.5 * (offsets / self.rbf_width) ** 2)
        contacts = self.pair_net(torch.cat((
            ligand_features[ligand_indices],
            protein_features[protein_indices],
            distance_rbf,
        ), dim=-1))
        contacts = contacts + self.pair_type_embedding.weight[pair_index]
        scores = contacts @ self.pair_queries[pair_index]
        weights = torch.softmax(scores, dim=1)
        return (
            (contacts * weights[..., None]).sum(dim=1)
            + contacts.mean(dim=1)
            + contacts.max(dim=1).values
        ) / 3.0

    def _ligand_direction_invariants(
        self,
        e3_coords: torch.Tensor,
        poi_coords: torch.Tensor,
        ligand_coords: torch.Tensor,
        distances,
        lig_edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return angle and SE(3)-invariant, reflection-sensitive handedness."""
        if self.continuous_geometry:
            e3_direction = torch.softmax(-distances[0].T / 8., 1) @ e3_coords - ligand_coords
            poi_direction = torch.softmax(-distances[1].T / 8., 1) @ poi_coords - ligand_coords
            e3_direction = e3_direction / torch.sqrt(e3_direction.square().sum(-1, keepdim=True) + .01)
            poi_direction = poi_direction / torch.sqrt(poi_direction.square().sum(-1, keepdim=True) + .01)
        else:
            e3_nearest = distances[0].argmin(dim=0)
            poi_nearest = distances[1].argmin(dim=0)
            e3_direction = F.normalize(e3_coords[e3_nearest] - ligand_coords, dim=-1, eps=.1)
            poi_direction = F.normalize(poi_coords[poi_nearest] - ligand_coords, dim=-1, eps=.1)
        cosine = (e3_direction * poi_direction).sum(dim=-1)

        bond_direction = torch.zeros_like(ligand_coords)
        if lig_edge_index is not None and lig_edge_index.numel():
            source, target = lig_edge_index.long()
            bond_direction.index_add_(
                0, source, ligand_coords[target] - ligand_coords[source],
            )
        bond_direction = (bond_direction / torch.sqrt(bond_direction.square().sum(-1, keepdim=True) + .01)
                          if self.continuous_geometry else F.normalize(bond_direction, dim=-1, eps=.1))
        handedness = (
            torch.cross(e3_direction, poi_direction, dim=-1) * bond_direction
        ).sum(dim=-1)
        return torch.stack((cosine, handedness), dim=-1)

    def forward(self, node_features: torch.Tensor, node_coords: torch.Tensor,
                node_types: torch.Tensor, batch: torch.Tensor = None,
                lig_edge_index: torch.Tensor = None, node_residual: torch.Tensor = None,
                return_protein_pair_contacts: bool = False) -> dict:
        if batch is None:
            batch = torch.zeros(
                node_features.shape[0], dtype=torch.long,
                device=node_features.device,
            )
        node_types = node_types.to(device=node_features.device)
        node_coords = node_coords.to(device=node_features.device)
        features = node_features.new_zeros(node_features.shape[0], self.hidden_dim)
        protein_mask = node_types != 2
        ligand_mask = node_types == 2
        features[protein_mask] = self.protein_feature_proj(
            node_features[protein_mask]
        )
        features[ligand_mask] = self.ligand_feature_encoder(
            node_features[ligand_mask]
        )
        features = features + self.type_embedding(node_types)
        if node_residual is not None:
            if node_residual.shape != features.shape or not torch.isfinite(node_residual).all():
                raise ValueError('condition residual must match embedded node features')
            # Apply only after categorical atom encoding, never to raw category IDs.
            features = features + node_residual

        joint = None
        if self.joint_geometry:
            if bool((batch != 0).any()):
                raise ValueError('joint geometry currently requires one complex')
            features, joint = self._joint_contacts(features, node_coords, node_types)

        global_outputs = []
        e3l_outputs = []
        pol_outputs = []
        ligand_token_outputs = []
        protein_pair_outputs = []
        n_batch = int(batch.max().item()) + 1
        for batch_index in range(n_batch):
            batch_mask = batch == batch_index
            component_tokens = []
            component_coords = []
            component_features_by_type = []
            component_descriptors = []
            for component_type in range(3):
                mask = batch_mask & (node_types == component_type)
                if not bool(mask.any()):
                    raise ValueError(
                        f"batch {batch_index} has no nodes of type {component_type}"
                    )
                component_features = features[mask]
                coords = node_coords[mask]
                attention = torch.softmax(
                    component_features @ self.component_queries[component_type],
                    dim=0,
                )
                token = ((component_features * attention[:, None]).sum(dim=0)
                         + component_features.mean(dim=0))
                token = (token / 2. if self.continuous_geometry
                         else (token + component_features.max(dim=0).values) / 3.)
                component_tokens.append(token)
                component_coords.append(coords)
                component_features_by_type.append(component_features)
                centered = coords - coords.mean(dim=0, keepdim=True)
                radius = torch.linalg.vector_norm(centered) / math.sqrt(len(coords))
                component_descriptors.extend((
                    radius / self.max_distance,
                    torch.log1p(coords.new_tensor(float(coords.shape[0]))) / 10.0,
                ))

            pair_distances = [
                torch.cdist(component_coords[a], component_coords[b], compute_mode='donot_use_mm_for_euclid_dist')
                for a, b in self.PAIRS
            ]
            pair_descriptors = [self._pair_descriptor(distances) for distances in pair_distances]
            e3_contacts = self._ligand_centric_contacts(
                component_features_by_type[0], component_features_by_type[2], 0,
                pair_distances[0].T,
            )
            pol_contacts = self._ligand_centric_contacts(
                component_features_by_type[1], component_features_by_type[2], 1,
                pair_distances[1].T,
            )
            ligand_tokens = self.ligand_contact_fusion(torch.cat((
                component_features_by_type[2], e3_contacts, pol_contacts,
                self._ligand_direction_invariants(
                    component_coords[0], component_coords[1],
                    component_coords[2], pair_distances, lig_edge_index,
                ),
            ), dim=-1))
            protein_pair = self._local_pair_token(
                component_features_by_type[0], component_coords[0],
                component_features_by_type[1], component_coords[1], 2, pair_distances[2],
                return_contacts=return_protein_pair_contacts,
            )
            if return_protein_pair_contacts:
                pair_token, contact_tokens, left, right = protein_pair
                protein_pair_outputs.append({
                    'tokens': contact_tokens,
                    'direction': ((right - left) / torch.sqrt((right - left).square().sum(-1, keepdim=True) + .01)
                                  if self.continuous_geometry else F.normalize(right - left, dim=-1, eps=.1)),
                    # All rigid updates rotate about the ligand center.
                    'lever': ((left + right) / 2 - component_coords[2].mean(0)) / 10.0,
                })
            else:
                pair_token = protein_pair
            local_pair_tokens = [
                e3_contacts.mean(dim=0),
                pol_contacts.mean(dim=0),
                pair_token,
            ]
            geometry = torch.cat((
                *pair_descriptors, torch.stack(component_descriptors),
            ))
            geometry_token = self.geometry_net(geometry)
            global_outputs.append(self.fusion(torch.cat((
                *component_tokens, *local_pair_tokens, geometry_token,
            ))))
            e3l_outputs.append(self.interface_fusion(torch.cat((
                component_tokens[0], component_tokens[2],
                local_pair_tokens[0], geometry_token,
            ))))
            pol_outputs.append(self.interface_fusion(torch.cat((
                component_tokens[1], component_tokens[2],
                local_pair_tokens[1], geometry_token,
            ))))
            ligand_token_outputs.append(ligand_tokens)
        result = {
            "global": torch.stack(global_outputs, dim=0),
            "e3l": torch.stack(e3l_outputs, dim=0),
            "pol": torch.stack(pol_outputs, dim=0),
            "ligand_tokens": ligand_token_outputs,
        }
        if return_protein_pair_contacts:
            result['protein_pair_contacts'] = protein_pair_outputs
        if joint is not None:
            result['joint_geometry'] = joint
        return result


class TernaryRelationEncoder(nn.Module):
    """Geometry-aware relations survive every layer instead of being pooled away.

    Sixteen soft anchors per protein bound the global pair cost; ligand atoms
    remain explicit. Full-resolution nodes read relations back before motion
    decoding. Anchors use invariant geometric patches in the current profile,
    are SE(3)-equivariant, and independent
    of residue ordering. No contact radius removes distant candidate interfaces.
    """

    def __init__(self, input_dim, hidden_dim, *, geometric_anchors=False, refined=False,
                 stable_anchors=False):
        super().__init__()
        self.geometric_anchors = geometric_anchors
        self.refined = refined
        self.stable_anchors = stable_anchors
        self.hidden_dim = hidden_dim
        self.protein_feature_proj = nn.Linear(input_dim, hidden_dim)
        self.ligand_feature_encoder = LigandFeatureEncoder(hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.anchor_queries = nn.Parameter(torch.randn(2, 16, hidden_dim) * .02)
        self.bond_message = nn.Linear(2 * hidden_dim + (4 if geometric_anchors else 0), hidden_dim)
        if refined:
            self.bond_message = nn.Sequential(self.bond_message, nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_type = nn.Embedding(9, hidden_dim)
        self.pair_input = nn.Sequential(nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.SiLU())
        self.pair_updates = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim),
                          nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(2)])
        triangle_dim = max(4, hidden_dim // 4)
        self.triangle_left = nn.ModuleList([nn.Linear(hidden_dim, triangle_dim, bias=False) for _ in range(2)])
        self.triangle_right = nn.ModuleList([nn.Linear(hidden_dim, triangle_dim, bias=False) for _ in range(2)])
        self.triangle_output = nn.ModuleList([nn.Linear(triangle_dim, hidden_dim, bias=False) for _ in range(2)])
        if geometric_anchors:
            self.triangle_mix = nn.Parameter(torch.zeros(2, 3))
            self.local_pair_message = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim + 4), nn.Linear(2 * hidden_dim + 4, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_updates = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim),
                          nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(2)])
        self.pair_logits = nn.Linear(hidden_dim, 1, bias=False)
        self.read_logits = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, 1, bias=False))
        self.node_read = nn.Sequential(nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim),
                                       nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion = nn.Sequential(nn.LayerNorm(6 * hidden_dim), nn.Linear(6 * hidden_dim, hidden_dim), nn.SiLU())
        self.interface_fusion = nn.Sequential(nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU())

    @staticmethod
    def geometry(left, right, left_centers, right_centers):
        delta = right[None] - left[:, None]
        squared = delta.square().sum(-1)
        # Smooth even at coincident anchors; no norm/atan2 singularity.
        distance = torch.sqrt(squared + .01)
        direction = delta / distance[..., None]
        a = left - left_centers
        b = right - right_centers
        a = a / torch.sqrt(a.square().sum(-1, keepdim=True) + 1.)
        b = b / torch.sqrt(b.square().sum(-1, keepdim=True) + 1.)
        return torch.stack((torch.log1p(distance / 10.),
                            (direction * a[:, None]).sum(-1),
                            (direction * b[None]).sum(-1),
                            (torch.linalg.cross(a[:, None], b[None]) * direction).sum(-1)), -1)

    @staticmethod
    def anchor_geometry(coords, count=16, content=None, stable=False):
        """Smooth farthest-point patches without discrete residue selection.

        Soft extrema preserve coordinate derivatives at ties. Exactly symmetric
        geometry with identical features cannot define an arbitrary orientation.
        """
        radius2 = (coords - coords.mean(0)).square().sum(-1)
        scale = radius2.mean().clamp_min(1.)
        # Four broad geometric seeds support sixteen semantic queries. Short,
        # soft recursion avoids amplifying ties through sixteen decisions.
        sharpness = 3. if stable else 32.
        seed_count = min(4, count) if stable else count
        nearest = radius2
        seeds, distances = [], []
        for index in range(seed_count):
            selection = sharpness * nearest / scale
            if content is not None:
                semantic = content[index::seed_count].mean(0) if stable else content[index]
                selection = selection + (2. * torch.tanh(semantic / 2.) if stable else semantic)
            seed = selection.softmax(-1) @ coords
            seeds.append(seed)
            distances.append((coords - seed).square().sum(-1))
            nearest = -scale / sharpness * torch.logsumexp(-sharpness * torch.stack(distances) / scale, dim=0)
        seeds = torch.stack(seeds)
        if stable:
            seeds = seeds[torch.arange(count, device=coords.device) % seed_count]
        return -(seeds[:, None] - coords[None]).square().sum(-1) / (scale / (4. if stable else 16.))

    def ligand_messages(self, features, edge_index, edge_attr):
        if edge_index is None or not edge_index.numel():
            return features
        source, destination = edge_index
        inputs = [features[source], features[destination]]
        if self.geometric_anchors:
            if edge_attr is None or edge_attr.shape != (len(source), 4):
                raise ValueError('ternary bond messages require four bond-type features per edge')
            inputs.append(edge_attr.to(features))
        values = self.bond_message(torch.cat(inputs, -1))
        counts = torch.bincount(source, minlength=len(features)).to(features).clamp_min(1.)
        return features + torch.zeros_like(features).index_add(0, source, values) / counts[:, None]

    def forward(self, node_features, node_coords, node_types, *, lig_edge_index=None,
                lig_edge_attr=None, node_residual=None):
        types, coords = node_types.long(), node_coords
        protein, ligand = types != 2, types == 2
        features = node_features.new_zeros(len(types), self.hidden_dim)
        features[protein] = self.protein_feature_proj(node_features[protein])
        features[ligand] = self.ligand_feature_encoder(node_features[ligand])
        features = features + self.type_embedding(types)
        if node_residual is not None:
            features = features + node_residual
        if lig_edge_index is not None and lig_edge_index.numel():
            lig_features = features[ligand]
            features = features.index_copy(0, ligand.nonzero().flatten(),
                                           self.ligand_messages(lig_features, lig_edge_index, lig_edge_attr))
        centers = torch.stack([coords[types == component].mean(0) for component in range(3)])
        if self.geometric_anchors:
            # Full-resolution protein interfaces complement the global anchors.
            # Compact support has zero value AND derivative at the 20 A boundary.
            updates = torch.zeros_like(features)
            for component in (0, 1):
                receivers = (types == component).nonzero().flatten()
                senders = (types == 1 - component).nonzero().flatten()
                for chunk in receivers.split(128):
                    distance2 = (coords[chunk, None] - coords[senders][None]).square().sum(-1)
                    row, column = (distance2 < 400.).nonzero(as_tuple=True)
                    receiver, sender = chunk[row], senders[column]
                    delta = coords[sender] - coords[receiver]
                    direction = delta / torch.sqrt(delta.square().sum(-1, keepdim=True) + .01)
                    a, b = coords[receiver] - centers[component], coords[sender] - centers[1 - component]
                    a = a / torch.sqrt(a.square().sum(-1, keepdim=True) + 1.)
                    b = b / torch.sqrt(b.square().sum(-1, keepdim=True) + 1.)
                    geometry = torch.stack((torch.log1p(torch.sqrt(distance2[row, column] + .01) / 10.),
                                            (direction * a).sum(-1), (direction * b).sum(-1),
                                            (torch.linalg.cross(a, b) * direction).sum(-1)), -1)
                    weight = (1 - distance2[row, column] / 400.).square()
                    values = self.local_pair_message(torch.cat((features[receiver], features[sender], geometry), -1))
                    normalizer = weight.new_ones(len(chunk)).index_add(0, row, weight)
                    message = torch.zeros_like(features[chunk]).index_add(0, row, weight[:, None] * values)
                    updates = updates.index_copy(0, chunk, message / normalizer[:, None])
            features = features + updates
        latent, positions, assignments, material_weights = [], [], [], []
        for component in (0, 1):
            nodes = features[types == component]
            logits = self.anchor_queries[component] @ nodes.T / math.sqrt(self.hidden_dim)
            if self.geometric_anchors:
                logits = logits + self.anchor_geometry(coords[types == component], content=logits,
                                                       stable=self.stable_anchors)
            weights = logits.softmax(-1)
            material_weights.append(weights)
            latent.append(weights @ nodes)
            positions.append(weights @ coords[types == component])
            assignments.append(logits.T.softmax(-1))
        latent.append(features[ligand])
        positions.append(coords[ligand])
        tokens, points = torch.cat(latent), torch.cat(positions)
        latent_types = torch.cat((types.new_zeros(16), types.new_ones(16), types.new_full((int(ligand.sum()),), 2)))
        assignment = features.new_zeros(len(features), len(tokens))
        for component, section in ((0, slice(0, 16)), (1, slice(16, 32))):
            assignment[types == component, section] = assignments[component]
        assignment[ligand, 32:] = torch.eye(int(ligand.sum()), device=coords.device, dtype=features.dtype)
        geometry = self.geometry(points, points, centers[latent_types], centers[latent_types])
        pair = self.pair_input(torch.cat((tokens[:, None].expand(-1, len(tokens), -1),
                                          tokens[None].expand(len(tokens), -1, -1), geometry), -1))
        pair = pair + self.pair_type(latent_types[:, None] * 3 + latent_types[None])
        balance = torch.bincount(latent_types, minlength=3).to(features).log()[latent_types]
        for index, (pair_update, node_update) in enumerate(zip(self.pair_updates, self.node_updates)):
            left, right = self.triangle_left[index](pair), self.triangle_right[index](pair)
            if self.geometric_anchors:
                intermediate_weight = self.triangle_mix[index].softmax(-1)[latent_types] / balance.exp()
                triangle = torch.einsum('ikh,kjh,k->ijh', left, right, intermediate_weight)
            else:
                triangle = torch.einsum('ikh,kjh->ijh', left, right) / len(tokens)
            triangle = self.triangle_output[index](triangle)
            pair = pair + pair_update(torch.cat((pair, tokens[:, None] + tokens[None], triangle), -1))
            attention = (self.pair_logits(pair).squeeze(-1) - balance[None]).softmax(-1)
            message = attention @ tokens
            evidence = (attention[..., None] * pair).sum(1)
            tokens = tokens + node_update(torch.cat((tokens, message, evidence), -1))
        geometry = self.geometry(coords, points, centers[types], centers[latent_types])
        relation = torch.einsum('ik,kjh->ijh', assignment, pair)
        read = (self.read_logits(torch.cat((relation, geometry), -1)).squeeze(-1)
                - balance[None]).softmax(-1)
        features = features + self.node_read(torch.cat((features, read @ tokens,
                                                        (read[..., None] * relation).sum(1)), -1))
        bodies = [features[types == component].mean(0) for component in range(3)]
        pair_tokens = [pair[latent_types == a][:, latent_types == b].mean((0, 1))
                       for a, b in ((0, 2), (1, 2), (0, 1))]
        return {'global': self.fusion(torch.cat((*bodies, *pair_tokens)))[None],
                'e3l': self.interface_fusion(torch.cat((bodies[0], bodies[2], pair_tokens[0])))[None],
                'pol': self.interface_fusion(torch.cat((bodies[1], bodies[2], pair_tokens[1])))[None],
                'protein_pair': self.interface_fusion(torch.cat((bodies[0], bodies[1], pair_tokens[2])))[None],
                'ligand_tokens': [features[ligand]],
                'joint_geometry': {'features': features, 'relation': relation, 'geometry': geometry,
                                   'read': read, 'points': points, 'latent_types': latent_types,
                                   'material_weights': material_weights,
                                   'latent_features': tokens, 'pair': pair}}


BRANCHES = ('e3_ligand', 'poi_ligand')


def bounded_vector(value):
    """Isotropic compression, retaining direction when the raw norm overflows."""
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    finite = torch.isfinite(norm)
    ordinary = value / (1 + torch.where(finite, norm, torch.zeros_like(norm)))
    scale = value.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    scaled = value / scale
    stable = scaled / (scale.reciprocal()
                       + torch.linalg.vector_norm(scaled, dim=-1, keepdim=True))
    # Ordinary inputs retain the historical float arithmetic exactly. Scaling
    # is only needed when squaring finite components overflowed the raw norm.
    return torch.where(finite, ordinary, stable)


class TokenNegotiator(nn.Module):
    """Let global, interface, linker, and diffusion-state tokens communicate."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float,
                 num_token_types: int = 5):
        super().__init__()
        self.token_types = nn.Parameter(
            torch.randn(num_token_types, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.token_types.shape[0]:
            raise ValueError(
                f"expected [B,{self.token_types.shape[0]},D] tokens, "
                f"got {tuple(tokens.shape)}"
            )
        return self.encoder(tokens + self.token_types.unsqueeze(0))


class JudgeHead(nn.Module):
    """Reserved global confidence head; frozen until labels are available."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),  # scalar confidence
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1] confidence score."""
        return self.confidence_net(x)


class StateHead(nn.Module):
    """
    Predicts thermodynamic/functional state indicators:
    - ΔΔG estimate
    - Productive complex probability
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),  # [ΔΔG, p_productive]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 2] with [ΔΔG_estimate, productive_probability]."""
        return self.net(x)


class TorsionCoordinator(nn.Module):
    """Resolve two interface opinions into one score per rotatable bond.

    A PROTAC has one shared set of torsional degrees of freedom.  The two
    DiffDock experts therefore provide evidence, not independently executable
    torsion actions.  This head sees both per-bond scores together with the
    ternary context and predicts a per-bond mixture weight and residual.

    The final projection is zero-initialized, so an untrained coordinator starts
    at the neutral expert mean while remaining fully learnable.
    """

    def __init__(self, context_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        # Per-bond evidence is bounded before entering the MLP. This prevents
        # large but finite expert scores from overflowing in a raw product.
        self.net = nn.Sequential(
            nn.Linear(context_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        score_e3l: torch.Tensor,
        score_pol: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coordinated score, E3-side weight, and learned residual."""
        if score_e3l.shape != score_pol.shape:
            raise ValueError(
                "torsion experts must describe the same rotatable bonds: "
                f"{tuple(score_e3l.shape)} != {tuple(score_pol.shape)}"
            )
        if score_e3l.ndim != 2:
            raise ValueError(
                f"torsion scores must have shape [B,K], got {score_e3l.shape}"
            )
        if context.shape[0] != score_e3l.shape[0]:
            raise ValueError("torsion context and scores have different batches")

        n_torsions = score_e3l.shape[1]
        if context.ndim == 2:
            context_per_bond = context.unsqueeze(1).expand(-1, n_torsions, -1)
        elif context.ndim == 3 and context.shape[1] == n_torsions:
            context_per_bond = context
        else:
            raise ValueError(
                "torsion context must have shape [B,D] or [B,K,D], got "
                f"{tuple(context.shape)} for K={n_torsions}"
            )
        bounded_e3l = torch.tanh(score_e3l)
        bounded_pol = torch.tanh(score_pol)
        bond_evidence = torch.stack(
            (
                bounded_e3l,
                bounded_pol,
                bounded_e3l - bounded_pol,
                bounded_e3l * bounded_pol,
            ),
            dim=-1,
        )
        coordination = self.net(
            torch.cat((context_per_bond, bond_evidence), dim=-1)
        )
        weight_e3l = torch.sigmoid(coordination[..., 0])
        residual = 10.0 * torch.tanh(coordination[..., 1])
        score = (
            weight_e3l * score_e3l
            + (1.0 - weight_e3l) * score_pol
            + residual
        )
        return score, weight_e3l, residual





ARCHITECTURE = 'residual_consensus_v2'
DIRECT_ARCHITECTURE = 'ternary_geometry_independent_v1'
CONSENSUS_CHECKPOINT_VERSION = 38
INTERFACE_OBJECTIVE_CHECKPOINT_VERSION = 22
OBJECTIVE_CONTRACT_CHECKPOINT_VERSION = 21
ENDPOINT_PROTOCOL_CHECKPOINT_VERSION = 20
LEGACY_CONSENSUS_CHECKPOINT_VERSION = 18
PREVIOUS_CONSENSUS_CHECKPOINT_VERSION = 19
SOURCE_NAMES = ('sdpi', 'bitseed')
SOURCE_CONTRACT_VERSION = 'candidate_source_conditioning_v1'
LEGACY_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v1'
FEATURE_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v2'
PREVIOUS_PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v3'
HORIZONTAL_MODEL_PROFILE = 'ternary_endpoint_v4'
CONDITIONAL_MODEL_PROFILE = 'ternary_endpoint_v5'
GEOMETRIC_MODEL_PROFILE = 'ternary_endpoint_v6'
VECTOR_MODEL_PROFILE = 'ternary_endpoint_v7'
JOINT_MODEL_PROFILE = 'ternary_endpoint_v8'
INTEGRATED_MODEL_PROFILE = 'ternary_endpoint_v9'
PRODUCTION_MODEL_PROFILE = CONSTRAINT_MODEL_PROFILE = 'ternary_constraints_v1'
MODEL_PROFILES = (LEGACY_PRODUCTION_MODEL_PROFILE, FEATURE_PRODUCTION_MODEL_PROFILE,
                  PREVIOUS_PRODUCTION_MODEL_PROFILE, HORIZONTAL_MODEL_PROFILE,
                  CONDITIONAL_MODEL_PROFILE, GEOMETRIC_MODEL_PROFILE, VECTOR_MODEL_PROFILE,
                  JOINT_MODEL_PROFILE, INTEGRATED_MODEL_PROFILE, PRODUCTION_MODEL_PROFILE)


def production_model_config(node_dim, input_dim, num_transformer_layers=3,
                            source_conditioning=False, *,
                            model_profile=PRODUCTION_MODEL_PROFILE,
                            expert_scalar_dim=58, expert_film_dim=48, expert_vector_dim=20):
    """One explicit new-model recipe; older weights retain their own config."""
    if model_profile not in MODEL_PROFILES:
        raise ValueError(f'unknown model profile: {model_profile}')
    config = dict(node_dim=node_dim, input_dim=input_dim,
                num_transformer_layers=num_transformer_layers, dropout=0.,
                source_conditioning=bool(source_conditioning),
                direct_ternary_motion=True, global_node_context=True,
                spatial_geometry=True, motion_attention=True,
                global_node_attention=True, rotation_scale_normalization=True,
                model_profile=model_profile)
    if model_profile == PRODUCTION_MODEL_PROFILE:
        for name in ('global_node_context', 'spatial_geometry', 'motion_attention',
                     'global_node_attention', 'rotation_scale_normalization'):
            del config[name]
    if model_profile in MODEL_PROFILES[1:]:
        if isinstance(expert_scalar_dim, bool) or not isinstance(expert_scalar_dim, int) or expert_scalar_dim <= 0:
            raise ValueError('expert scalar dimension must be positive')
        config['expert_scalar_dim'] = expert_scalar_dim
    if model_profile in MODEL_PROFILES[5:]:
        if isinstance(expert_film_dim, bool) or not isinstance(expert_film_dim, int) or not 0 < expert_film_dim <= expert_scalar_dim:
            raise ValueError('expert FiLM dimension must be positive and no larger than scalar read width')
        config['expert_film_dim'] = expert_film_dim
    if model_profile in MODEL_PROFILES[6:]:
        if isinstance(expert_vector_dim, bool) or not isinstance(expert_vector_dim, int) or expert_vector_dim < 1:
            raise ValueError('expert vector width must be positive')
        config['expert_vector_dim'] = expert_vector_dim
    return config




def candidate_source_tensor(source, reference):
    """Normalize one immutable trajectory source to a ``[1]`` class tensor."""
    if isinstance(source, str):
        if source not in SOURCE_NAMES:
            raise ValueError(f'unknown candidate source: {source!r}')
        source = SOURCE_NAMES.index(source)
    value = torch.as_tensor(source, device=reference.device).reshape(-1)
    if (value.shape != (1,) or value.dtype == torch.bool or value.is_complex()
            or not bool(((value == 0) | (value == 1)).all())):
        raise ValueError('candidate source must be one SDPI/BiTSeed class')
    return value.long()




class ConsensusFuse(nn.Module):
    """One production architecture: frozen evidence and joint constraint motion.

    Historical motion heads live on model-v4, not behind runtime switches here.
    Feature capability flags are consumed by the shared expert/flow executor.
    """

    supports_conditions = False
    direct_ternary_motion = True
    endpoint_v2 = continuous_torsion = symmetric_torsion = True
    mutual_experts = enhanced_conditions = refined_conditions = True
    joint_conditions = integrated_conditions = constraint_motion = True

    def __init__(self, node_dim=128, input_dim=None, num_transformer_layers=3,
                 dropout=0., source_conditioning=False,
                 model_profile=PRODUCTION_MODEL_PROFILE, direct_ternary_motion=True,
                 expert_scalar_dim=58, expert_film_dim=48, expert_vector_dim=20):
        super().__init__()
        if model_profile != PRODUCTION_MODEL_PROFILE or not direct_ternary_motion or dropout != 0.:
            raise ValueError('model-4.5 only executes ternary_constraints_v1; use model-v4 for historical models')
        self.config = production_model_config(node_dim, input_dim or node_dim,
            num_transformer_layers, source_conditioning, model_profile=model_profile,
            expert_scalar_dim=expert_scalar_dim, expert_film_dim=expert_film_dim,
            expert_vector_dim=expert_vector_dim)
        self.pooling = TernaryRelationEncoder(self.config['input_dim'], node_dim,
            geometric_anchors=True, refined=True, stable_anchors=True)
        self.expert_context_encoder = nn.Sequential(
            nn.Linear(2 * expert_scalar_dim + 2 + 8 * expert_vector_dim, node_dim),
            nn.SiLU(), nn.LayerNorm(node_dim))
        self.protein_geometry_encoder = nn.Sequential(nn.Linear(32, node_dim), nn.SiLU(),
                                                       nn.Linear(node_dim, node_dim))
        self.condition_query = nn.Linear(node_dim, node_dim, bias=False)
        self.condition_key = nn.Linear(node_dim, node_dim, bias=False)
        self.condition_distance = nn.Sequential(nn.Linear(2, node_dim), nn.SiLU(),
                                                 nn.Linear(node_dim, 1, bias=False))
        self.condition_value = nn.Sequential(nn.Linear(2 * node_dim + 4, node_dim), nn.SiLU(),
                                              nn.Linear(node_dim, node_dim))
        self.condition_context = nn.Sequential(nn.LayerNorm(4 * node_dim + 9),
                                               nn.Linear(4 * node_dim + 9, node_dim), nn.SiLU())
        self.condition_film = nn.ModuleList([nn.Linear(node_dim, 2 * expert_film_dim) for _ in range(3)])
        for head in self.condition_film:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.transformer = TokenNegotiator(node_dim, num_transformer_layers, 0., num_token_types=6)
        self.evidence_encoder = nn.Sequential(nn.Linear(23, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        self.source_embedding = None
        if source_conditioning:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                self.source_embedding = nn.Embedding(len(SOURCE_NAMES), node_dim)
        self.bond_context_encoder = BondContextEncoder(node_dim, chemical_context=True,
                                                       phase_harmonics=3, symmetric=True)
        from tricoopdiff.models.constraint_motion import ConstraintMotionHead
        self.constraint_head = ConstraintMotionHead(node_dim)

    @staticmethod
    def proposal_evidence(graph, proposals):
        """SE(3)-invariant strengths, agreements and alignment of binary motions."""
        vectors = torch.stack([bounded_vector(proposals[b][c]) for b in BRANCHES
                               for c in ('tr', 'rot')], dim=1)
        coords, types = graph['node_coords'], graph['node_types']
        lever = bounded_vector((coords[types == 1].mean(0) - coords[types == 2].mean(0))[None])
        gram = vectors @ vectors.transpose(-1, -2)
        triangle = torch.triu_indices(4, 4, device=vectors.device)
        alignment = (vectors * lever[:, None]).sum(-1)
        return torch.cat((gram[:, triangle[0], triangle[1]], alignment), -1)

    def _joint_condition_read(self, query, partner, geometry, attention):
        shape = geometry.shape[:2]
        values = self.condition_value(torch.cat((query[:, None].expand(*shape, -1),
                                                 partner[None].expand(*shape, -1), geometry), -1))
        return (attention[..., None] * values).sum(1)

    def _protein_geometry(self, graph):
        geometry = graph.get('protein_local_geometry')
        count = int((graph['node_types'] != 2).sum())
        if geometry is None or geometry.shape != (count, 32) or not torch.isfinite(geometry).all():
            raise ValueError('constraint model requires finite protein_local_geometry [N,32]')
        return self.protein_geometry_encoder(geometry.to(graph['node_features']))

    def expert_conditions(self, graph, diffusion_state):
        """Both binary branches read the other protein/interface, in one frame.

        Global soft correspondence has no distance cutoff, so a distant initial
        pose still sees the partner. Shared ligand nodes bridge both interfaces;
        only current input features/coordinates/time are used, never targets.
        Results use ligand-first, full-receptor order, before expert cropping.
        """
        coords, types, raw = (graph[key] for key in ('node_coords', 'node_types', 'node_features'))
        tokens = self.pooling.protein_feature_proj(raw[types != 2])
        tokens = tokens + self._protein_geometry(graph)
        ligand = self.pooling.ligand_feature_encoder(raw[types == 2])
        ligand = self.pooling.ligand_messages(
            ligand + self.pooling.type_embedding.weight[2],
            graph['lig_edge_index'], graph['lig_edge_attr'])
        proteins = [tokens[types[types != 2] == component] for component in (0, 1)]
        positions = [coords[types == component] for component in (0, 1, 2)]
        results = []
        for own in (0, 1):
            partner = 1 - own
            queries = torch.cat((ligand, proteins[own]), 0)
            query_coords = torch.cat((positions[2], positions[own]), 0)
            keys = self.condition_key(proteins[partner])
            query_centers = torch.cat((
                positions[2].mean(0).expand(len(ligand), -1),
                positions[own].mean(0).expand(len(proteins[own]), -1)), 0)
            reads = []
            for start in range(0, len(queries), 128):
                section = slice(start, start + 128)
                distance = torch.cdist(query_coords[section], positions[partner],
                                       compute_mode='donot_use_mm_for_euclid_dist')
                geometry = torch.stack((torch.log1p(distance / 10.0), 1 / (1 + distance / 10.0)), -1)
                logits = self.condition_query(queries[section]) @ keys.T / math.sqrt(keys.shape[-1])
                logits = logits + self.condition_distance(geometry).squeeze(-1)
                attention = logits.softmax(-1)
                read = attention @ proteins[partner]
                pair_geometry = self.pooling.geometry(
                    query_coords[section], positions[partner], query_centers[section],
                    positions[partner].mean(0).expand(len(proteins[partner]), -1))
                inputs = (queries[section], proteins[partner], pair_geometry, attention)
                read = read + (checkpoint(self._joint_condition_read, *inputs, use_reentrant=False)
                               if torch.is_grad_enabled() else self._joint_condition_read(*inputs))
                reads.append(read)
            opposite = torch.cat(reads, 0)
            interface = (opposite[:len(ligand)] + ligand).mean(0, keepdim=True)
            context = self.condition_context(torch.cat((
                queries, opposite, interface.expand(len(queries), -1),
                proteins[own].mean(0, keepdim=True).expand(len(queries), -1),
                diffusion_state.expand(len(queries), -1)), -1))
            results.append([head(context).chunk(2, dim=-1) for head in self.condition_film])
        return results

    def forward(self, graph, proposals, diffusion_state, source=0, *, conditions=None):
        if conditions is not None:
            raise ValueError('model-4.5 does not implement the historical PROTAC condition prototype')
        coords, types = graph['node_coords'], graph['node_types']
        expert_features = graph.get('expert_node_features')
        expected = (len(coords), 2 * self.config['expert_scalar_dim'] + 2)
        if expert_features is None or expert_features.shape != expected or not torch.isfinite(expert_features).all():
            raise ValueError(f'constraint model requires finite expert node features {expected}')
        vectors = graph.get('expert_node_vectors')
        if (vectors is None or vectors.shape != (len(coords), 2 * self.config['expert_vector_dim'], 3)
                or not torch.isfinite(vectors).all()):
            raise ValueError('constraint model requires finite node vectors from both experts')
        vectors = bounded_vector(vectors)
        centers = torch.stack([coords[types == i].mean(0) for i in range(3)])
        directions = bounded_vector(centers[None] - coords[:, None])
        alignment = torch.einsum('nvc,nkc->nvk', vectors, directions)
        invariants = torch.cat((vectors.square().sum(-1, keepdim=True), alignment), -1)
        residual = self.expert_context_encoder(torch.cat((expert_features, invariants.flatten(1)), -1))
        residual = residual.index_add(0, (types != 2).nonzero().flatten(), self._protein_geometry(graph))
        pooled = self.pooling(graph['node_features'], coords, types,
            lig_edge_index=graph['lig_edge_index'], lig_edge_attr=graph['lig_edge_attr'], node_residual=residual)
        opinions = self.proposal_evidence(graph, proposals)
        evidence = self.evidence_encoder(torch.cat((diffusion_state, opinions), -1))
        source_index = candidate_source_tensor(source, evidence)
        if self.source_embedding is not None:
            evidence = evidence + self.source_embedding(source_index)
        tokens = torch.stack((pooled['global'], pooled['e3l'], pooled['pol'],
                              pooled['ligand_tokens'][0].mean(0, keepdim=True), evidence,
                              pooled['protein_pair']), dim=1)
        context = self.transformer(tokens).mean(1)
        bonds = self.bond_context_encoder(pooled['ligand_tokens'][0], coords[types == 2],
            graph['lig_edge_index'], graph['lig_edge_mask'], graph['lig_mask_rotate'], context,
            edge_attr=graph.get('lig_edge_attr'))
        return self.constraint_head(pooled['joint_geometry'], graph, proposals, context,
                                    opinions, bonds, diffusion_state)



class ConsensusCoopDiff(nn.Module):
    """Frozen binary reads, optionally conditioned, and a full ternary field."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, *,
                 expert_input_grad=True, checkpoint_ternary=False, expert_coordinate_grad=True):
        super().__init__()
        if getattr(trifuse, 'supports_conditions', False) and self.architecture == ARCHITECTURE:
            raise ValueError('conditional fuse requires its separate model architecture')
        if any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval()
        self.trifuse = trifuse
        if getattr(trifuse, 'endpoint_v2', False):
            if (not callable(getattr(frozen_expert, 'forward_features', None))
                    or frozen_expert.node_scalar_dim != trifuse.config['expert_scalar_dim']):
                raise ValueError('v2 requires a matching frozen expert node-feature interface')
        if getattr(trifuse, 'direct_ternary_motion', False):
            self.architecture = DIRECT_ARCHITECTURE
        self.expert_input_grad = bool(expert_input_grad)
        self.expert_coordinate_grad = bool(expert_coordinate_grad)
        if getattr(trifuse, 'mutual_experts', False) and not self.expert_input_grad:
            raise ValueError('conditional experts require autograd through their frozen computation')
        if getattr(trifuse, 'enhanced_conditions', False) and frozen_expert.ns != trifuse.config['expert_film_dim']:
            raise ValueError('FiLM width must match the frozen expert leading even-scalar width')
        if getattr(trifuse, 'refined_conditions', False) and frozen_expert.node_vector_dim != trifuse.config['expert_vector_dim']:
            raise ValueError('expert vector width differs from model configuration')
        self.checkpoint_ternary = bool(checkpoint_ternary)
        self._conditioning = SDPIFlow()

    def _expert_context(self):
        # Frozen weights do not require detached input Jacobians. Respect the
        # caller's no_grad during inference; never force enable_grad here.
        return nullcontext() if self.expert_input_grad else torch.no_grad()

    def _read_expert(self, graph, film=None):
        # Stop only the frozen expert's direct geometry Jacobian. FiLM retains
        # autograd through the expert, as do the ternary geometry and updater.
        # Clone privately: detaching must not sever the shared rollout state.
        if not self.expert_coordinate_grad:
            graph = graph.clone()
            for kind in ('ligand', 'receptor'):
                graph[kind].pos = graph[kind].pos.detach()
        # Recompute frozen activations during backward rather than retaining
        # eleven expert layers at every rollout step. Clone because DiffDock
        # stores intermediate attributes on its private graph.
        def read(private_graph, film):
            if getattr(self.trifuse, 'endpoint_v2', False):
                if getattr(self.trifuse, 'refined_conditions', False):
                    return self.expert.forward_features(private_graph.clone(), film=film, return_vectors=True)
                if film is not None:
                    return self.expert.forward_features(private_graph.clone(), film=film)
                return self.expert.forward_features(private_graph.clone())
            return tuple(self.expert(private_graph.clone())[:3])

        needs_input_grad = (graph['ligand'].pos.requires_grad
                            or graph['receptor'].pos.requires_grad
                            or film is not None)
        if self.expert_input_grad and torch.is_grad_enabled() and needs_input_grad:
            return checkpoint(read, graph, film, use_reentrant=False)
        return read(graph, film)

    def _read_experts(self, requests):
        """Pack ragged complexes into one native expert call, preserving FiLM order."""
        if len(requests) == 1:
            return [self._read_expert(*requests[0])]
        from torch_geometric.data import Batch

        graphs, films = zip(*requests)
        ligand_counts = [len(graph['ligand'].pos) for graph in graphs]
        receptor_counts = [len(graph['receptor'].pos) for graph in graphs]
        torsion_counts = [0 if self.expert.model.no_torsion else int(graph['ligand'].edge_mask.sum())
                          for graph in graphs]
        packed_graphs = []
        for graph in graphs:
            graph = graph.clone()
            del graph.num_graphs
            for kind in ('ligand', 'receptor'):
                del graph[kind].batch
            if 'mask_rotate' in graph['ligand']:
                del graph['ligand'].mask_rotate  # Not consumed by native score forward.
            packed_graphs.append(graph)
        packed = Batch.from_data_list(packed_graphs)
        film = None
        if films[0] is not None:
            film = [tuple(torch.cat([values[layer][channel][:count] for values, count in zip(films, ligand_counts)]
                                    + [values[layer][channel][count:] for values, count in zip(films, ligand_counts)])
                          for channel in (0, 1)) for layer in range(len(films[0]))]
        outputs = self._read_expert(packed, film)
        torsions = outputs[2].reshape(-1).split(torsion_counts)
        features = []
        for nodes in outputs[3:]:
            ligand, receptor = nodes.split((sum(ligand_counts), sum(receptor_counts)))
            features.append([torch.cat(pair) for pair in zip(ligand.split(ligand_counts), receptor.split(receptor_counts))])
        return [(outputs[0][i:i + 1], outputs[1][i:i + 1], torsions[i],
                 *(values[i] for values in features)) for i in range(len(graphs))]

    def forward(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
                r_e3_to_poi=None, receptor_crop_cutoff=None, diffusion_state=None,
                source=0, *, conditions=None):
        return self.forward_many([dict(e3_data=e3_data, poi_data=poi_data, ligand_data=ligand_data,
                                      t_e3l=t_e3l, t_pol=t_pol, r_e3_to_poi=r_e3_to_poi,
                                      receptor_crop_cutoff=receptor_crop_cutoff,
                                      diffusion_state=diffusion_state, source=source, conditions=conditions)])[0]

    def forward_many(self, requests):
        """Pack independent binary branches together for the integrated profile."""
        generators = [self._forward_one(**request) for request in requests]
        integrated = getattr(self.trifuse, 'integrated_conditions', False)
        pending, results = {}, [None] * len(generators)
        for index, generator in enumerate(generators):
            pending[index] = next(generator)
        while pending:
            indices = list(pending)
            reads = ([request for index in indices for request in pending[index]]
                     if integrated else [pending[index] for index in indices])
            with self._expert_context():
                outputs = self._read_experts(reads)
            if integrated:
                outputs = [outputs[start:start + 2] for start in range(0, len(outputs), 2)]
            pending = {}
            for index, output in zip(indices, outputs):
                try:
                    pending[index] = generators[index].send(output)
                except StopIteration as finished:
                    results[index] = finished.value
        return results

    def _forward_one(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
                r_e3_to_poi=None, receptor_crop_cutoff=None, diffusion_state=None,
                source=0, *, conditions=None):
        if conditions is not None and not self.trifuse.supports_conditions:
            raise ValueError('legacy model cannot consume PROTAC conditions')
        if t_e3l.shape != (1,):
            raise ValueError('model-v2 currently requires B=1')
        if t_pol is not None and not torch.equal(t_e3l, t_pol):
            raise ValueError('consensus requires one shared flow clock')
        if diffusion_state is None or diffusion_state.shape != (1, 9):
            raise ValueError('model-v2 requires explicit diffusion_state [1,9]')
        frame = (r_e3_to_poi[0] if r_e3_to_poi is not None
                 else torch.eye(3, device=t_e3l.device, dtype=ligand_data['coords'].dtype))
        raw, node_reads, vector_reads = {}, [], []
        graph = build_trifuse_graph(
            e3_data['coords'][0], e3_data['features'][0],
            poi_data['coords'][0], poi_data['features'][0],
            ligand_data['coords'][0], ligand_data['features'][0],
            lig_edge_index=ligand_data['edge_index'], lig_edge_mask=ligand_data['edge_mask'],
            lig_mask_rotate=ligand_data['mask_rotate'], lig_edge_attr=ligand_data['edge_attr'])
        ternary_graph = graph
        if getattr(self.trifuse, 'integrated_conditions', False):
            geometry = []
            for name, receptor in (('E3', e3_data), ('POI', poi_data)):
                value = receptor.get('local_geometry')
                if (not isinstance(value, torch.Tensor)
                        or value.shape != (1, receptor['coords'].shape[1], 32)
                        or not torch.isfinite(value).all()):
                    raise ValueError(f'v9 requires finite {name} local_geometry [1,N,32]')
                geometry.append(value[0].to(graph['node_features']))
            graph['protein_local_geometry'] = torch.cat(geometry)
        film_conditions = None
        if getattr(self.trifuse, 'mutual_experts', False):
            # Do not retain all protein-pair attention/geometry intermediates
            # at every rollout time. Recompute the entire conditioning branch.
            film_conditions = (checkpoint(self.trifuse.expert_conditions, graph, diffusion_state,
                                          use_reentrant=False)
                               if self.checkpoint_ternary and self.training and torch.is_grad_enabled()
                               else self.trifuse.expert_conditions(graph, diffusion_state))
        requests, metadata = [], []
        # Private graph tensors ensure even in-place expert internals cannot
        # mutate the shared state or the other branch's input.
        for index, (branch, receptor) in enumerate(zip(BRANCHES, (e3_data, poi_data))):
            center = receptor['coords'][0].mean(0)
            rotation = frame if index else torch.eye(3, device=frame.device, dtype=frame.dtype)
            rec_pos = (receptor['coords'][0] - center) @ rotation
            lig_pos = (ligand_data['coords'][0] - center) @ rotation
            cutoff = (receptor_crop_cutoff[0] if receptor_crop_cutoff is not None
                      and self.expert.model.dynamic_max_cross else None)
            crop = crop_receptor_to_ligand(
                rec_pos, receptor['features'][0], lig_pos,
                rec_edge_index=receptor.get('edge_index'), cutoff=cutoff,
                min_residues=1, min_residues_with_edges=8, require_internal_edges=True)
            graph = build_binary_graph(
                crop['coords'], crop['features'], lig_pos, ligand_data['features'][0],
                rec_edge_index=crop['edge_index'], lig_edge_index=ligand_data['edge_index'],
                lig_edge_attr=ligand_data['edge_attr'], lig_edge_mask=ligand_data['edge_mask'],
                lig_mask_rotate=ligand_data['mask_rotate'])
            _set_time_embeddings(graph, t_e3l, self.expert.model)
            film = None
            if film_conditions is not None:
                n_lig = len(lig_pos)
                film = [(torch.cat((gamma[:n_lig], gamma[n_lig:][crop['mask']])),
                         torch.cat((beta[:n_lig], beta[n_lig:][crop['mask']])))
                        for gamma, beta in film_conditions[index]]
            requests.append((graph, film))
            metadata.append((branch, rotation, crop, len(rec_pos), len(lig_pos)))
        integrated = getattr(self.trifuse, 'integrated_conditions', False)
        packed_outputs = (yield requests) if integrated else None
        for index, (branch, rotation, crop, n_rec, n_lig) in enumerate(metadata):
            outputs = packed_outputs[index] if integrated else (yield requests[index])
            tr, rot, tor = outputs[:3]
            if getattr(self.trifuse, 'endpoint_v2', False):
                scalars = outputs[3]
                if (scalars.shape != (n_lig + len(crop['coords']), self.expert.node_scalar_dim)
                        or not torch.isfinite(scalars).all()):
                    raise FloatingPointError(f'{branch}: invalid frozen node features')
                visible = crop['mask']
                receptor_scalars = scalars.new_zeros(n_rec, scalars.shape[1]).index_copy(
                    0, visible.nonzero().flatten(), scalars[n_lig:])
                node_reads.append((scalars[:n_lig], receptor_scalars, visible))
                if getattr(self.trifuse, 'refined_conditions', False):
                    vectors = outputs[4] @ rotation.T
                    if not torch.isfinite(vectors).all():
                        raise FloatingPointError('non-finite expert vector features')
                    receptor_vectors = vectors.new_zeros(n_rec, *vectors.shape[1:]).index_copy(
                        0, visible.nonzero().flatten(), vectors[n_lig:])
                    vector_reads.append((vectors[:n_lig], receptor_vectors))
            if not all(torch.isfinite(v).all() for v in (tr, rot, tor)):
                raise FloatingPointError(f'{branch}: non-finite frozen proposal')
            raw[branch] = {'tr': tr @ rotation.T, 'rot': rot @ rotation.T,
                           'tor': tor.reshape(1, -1)}
        proposals = self._conditioning.normalized_predictions(
            {'interface_scores': raw, 'torsion_score': raw[BRANCHES[0]]['tor']}, t_e3l)
        graph = ternary_graph
        if node_reads:
            a, b = node_reads
            s = self.expert.node_scalar_dim
            # Preserve each interface's node identity. Invisible crop nodes
            # carry explicit validity bits, not ambiguous zero observations.
            e3 = torch.cat((a[1], a[1].new_zeros(len(a[1]), s),
                            a[2][:, None].to(a[1]), a[1].new_zeros(len(a[1]), 1)), -1)
            poi = torch.cat((b[1].new_zeros(len(b[1]), s), b[1],
                             b[1].new_zeros(len(b[1]), 1), b[2][:, None].to(b[1])), -1)
            ligand = torch.cat((a[0], b[0], a[0].new_ones(len(a[0]), 2)), -1)
            graph['expert_node_features'] = torch.cat((e3, poi, ligand), 0)
        if vector_reads:
            a, b = vector_reads
            graph['expert_node_vectors'] = torch.cat((
                torch.cat((a[1], torch.zeros_like(a[1])), 1),
                torch.cat((torch.zeros_like(b[1]), b[1]), 1), torch.cat((a[0], b[0]), 1)), 0)
        condition_kwargs = {'conditions': conditions} if conditions is not None else {}
        if self.checkpoint_ternary and self.training and torch.is_grad_enabled():
            result = checkpoint(self.trifuse, graph, proposals, diffusion_state,
                                source=source, use_reentrant=False, **condition_kwargs)
        else:
            result = self.trifuse(graph, proposals, diffusion_state, source=source, **condition_kwargs)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
