"""Frozen binary reads and trainable ternary motion.

Legacy checkpoints retain the dual-interface residual field. New direct mode
uses node-level ternary geometry to decide independent ligand/POI centroid
motions; expert proposals are context only. All vectors use the E3 frame.
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
            indices = features[:, column].long().clamp_(0, size - 1)
            encoded = encoded + embedding(indices)
        return self.norm(encoded / math.sqrt(len(self.embeddings)))


class BondContextEncoder(nn.Module):
    """Build one context token per canonical rotatable bond."""

    def __init__(self, hidden_dim: int, n_rbf: int = 16,
                 max_distance: float = 8.0):
        super().__init__()
        self.register_buffer(
            "rbf_centers", torch.linspace(0.0, max_distance, n_rbf),
        )
        self.rbf_width = max_distance / max(n_rbf - 1, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + n_rbf + 2),
            nn.Linear(hidden_dim * 4 + n_rbf + 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, ligand_tokens: torch.Tensor, ligand_coords: torch.Tensor,
                edge_index: torch.Tensor, edge_mask: torch.Tensor,
                mask_rotate, global_context: torch.Tensor) -> torch.Tensor:
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
        token = self.net(torch.cat((
            left, right, (left - right).abs(), global_per_bond,
            distance_rbf, fragment,
        ), dim=-1))
        return token.unsqueeze(0)


class TernaryGeometryEncoder(nn.Module):
    """Encode typed components and their chemically labelled local contacts.

    Global distance histograms alone cannot distinguish which residue or atom
    participates in an interface.  For each component pair we therefore select
    the nearest ``max_pair_tokens`` contacts and fuse both endpoint features
    with an RBF encoding of their separation.  Only scalar distances enter this
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
                 global_node_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        """Typed kNN messages, including every component even when far apart.

        Edges point from receiver to sender. Distances encode scalar features;
        directions are retained for the equivariant motion readout, not pooled.
        """
        receivers, senders = [], []
        all_indices = torch.arange(len(coords), device=coords.device)
        for component in range(3):
            candidates = all_indices[types == component]
            if not len(candidates):
                raise ValueError('joint geometry requires all three components')
            distances = torch.cdist(coords, coords[candidates])
            distances = distances.masked_fill(all_indices[:, None] == candidates[None], float('inf'))
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
        direction = delta / distance[:, None].clamp_min(.1)
        if self.spatial_geometry:
            geometry = torch.cat((geometry, self._spatial_edge_geometry(
                coords, types, receiver, direction)), dim=-1)
        counts = features.new_zeros(len(coords), 1).index_add(
            0, receiver, features.new_ones(len(receiver), 1)).clamp_min(1)
        for layer, (message, update) in enumerate(zip(self.messages, self.message_updates)):
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
        statistics = torch.stack((
            distances.min() / scale,
            distances.mean() / scale,
            distances.std(unbiased=False) / scale,
            (distances < 4.0).to(distances.dtype).mean(),
            (distances < 8.0).to(distances.dtype).mean(),
            (distances < 12.0).to(distances.dtype).mean(),
        ))
        return torch.cat((rbf, statistics), dim=0)

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
        """Pool the closest feature-labelled contacts for one interface."""
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
        e3_nearest = distances[0].argmin(dim=0)
        poi_nearest = distances[1].argmin(dim=0)
        e3_direction = F.normalize(
            e3_coords[e3_nearest] - ligand_coords, dim=-1, eps=.1,
        )
        poi_direction = F.normalize(
            poi_coords[poi_nearest] - ligand_coords, dim=-1, eps=.1,
        )
        cosine = (e3_direction * poi_direction).sum(dim=-1)

        bond_direction = torch.zeros_like(ligand_coords)
        if lig_edge_index is not None and lig_edge_index.numel():
            source, target = lig_edge_index.long()
            bond_direction.index_add_(
                0, source, ligand_coords[target] - ligand_coords[source],
            )
        bond_direction = F.normalize(bond_direction, dim=-1, eps=.1)
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
                token = (
                    (component_features * attention[:, None]).sum(dim=0)
                    + component_features.mean(dim=0)
                    + component_features.max(dim=0).values
                ) / 3.0
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
                torch.cdist(component_coords[a], component_coords[b])
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
                    'direction': F.normalize(right - left, dim=-1, eps=.1),
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


BRANCHES = ('e3_ligand', 'poi_ligand')


def bounded_vector(value):
    """Isotropic vector compression, preserving rotation equivariance."""
    return value / (1 + torch.linalg.vector_norm(value, dim=-1, keepdim=True))


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


class LegacyResidualMotion:
    """Historical checkpoint behavior, isolated from the direct motion path.

    A mixin keeps historical parameter names unchanged. No legacy head is
    instantiated when the production controller selects independent motion.
    """

    def _init_legacy_motion(self, node_dim):
        self.gate_head = nn.Linear(node_dim, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.)
        self.residual_head = nn.Linear(node_dim, 4 * 8)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.bond_context_encoder = BondContextEncoder(node_dim)
        self.torsion_coordinator = TorsionCoordinator(node_dim)
        self.judge_head = JudgeHead(node_dim).requires_grad_(False)
        self.state_head = StateHead(node_dim).requires_grad_(False)
        if self.protein_pair_motion:
            self.protein_pair_head = nn.Sequential(
                nn.Linear(2 * node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, 4))
            nn.init.zeros_(self.protein_pair_head[-1].weight)
            nn.init.zeros_(self.protein_pair_head[-1].bias)

    def _legacy_motion(self, pooled, graph, proposals, context, basis, raw,
                       condition_gate_bias, source_context):
        coords, types = graph['node_coords'], graph['node_types']
        gate_logits = self.gate_head(context)
        if condition_gate_bias is not None:
            gate_logits = gate_logits + condition_gate_bias
        if source_context is not None:
            gate_logits = gate_logits + self.source_gate_head(source_context)
        gates = torch.sigmoid(gate_logits)
        coefficients = self.residual_head(context).reshape(-1, 4, 8)
        residual = torch.einsum('bik,bkj->bij', coefficients, basis)
        pair_motion = torch.zeros_like(residual)
        if self.protein_pair_motion:
            contacts = pooled['protein_pair_contacts'][0]
            weights = self.protein_pair_head(torch.cat((
                contacts['tokens'], context.expand(len(contacts['tokens']), -1)), dim=-1))
            direction = contacts['direction']
            torque = torch.cross(bounded_vector(contacts['lever']), direction, dim=-1)
            pair_motion = torch.stack([
                (weights[:, i:i+1] * (direction if i % 2 == 0 else torque)).mean(0)
                for i in range(4)], dim=0).unsqueeze(0)
            residual = residual + pair_motion
        prior = gates.unsqueeze(-1) * torch.stack(raw, 1)
        field = prior + residual
        bond_context = self.bond_context_encoder(
            pooled['ligand_tokens'][0], coords[types == 2], graph['lig_edge_index'],
            graph['lig_edge_mask'], graph['lig_mask_rotate'], context)
        tor_a, tor_b = (proposals[b]['tor'] for b in BRANCHES)
        torsion, tor_gate, tor_residual = self.torsion_coordinator(bond_context, tor_a, tor_b)
        predictions = {
            branch: {'tr': field[:, 2 * i], 'rot': field[:, 2 * i + 1],
                     'tor': proposals[branch]['tor']}
            for i, branch in enumerate(BRANCHES)}
        predictions['torsion'] = torsion
        residual_penalty = residual.square().mean()
        if tor_residual.numel():
            residual_penalty = residual_penalty + tor_residual.square().mean()
        conflicts = torch.stack([
            1 - torch.nn.functional.cosine_similarity(raw[i], raw[i + 2], dim=-1)
            for i in (0, 1)], dim=-1)
        return {
            'normalized_velocity': predictions, 'rank_features': context,
            'residual_penalty': residual_penalty,
            'consensus_diagnostics': {
                'rigid_gates': gates.detach(), 'direction_disagreement': conflicts.detach(),
                'rigid_prior_rms': prior.detach().square().mean().sqrt(),
                'rigid_residual_rms': residual.detach().square().mean().sqrt(),
                'protein_pair_motion_rms': pair_motion.detach().square().mean().sqrt(),
                'torsion_gate': tor_gate.detach(),
                'torsion_disagreement': (tor_a - tor_b).detach().abs()},
            'torsion_coordination': {'e3l_weight': tor_gate, 'residual': tor_residual},
            'confidence': self.judge_head(context), 'state': self.state_head(context)}


ARCHITECTURE = 'residual_consensus_v2'
DIRECT_ARCHITECTURE = 'ternary_geometry_independent_v1'
CONSENSUS_CHECKPOINT_VERSION = 23
INTERFACE_OBJECTIVE_CHECKPOINT_VERSION = 22
OBJECTIVE_CONTRACT_CHECKPOINT_VERSION = 21
ENDPOINT_PROTOCOL_CHECKPOINT_VERSION = 20
LEGACY_CONSENSUS_CHECKPOINT_VERSION = 18
PREVIOUS_CONSENSUS_CHECKPOINT_VERSION = 19
SOURCE_NAMES = ('sdpi', 'bitseed')
SOURCE_CONTRACT_VERSION = 'candidate_source_conditioning_v1'
PRODUCTION_MODEL_PROFILE = 'ternary_endpoint_v1'


def production_model_config(node_dim, input_dim, num_transformer_layers=3,
                            source_conditioning=False):
    """One explicit new-model recipe; older weights retain their own config."""
    return dict(node_dim=node_dim, input_dim=input_dim,
                num_transformer_layers=num_transformer_layers, dropout=0.,
                source_conditioning=bool(source_conditioning),
                direct_ternary_motion=True, global_node_context=True,
                spatial_geometry=True, motion_attention=True,
                global_node_attention=True, rotation_scale_normalization=True,
                model_profile=PRODUCTION_MODEL_PROFILE)


def rotation_levers(coords, normalize_body_scale=False):
    """Dimensionless centered levers; clamp squared radius before sqrt."""
    centered = coords - coords.mean(0)
    scale = (centered.square().sum(-1).mean().clamp_min(1.).sqrt()
             if normalize_body_scale else 10.)
    return bounded_vector(centered / scale)


def candidate_source_tensor(source, reference):
    """Normalize one immutable trajectory source to a ``[1]`` class tensor."""
    if isinstance(source, str):
        if source not in SOURCE_NAMES:
            raise ValueError(f'unknown candidate source: {source!r}')
        source = SOURCE_NAMES.index(source)
    value = torch.as_tensor(source, device=reference.device, dtype=torch.long).reshape(-1)
    if value.shape != (1,) or not bool(((value >= 0) & (value < len(SOURCE_NAMES))).all()):
        raise ValueError('candidate source must be one SDPI/BiTSeed class')
    return value


def attentive_interface_motion(weights, direction, torque, sender_types):
    """Separate interface/channel attention; large partners cannot win by count."""
    fields = torch.stack((direction, torque), dim=1)
    interfaces = []
    for component in sender_types.unique():
        selected = sender_types == component
        attention = weights[selected, 2:4].softmax(dim=0)
        interfaces.append((attention[..., None] * weights[selected, :2, None]
                           * fields[selected]).sum(0))
    motion = torch.stack(interfaces).mean(0)
    return motion[:1], motion[1:]


class ConsensusFuse(LegacyResidualMotion, nn.Module):
    """Decide motion from ternary geometry, with an explicit legacy mode."""

    supports_conditions = False

    def __init__(self, node_dim=128, input_dim=None, num_transformer_layers=3,
                 dropout=0.1, source_conditioning=False,
                 protein_pair_motion=False,
                 direct_ternary_motion=False,
                 global_node_context=False,
                 spatial_geometry=False,
                 motion_attention=False,
                 global_node_attention=False,
                 rotation_scale_normalization=False,
                 model_profile=None):
        super().__init__()
        self.config = dict(node_dim=node_dim, input_dim=input_dim or node_dim,
                           num_transformer_layers=num_transformer_layers, dropout=dropout,
                           source_conditioning=bool(source_conditioning))
        self.protein_pair_motion = bool(protein_pair_motion)
        self.direct_ternary_motion = bool(direct_ternary_motion)
        self.motion_attention = bool(motion_attention)
        self.rotation_scale_normalization = bool(rotation_scale_normalization)
        for name, enabled in (('global_node_context', global_node_context),
                              ('spatial_geometry', spatial_geometry),
                              ('motion_attention', motion_attention),
                              ('global_node_attention', global_node_attention),
                              ('rotation_scale_normalization', rotation_scale_normalization)):
            if enabled:
                if not self.direct_ternary_motion:
                    raise ValueError(f'{name} requires direct ternary motion')
                self.config[name] = True
        if self.direct_ternary_motion and self.protein_pair_motion:
            raise ValueError('direct ternary motion replaces the protein-pair residual path')
        if self.direct_ternary_motion:
            self.config['direct_ternary_motion'] = True
        if self.protein_pair_motion:
            self.config['protein_pair_motion'] = True
        if model_profile is not None:
            if model_profile != PRODUCTION_MODEL_PROFILE:
                raise ValueError(f'unknown model profile: {model_profile}')
            self.config['model_profile'] = model_profile
            expected = production_model_config(node_dim, self.config['input_dim'],
                                               num_transformer_layers, source_conditioning)
            if self.config != expected:
                raise ValueError('production model profile cannot silently change its architecture or dropout')
        self.pooling = TernaryGeometryEncoder(self.config['input_dim'], node_dim,
                                             joint_geometry=self.direct_ternary_motion,
                                             global_node_context=global_node_context,
                                             spatial_geometry=spatial_geometry,
                                             global_node_attention=global_node_attention)
        self.transformer = TokenNegotiator(node_dim, num_transformer_layers, dropout)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(9 + 8 * 8, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        if source_conditioning:
            # Source-only parameters must not perturb the RNG sequence used by
            # shared B/C parameters; otherwise the factorial comparison starts
            # from different common weights.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.source_embedding = nn.Embedding(len(SOURCE_NAMES), node_dim)
                if not self.direct_ternary_motion:
                    self.source_gate_head = nn.Linear(node_dim, 4, bias=False)
                    nn.init.normal_(self.source_gate_head.weight, std=0.01)
        else:
            self.source_embedding = None
        if not source_conditioning or self.direct_ternary_motion:
            self.source_gate_head = None
        if not self.direct_ternary_motion:
            self._init_legacy_motion(node_dim)
        else:
            self.bond_context_encoder = BondContextEncoder(node_dim)
            self.motion_head = nn.Sequential(
                nn.Linear(3 * node_dim + self.pooling.edge_geometry_dim, node_dim),
                nn.SiLU(), nn.Linear(node_dim, 4 if self.motion_attention else 2))
            self.direct_torsion_head = nn.Sequential(
                nn.Linear(node_dim + 2, node_dim), nn.SiLU(), nn.Linear(node_dim, 1))
            for head in (self.motion_head, self.direct_torsion_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def active_state_dict(self, state):
        """Migrate old direct weights by dropping only known inactive heads."""
        if not self.direct_ternary_motion:
            return dict(state)
        obsolete = ('gate_head.', 'residual_head.', 'source_gate_head.',
                    'torsion_coordinator.', 'judge_head.', 'state_head.')
        return {name: value for name, value in state.items() if not name.startswith(obsolete)}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Works both for loading the fuse alone and as a child of the full model.
        local = {name[len(prefix):]: value for name, value in state_dict.items()
                 if name.startswith(prefix)}
        for name in local.keys() - self.active_state_dict(local).keys():
            del state_dict[prefix + name]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _direct_motion(self, pooled, graph, proposals, context):
        """Experts supply context only; all executable motion is learned here.

        Historical branch slots now denote ligand and POI centroid velocities.
        The explicit output-mode tag changes the decoder and teacher labels;
        it must survive scaling and sampling, never be inferred from slot names.
        """
        from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE

        joint = pooled['joint_geometry']
        receiver, sender = joint['receiver'], joint['sender']
        coords, types = graph['node_coords'], graph['node_types']
        weights = self.motion_head(torch.cat((
            joint['features'][receiver], joint['features'][sender],
            context.expand(len(receiver), -1), joint['geometry']), -1))
        predictions = {'rigid_output_mode': INDEPENDENT_RIGID_MODE}
        fields = []
        for branch, component in zip(BRANCHES, (2, 1)):
            selected = types[receiver] == component
            if self.motion_attention:
                # Internal edges describe body shape, not an external interface
                # driving rigid motion. Keep them in messages, not this readout.
                selected = selected & (types[sender] != component)
            direction = joint['direction'][selected]
            component_mask = types == component
            component_levers = rotation_levers(coords[component_mask], self.rotation_scale_normalization)
            all_levers = torch.zeros_like(coords).index_copy(0, component_mask.nonzero().flatten(), component_levers)
            lever = all_levers[receiver[selected]]
            torque = torch.cross(lever, direction, dim=-1)
            if self.motion_attention:
                tr, rot = attentive_interface_motion(
                    weights[selected], direction, torque, types[sender[selected]])
            else:
                tr = (weights[selected, 0:1] * direction).mean(0, keepdim=True)
                rot = (weights[selected, 1:2] * torque).mean(0, keepdim=True)
            predictions[branch] = {'tr': tr, 'rot': rot, 'tor': proposals[branch]['tor']}
            fields.extend((tr, rot))
        bonds = self.bond_context_encoder(
            pooled['ligand_tokens'][0], coords[types == 2], graph['lig_edge_index'],
            graph['lig_edge_mask'], graph['lig_mask_rotate'], context)
        tor_a, tor_b = (proposals[b]['tor'] for b in BRANCHES)
        opinions = torch.stack((tor_a.tanh(), tor_b.tanh()), -1)
        predictions['torsion'] = self.direct_torsion_head(torch.cat((bonds, opinions), -1)).squeeze(-1)
        return {'normalized_velocity': predictions, 'rank_features': context,
                'residual_penalty': context.new_zeros(()),
                'consensus_diagnostics': {
                    'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                    'rigid_motion_rms': torch.stack(fields).detach().square().mean().sqrt()}}

    def prepare_conditions(self, graph, conditions):
        raise ValueError('legacy ConsensusFuse does not accept PROTAC conditions')

    def forward(self, graph, proposals, diffusion_state, source=0, *, conditions=None):
        condition_context, condition_gate_bias = None, None
        if conditions is not None:
            graph, condition_context, condition_gate_bias = self.prepare_conditions(graph, conditions)
        coords, types = graph['node_coords'], graph['node_types']
        pooling_kwargs = {'node_residual': graph['condition_node_residual']} if conditions is not None else {}
        if self.protein_pair_motion:
            pooling_kwargs['return_protein_pair_contacts'] = True
        pooled = self.pooling(graph['node_features'], coords, types,
                              lig_edge_index=graph['lig_edge_index'], **pooling_kwargs)
        centers = [coords[types == i].mean(0, keepdim=True) for i in range(3)]
        directions = [bounded_vector((centers[i] - centers[2]) / 10) for i in (0, 1)]
        raw = [proposals[b][c] for b in BRANCHES for c in ('tr', 'rot')]
        # Cross products provide an independent geometry direction and axial channel.
        geometry_cross = torch.cross(directions[0], directions[1], dim=-1)
        proposal_cross = torch.cross(bounded_vector(raw[0]), bounded_vector(raw[2]), dim=-1)
        basis = torch.stack([bounded_vector(v) for v in raw] + directions
                            + [geometry_cross, proposal_cross], dim=1)
        gram = basis @ basis.transpose(-1, -2)
        evidence = self.evidence_encoder(torch.cat((diffusion_state, gram.flatten(1)), -1))
        if condition_context is not None:
            evidence = evidence + condition_context
        source_index = candidate_source_tensor(source, evidence)
        source_context = None
        if self.source_embedding is not None:
            source_context = self.source_embedding(source_index)
            evidence = evidence + source_context
        tokens = torch.stack((pooled['global'], pooled['e3l'], pooled['pol'],
                              pooled['ligand_tokens'][0].mean(0, keepdim=True), evidence), dim=1)
        negotiated = self.transformer(tokens)
        context = negotiated.mean(1)
        if self.direct_ternary_motion:
            return self._direct_motion(pooled, graph, proposals, context)
        return self._legacy_motion(pooled, graph, proposals, context, basis, raw,
                                   condition_gate_bias, source_context)


class ConsensusCoopDiff(nn.Module):
    """Frozen independent DiffDock reads followed by a ternary motion field."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, *,
                 expert_input_grad=True, checkpoint_ternary=False):
        super().__init__()
        if getattr(trifuse, 'supports_conditions', False) and self.architecture == ARCHITECTURE:
            raise ValueError('conditional fuse requires its separate model architecture')
        if any('lora' in name for name, _ in frozen_expert.named_parameters()):
            raise ValueError('model-v2 Stage 1 requires an unmodified frozen expert')
        self.expert = frozen_expert.requires_grad_(False).eval()
        self.trifuse = trifuse
        if getattr(trifuse, 'direct_ternary_motion', False):
            self.architecture = DIRECT_ARCHITECTURE
        self.expert_input_grad = bool(expert_input_grad)
        self.checkpoint_ternary = bool(checkpoint_ternary)
        self._conditioning = SDPIFlow()

    def _expert_context(self):
        # Frozen weights do not require detached input Jacobians. Respect the
        # caller's no_grad during inference; never force enable_grad here.
        return nullcontext() if self.expert_input_grad else torch.no_grad()

    def _read_expert(self, graph):
        # Recompute frozen activations during backward rather than retaining
        # eleven expert layers at every rollout step. Clone because DiffDock
        # stores intermediate attributes on its private graph.
        def read(private_graph):
            return tuple(self.expert(private_graph.clone())[:3])

        needs_input_grad = (graph['ligand'].pos.requires_grad
                            or graph['receptor'].pos.requires_grad)
        if self.expert_input_grad and torch.is_grad_enabled() and needs_input_grad:
            return checkpoint(read, graph, use_reentrant=False)
        return read(graph)

    def forward(self, e3_data, poi_data, ligand_data, t_e3l, t_pol=None,
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
        raw = {}
        # Private graph tensors ensure even in-place expert internals cannot
        # mutate the shared state or the other branch's input.
        with self._expert_context():
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
                tr, rot, tor = self._read_expert(graph)
                if not all(torch.isfinite(v).all() for v in (tr, rot, tor)):
                    raise FloatingPointError(f'{branch}: non-finite frozen proposal')
                raw[branch] = {'tr': tr @ rotation.T, 'rot': rot @ rotation.T,
                               'tor': tor.reshape(1, -1)}
            proposals = self._conditioning.normalized_predictions(
                {'interface_scores': raw, 'torsion_score': raw[BRANCHES[0]]['tor']}, t_e3l)
        graph = build_trifuse_graph(
            e3_data['coords'][0], e3_data['features'][0],
            poi_data['coords'][0], poi_data['features'][0],
            ligand_data['coords'][0], ligand_data['features'][0],
            lig_edge_index=ligand_data['edge_index'], lig_edge_mask=ligand_data['edge_mask'],
            lig_mask_rotate=ligand_data['mask_rotate'])
        condition_kwargs = {'conditions': conditions} if conditions is not None else {}
        if self.checkpoint_ternary and self.training and torch.is_grad_enabled():
            result = checkpoint(self.trifuse, graph, proposals, diffusion_state,
                                source=source, use_reentrant=False, **condition_kwargs)
        else:
            result = self.trifuse(graph, proposals, diffusion_state, source=source, **condition_kwargs)
        # Kept explicitly as diagnostic evidence, never consumed as another update.
        result['binary_proposals'] = proposals
        return result
