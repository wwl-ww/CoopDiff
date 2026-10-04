"""Shared categorical, bond, and ternary geometry encoders."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .diffdock.constants import lig_feature_dims


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
        direction = F.normalize(delta, dim=-1, eps=1e-8)
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

    def _pair_descriptor(self, coords_a: torch.Tensor,
                         coords_b: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(coords_a, coords_b).reshape(-1)
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
        return_contacts: bool = False,
    ):
        """Pool the closest feature-labelled contacts for one interface."""
        distances = torch.cdist(coords_a, coords_b)
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
        protein_coords: torch.Tensor,
        ligand_features: torch.Tensor,
        ligand_coords: torch.Tensor,
        pair_index: int,
    ) -> torch.Tensor:
        """Return one contact token per ligand atom with guaranteed coverage."""
        distances = torch.cdist(ligand_coords, protein_coords)
        k = min(self.contacts_per_ligand, protein_coords.shape[0])
        values, protein_indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True,
        )
        ligand_indices = torch.arange(
            ligand_coords.shape[0], device=ligand_coords.device,
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
        lig_edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return angle and SE(3)-invariant, reflection-sensitive handedness."""
        e3_nearest = torch.cdist(ligand_coords, e3_coords).argmin(dim=1)
        poi_nearest = torch.cdist(ligand_coords, poi_coords).argmin(dim=1)
        e3_direction = F.normalize(
            e3_coords[e3_nearest] - ligand_coords, dim=-1, eps=1e-8,
        )
        poi_direction = F.normalize(
            poi_coords[poi_nearest] - ligand_coords, dim=-1, eps=1e-8,
        )
        cosine = (e3_direction * poi_direction).sum(dim=-1)

        bond_direction = torch.zeros_like(ligand_coords)
        if lig_edge_index is not None and lig_edge_index.numel():
            source, target = lig_edge_index.long()
            bond_direction.index_add_(
                0, source, ligand_coords[target] - ligand_coords[source],
            )
        bond_direction = F.normalize(bond_direction, dim=-1, eps=1e-8)
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
                radius = torch.sqrt(centered.square().sum(dim=-1).mean())
                component_descriptors.extend((
                    radius / self.max_distance,
                    torch.log1p(coords.new_tensor(float(coords.shape[0]))) / 10.0,
                ))

            pair_descriptors = [
                self._pair_descriptor(component_coords[a], component_coords[b])
                for a, b in self.PAIRS
            ]
            e3_contacts = self._ligand_centric_contacts(
                component_features_by_type[0], component_coords[0],
                component_features_by_type[2], component_coords[2], 0,
            )
            pol_contacts = self._ligand_centric_contacts(
                component_features_by_type[1], component_coords[1],
                component_features_by_type[2], component_coords[2], 1,
            )
            ligand_tokens = self.ligand_contact_fusion(torch.cat((
                component_features_by_type[2], e3_contacts, pol_contacts,
                self._ligand_direction_invariants(
                    component_coords[0], component_coords[1],
                    component_coords[2], lig_edge_index,
                ),
            ), dim=-1))
            protein_pair = self._local_pair_token(
                component_features_by_type[0], component_coords[0],
                component_features_by_type[1], component_coords[1], 2,
                return_contacts=return_protein_pair_contacts,
            )
            if return_protein_pair_contacts:
                pair_token, contact_tokens, left, right = protein_pair
                protein_pair_outputs.append({
                    'tokens': contact_tokens,
                    'direction': F.normalize(right - left, dim=-1, eps=1e-8),
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
