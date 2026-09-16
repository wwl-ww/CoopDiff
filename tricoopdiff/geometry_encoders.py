"""Shared categorical, bond, and ternary geometry encoders."""

import math

import torch
from torch import nn
from torch.nn import functional as F

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
    transform; equivariant rigid scores remain owned by the experts.
    """

    PAIRS = ((0, 2), (1, 2), (0, 1))  # E3-L, POI-L, E3-POI

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_rbf: int = 16, max_distance: float = 40.0,
                 max_pair_tokens: int = 24, contacts_per_ligand: int = 4):
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

    def _pair_descriptor(self, coords_a: torch.Tensor,
                         coords_b: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(coords_a, coords_b).reshape(-1)
        offsets = distances[:, None] - self.rbf_centers[None, :]
        rbf = torch.exp(-0.5 * (offsets / self.rbf_width) ** 2).mean(dim=0)
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
    ) -> torch.Tensor:
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
        return (
            (pair_features * attention[:, None]).sum(dim=0)
            + pair_features.mean(dim=0)
            + pair_features.max(dim=0).values
        ) / 3.0

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
                lig_edge_index: torch.Tensor = None, node_residual: torch.Tensor = None) -> dict:
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

        global_outputs = []
        e3l_outputs = []
        pol_outputs = []
        ligand_token_outputs = []
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
            local_pair_tokens = [
                e3_contacts.mean(dim=0),
                pol_contacts.mean(dim=0),
                self._local_pair_token(
                    component_features_by_type[0], component_coords[0],
                    component_features_by_type[1], component_coords[1], 2,
                ),
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
        return {
            "global": torch.stack(global_outputs, dim=0),
            "e3l": torch.stack(e3l_outputs, dim=0),
            "pol": torch.stack(pol_outputs, dim=0),
            "ligand_tokens": ligand_token_outputs,
        }
