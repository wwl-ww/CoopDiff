"""
Frozen DiffDock two-body expert wrapper.

Wraps DiffDock's CGModel with all parameters frozen. Exposes per-layer outputs
for FiLM modulation injection. The expert processes two physically disconnected
subgraphs (E3-L and POI-L) in separate forward passes.

Key constraint: each subgraph contains exactly one protein + PROTAC, matching
DiffDock's pretraining input distribution exactly (zero OOD).
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from e3nn import o3
from torch_scatter import scatter_mean


class FrozenDiffDock(nn.Module):
    """
    Wrapper around DiffDock's CGModel that:
    1. Freezes all parameters (no gradient)
    2. Exposes per-layer outputs for FiLM injection
    3. Provides a step_conv_layer() interface for external orchestration

    The TriCoopDiff model orchestrates:
    - Running the embedding phase once
    - Running each conv layer individually, applying FiLM between layers
    - Running the final score heads
    """

    def __init__(self, cg_model: nn.Module, num_conv_layers: int = 6):
        """
        Args:
            cg_model: DiffDock's CGModel with pretrained weights loaded
            num_conv_layers: number of conv layers (typically 6 for DiffDock-L)
        """
        super().__init__()
        self.model = cg_model
        self.num_conv_layers = num_conv_layers

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model references for convenience
        self.ns = self.model.ns
        self.nv = self.model.nv
        self.sh_irreps = self.model.sh_irreps
        self.device = self.model.device

    def embedding(self, data):
        """
        Run the embedding phase of CGModel.
        Returns initial ligand and receptor node features plus edge info.

        This is a thin wrapper around CGModel.embedding().
        """
        return self.model.embedding(data)

    def build_cross_graph(self, data, cross_cutoff):
        """Build cross edges between ligand and receptor."""
        return self.model.build_cross_conv_graph(data, cross_cutoff)

    def build_center_graph(self, data):
        """Build center convolution graph for tr/rot score prediction."""
        return self.model.build_center_conv_graph(data)

    def build_bond_graph(self, data):
        """Build bond convolution graph for torsion score prediction."""
        return self.model.build_bond_conv_graph(data)

    def step_conv_layer(
        self,
        layer_idx: int,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr,
        edge_sh: torch.Tensor,
        edge_weight=None,
    ) -> torch.Tensor:
        """
        Run a single conv layer. This is the core per-layer interface
        that allows FiLM modulation between layers.

        Args:
            layer_idx: which conv layer to run (0-indexed)
            node_attr: [N_total, D] current node features (ligand + receptor)
            edge_index: [2, E] combined edge index
            edge_attr: edge features (tensor or list of tensors for multi-group)
            edge_sh: spherical harmonic edge embeddings
            edge_weight: optional edge weights

        Returns:
            updated node_attr [N_total, D']
        """
        return self.model.conv_layers[layer_idx](
            node_attr, edge_index, edge_attr, edge_sh, edge_weight=edge_weight
        )

    def compute_scores(
        self,
        data,
        lig_node_attr: torch.Tensor,
        graph_sigma_emb: torch.Tensor,
        tr_sigma: torch.Tensor,
        rot_sigma: torch.Tensor,
        tor_sigma: torch.Tensor,
    ) -> dict:
        """
        Compute final tr/rot/torsion scores from ligand node features.

        Returns:
            dict with keys: 'tr_score', 'rot_score', 'tor_score'
        """
        model = self.model

        # Build center convolution for tr/rot scores
        center_edge_index, center_edge_attr, center_edge_sh = model.build_center_conv_graph(data)
        center_edge_attr = model.center_edge_embedding(center_edge_attr)
        if model.fixed_center_conv:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1
            )
        else:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1
            )
        global_pred = model.final_conv(
            lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh,
            out_nodes=data.num_graphs
        )

        tr_pred = global_pred[:, :3] + (
            global_pred[:, 6:9] if not model.odd_parity else 0
        )
        rot_pred = global_pred[:, 3:6] + (
            global_pred[:, 9:] if not model.odd_parity else 0
        )

        # Fix magnitude of tr/rot scores
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * model.tr_final_layer(
            torch.cat([tr_norm, graph_sigma_emb], dim=1)
        )
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * model.rot_final_layer(
            torch.cat([rot_norm, graph_sigma_emb], dim=1)
        )

        if model.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = (
                rot_pred
                * model.so3_score_norm(rot_sigma.cpu()).unsqueeze(1).to(lig_node_attr.device)
            )

        # Torsion scores
        if model.no_torsion or data['ligand'].edge_mask.sum() == 0:
            tor_pred = torch.empty(0, device=lig_node_attr.device)
        else:
            tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = \
                model.build_bond_conv_graph(data)
            tor_bond_vec = (
                data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
            )
            tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

            tor_bonds_sh = o3.spherical_harmonics(
                "2e", tor_bond_vec, normalize=True, normalization='component'
            )
            tor_edge_sh = model.final_tp_tor(
                tor_edge_sh, tor_bonds_sh[tor_edge_index[0]]
            )

            tor_edge_attr = torch.cat([
                tor_edge_attr,
                lig_node_attr[tor_edge_index[1], :self.ns],
                tor_bond_attr[tor_edge_index[0], :self.ns],
            ], -1)
            tor_pred = model.tor_bond_conv(
                lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                out_nodes=data['ligand'].edge_mask.sum(), reduce='mean',
                edge_weight=tor_edge_weight
            )
            tor_pred = model.tor_final_layer(tor_pred).squeeze(1)
            edge_sigma = tor_sigma[
                data['ligand'].batch
            ][
                data['ligand', 'ligand'].edge_index[0]
            ][
                data['ligand'].edge_mask
            ]

            if model.scale_by_sigma:
                tor_pred = tor_pred * torch.sqrt(
                    torch.tensor(
                        model.tor_score_norm(edge_sigma.cpu().numpy())
                    ).float().to(lig_node_attr.device)
                )

        return {
            'tr_score': tr_pred,
            'rot_score': rot_pred,
            'tor_score': tor_pred,
        }

    def so3_score_norm(self, sigma):
        """Compute SO(3) score normalization factor."""
        # Borrowed from DiffDock's utils/so3.py
        sigma = torch.tensor(sigma) if not torch.is_tensor(sigma) else sigma
        return torch.sqrt(
            sigma**2 * (3 + sigma**2) / (1 + sigma**2)**2
        ) * (1 + sigma**2) / sigma**2

    def tor_score_norm(self, sigma):
        """Compute torsion score normalization factor."""
        # Borrowed from DiffDock's utils/torus.py
        return 1.0 / (sigma**2)

    def forward(self, data):
        """
        Standard forward (without FiLM). Used for verification that
        FiLM is transparent when zero-initialized and for L0 ablation.
        """
        return self.model(data)
