"""
FiLM (Feature-wise Linear Modulation) for injecting TriBrain coordination
signals into the frozen two-body expert.

Follows the DiT adaLN-Zero paradigm: each FiLM head is zero-initialized so that
at training start, the FiLM modulation is transparent (h' = h, output identical
to native DiffDock).
"""

import torch
import torch.nn as nn


class FiLMSharedMLP(nn.Module):
    """
    Shared MLP that maps TriBrain's pooled condition vector to a shared
    conditioning embedding, which is then consumed by per-layer FiLM heads.
    """

    def __init__(self, cond_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_cond: [B, cond_dim] pooled condition from TriBrain
        Returns:
            [B, hidden_dim] shared conditioning embedding
        """
        return self.mlp(h_cond)


class FiLMLayerHead(nn.Module):
    """
    Per-layer FiLM head. Takes the shared conditioning embedding and produces
    (gamma, beta) modulation parameters for one conv layer of one subgraph.

    Zero-initialized: the last Linear weight and bias are zero, so at init:
        h' = h * (1 + 0) + 0 = h  (transparent)
    """

    def __init__(self, node_dim: int, cond_dim: int = 256):
        """
        Args:
            node_dim: dimension of node features being modulated
            cond_dim: dimension of shared conditioning embedding
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, node_dim * 2),
        )
        # Zero-initialize for transparent modulation at training start
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, h_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_cond: [B, cond_dim] or [N, cond_dim] conditioning
        Returns:
            gamma: [B/N, node_dim] scale modulation
            beta:  [B/N, node_dim] shift modulation
        """
        out = self.net(h_cond)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


def apply_film(
    node_attr: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    node_mask: torch.Tensor = None,
    scalar_dim: int = None,
) -> torch.Tensor:
    """
    Apply FiLM modulation: h' = h * (1 + gamma) + beta

    Only the leading `scalar_dim` channels of each node are modulated. This
    restricts FiLM to the scalar (0e) block of the e3nn node features, which is
    required to preserve SE(3) equivariance of the tensor-product conv layers.

    Args:
        node_attr: [N, D] node features
        gamma: [B, scalar_dim] scale modulation (broadcast over masked nodes)
        beta: [B, scalar_dim] shift modulation
        node_mask: [N] bool, if provided, only modulate masked nodes (e.g., ligand only)
        scalar_dim: number of leading scalar channels to modulate. Defaults to
                    full feature width (legacy behaviour).

    Returns:
        modulated node features [N, D]
    """
    if scalar_dim is None:
        scalar_dim = node_attr.shape[-1]

    if node_mask is not None:
        # Only modulate the specified subset of nodes (e.g., ligand atoms)
        modulated = node_attr.clone()
        # gamma/beta: [B, scalar_dim] broadcast over the masked (ligand) nodes.
        modulated[node_mask, :scalar_dim] = (
            modulated[node_mask, :scalar_dim] * (1 + gamma) + beta
        )
        return modulated

    out = node_attr.clone()
    out[:, :scalar_dim] = out[:, :scalar_dim] * (1 + gamma) + beta
    return out
