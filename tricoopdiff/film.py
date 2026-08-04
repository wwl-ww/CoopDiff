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
) -> torch.Tensor:
    """
    Apply FiLM modulation: h' = h * (1 + gamma) + beta

    Args:
        node_attr: [N, D] node features
        gamma: [N, D] or [B, D] scale modulation
        beta: [N, D] or [B, D] shift modulation
        node_mask: [N] bool, if provided, only modulate masked nodes (e.g., ligand only)

    Returns:
        modulated node features [N, D]
    """
    if node_mask is not None:
        # Only modulate the specified subset of nodes (e.g., ligand atoms)
        modulated = node_attr.clone()
        modulated[node_mask] = modulated[node_mask] * (1 + gamma) + beta
        return modulated
    return node_attr * (1 + gamma) + beta
