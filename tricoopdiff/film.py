"""
FiLM (Feature-wise Linear Modulation) for injecting TriFuse coordination
signals into the frozen two-body expert.

Follows the DiT adaLN-Zero paradigm: each FiLM head is zero-initialized so that
at training start, the FiLM modulation is transparent (h' = h, output identical
to native DiffDock).

NOTE: only ``apply_film`` from this module is used by the production path
(model.py:_expert_forward_with_film). The actual FiLM (γ, β) heads live in
``trifuse.py:FiLMHead``. The former standalone ``FiLMSharedMLP`` /
``FiLMLayerHead`` classes here were dead code and have been removed.
"""

import torch


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
