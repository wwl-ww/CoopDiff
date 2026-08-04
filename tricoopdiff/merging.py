"""
Score merging strategies for combining two-body expert outputs.

Default (MVP): Direct addition -> s_q = s_q1 + s_q2
Upgrade (Ablation B): Learned coordination layer

The direct addition strategy is the MVP approach. Each two-body expert
independently predicts how PROTAC should move relative to its protein.
Adding them gives the combined effect.

A learned coordination layer (e.g., attention-based weighting) is reserved
as an ablation upgrade to measure whether learned merging helps over addition.
"""

import torch
import torch.nn as nn


def merge_direct_addition(scores_1: dict, scores_2: dict) -> dict:
    """
    Direct addition of scores from two subgraphs (MVP strategy).

    s_q = s_q1 + s_q2

    Rationale: Each two-body expert scores how PROTAC should move relative
    to its protein. The combined effect is the sum of both independent
    contributions.

    Args:
        scores_1: {'tr_score': [B,3], 'rot_score': [B,3], 'tor_score': [B,K]}
        scores_2: same format

    Returns:
        merged scores dict
    """
    merged = {}
    for key in scores_1:
        if scores_1[key] is not None and scores_2[key] is not None:
            merged[key] = scores_1[key] + scores_2[key]
        elif scores_1[key] is not None:
            merged[key] = scores_1[key]
        else:
            merged[key] = scores_2[key]
    return merged


class LearnedCoordinationLayer(nn.Module):
    """
    Learned score coordination layer (Ablation B upgrade).

    Instead of direct addition, uses attention-based weighting:
      s_q = α * s_q1 + (1-α) * s_q2

    where α is predicted from the TriBrain conditioning vector.

    This tests whether the model learns to weight the two subgraphs
    differently depending on context (e.g., linker length, pocket depth).
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),  # one weight per score type (tr, rot, tor)
            nn.Sigmoid(),
        )

    def forward(
        self,
        scores_1: dict,
        scores_2: dict,
        h_cond: torch.Tensor,
    ) -> dict:
        """
        Args:
            scores_1, scores_2: subgraph scores
            h_cond: [B, hidden_dim] TriBrain conditioning vector
        Returns:
            merged scores
        """
        weights = self.weight_net(h_cond)  # [B, 3]
        w_tr, w_rot, w_tor = weights[:, 0], weights[:, 1], weights[:, 2]

        merged = {
            'tr_score': (
                w_tr.unsqueeze(-1) * scores_1['tr_score']
                + (1 - w_tr.unsqueeze(-1)) * scores_2['tr_score']
            ),
            'rot_score': (
                w_rot.unsqueeze(-1) * scores_1['rot_score']
                + (1 - w_rot.unsqueeze(-1)) * scores_2['rot_score']
            ),
            'tor_score': (
                w_tor.unsqueeze(-1) * scores_1['tor_score']
                + (1 - w_tor.unsqueeze(-1)) * scores_2['tor_score']
            ),
            'merge_weights': {'tr': w_tr, 'rot': w_rot, 'tor': w_tor},
        }
        return merged
