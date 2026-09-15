"""Separately supervised pose ranking over invariant consensus features."""

import torch
from torch import nn


class PoseRanker(nn.Module):
    """Small scalar head that cannot alter the frozen pose generator."""

    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.config = {"input_dim": input_dim, "hidden_dim": hidden_dim}
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.config["input_dim"]:
            raise ValueError("ranker features must be [N,input_dim]")
        return self.net(features).squeeze(-1)


def listwise_ranking_loss(scores, dockq, temperature=0.05):
    """Cross entropy against a soft distribution over within-case DockQ."""
    if scores.ndim != 1 or dockq.shape != scores.shape or scores.numel() < 2:
        raise ValueError("listwise ranking requires matching vectors with N >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target = torch.softmax(dockq / temperature, dim=0)
    return -(target * torch.log_softmax(scores, dim=0)).sum()


def selected_pose_metrics(scores, dockq, threshold=0.23):
    """Return deployable top-1 quality and oracle retention for one case."""
    if scores.ndim != 1 or dockq.shape != scores.shape or not scores.numel():
        raise ValueError("ranking metrics require matching non-empty vectors")
    selected = int(torch.argmax(scores))
    oracle = int(torch.argmax(dockq))
    selected_dockq = float(dockq[selected])
    oracle_dockq = float(dockq[oracle])
    return {
        "selected_index": selected,
        "oracle_index": oracle,
        "selected_dockq": selected_dockq,
        "oracle_dockq": oracle_dockq,
        "oracle_retention": selected_dockq / max(oracle_dockq, 1e-8),
        "selected_hit": selected_dockq >= threshold,
        "oracle_hit": oracle_dockq >= threshold,
    }
