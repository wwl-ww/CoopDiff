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


class StateQualityRanker(nn.Module):
    """Independent current-pose quality head; never a generator or value head."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.config = {'input_dim': input_dim, 'hidden_dim': hidden_dim}
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.config['input_dim']:
            raise ValueError('state features must have shape [N,input_dim]')
        # Archives are fixed inputs: no gradient into a feature extractor/generator.
        logits = self.head(self.encoder(features.detach()))
        probabilities = torch.softmax(logits[:, 1:], dim=-1)
        return {'score': logits[:, 0], 'q_hat': torch.sigmoid(logits[:, 0]),
                'bin_logits': logits[:, 1:], 'p23': probabilities[:, 1:].sum(-1),
                'p49': probabilities[:, 2]}


def state_quality_loss(output, dockq):
    """Within-case ordering plus absolute quality and ordered-threshold supervision."""
    if dockq.ndim != 1 or dockq.shape != output['score'].shape or not dockq.numel():
        raise ValueError('quality supervision requires matching non-empty vectors')
    if not torch.isfinite(dockq).all() or not ((dockq >= 0) & (dockq <= 1)).all():
        raise ValueError('DockQ labels must be finite in [0,1]')
    labels = dockq.detach()
    bins = (labels >= .23).long() + (labels >= .49).long()
    # A singleton still provides absolute quality supervision, but no ordering.
    ranking = (output['score'].sum() * 0 if labels.numel() == 1
               else listwise_ranking_loss(output['score'], labels))
    regression = nn.functional.smooth_l1_loss(output['q_hat'], labels)
    classification = nn.functional.cross_entropy(output['bin_logits'], bins)
    return ranking + .5 * regression + .5 * classification


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
