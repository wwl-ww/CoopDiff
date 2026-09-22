"""Separately supervised pose ranking over invariant consensus features."""

import torch
from torch import nn


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


def state_ranking_only_loss(output, dockq, temperature=0.05):
    """Top-focused within-case objective without sparse absolute-class heads."""
    if dockq.ndim != 1 or dockq.shape != output['score'].shape or dockq.numel() < 2:
        raise ValueError('ranking-only supervision requires matching vectors with N >= 2')
    if (not torch.isfinite(dockq).all()
            or not bool(((dockq >= 0) & (dockq <= 1)).all())):
        raise ValueError('DockQ labels must be finite in [0,1]')
    return listwise_ranking_loss(output['score'], dockq.detach(), temperature)


def listwise_ranking_loss(scores, dockq, temperature=0.05):
    """Cross entropy against a soft distribution over within-case DockQ."""
    if scores.ndim != 1 or dockq.shape != scores.shape or scores.numel() < 2:
        raise ValueError("listwise ranking requires matching vectors with N >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target = torch.softmax(dockq / temperature, dim=0)
    return -(target * torch.log_softmax(scores, dim=0)).sum()

