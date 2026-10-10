"""Absolute quality, ordered thresholds and frozen-generator boundaries."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.ranking.ranking_inference import (StateQualityRanker, state_quality_loss,
                                      state_ranking_only_loss)


def test_quality_outputs_are_bounded_and_thresholds_ordered():
    model = StateQualityRanker(12, 16)
    output = model(torch.randn(8, 12))
    assert ((output['q_hat'] > 0) & (output['q_hat'] < 1)).all()
    assert (output['p49'] <= output['p23']).all()
    assert ((output['p23'] >= 0) & (output['p23'] <= 1)).all()


def test_quality_loss_updates_only_ranker_and_rejects_invalid_truth():
    torch.manual_seed(7)
    features = torch.randn(8, 12, requires_grad=True)
    model = StateQualityRanker(12, 16)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    labels = torch.linspace(0., 1., 8)
    loss = state_quality_loss(model(features), labels)
    loss.backward()
    assert features.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.head.weight)
    try:
        state_quality_loss(model(features), labels * 2)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid quality labels must be rejected')


def test_ranking_only_loss_is_temperature_sensitive_and_ignores_class_heads():
    model = StateQualityRanker(3, 4)
    output = model(torch.randn(5, 3))
    labels = torch.tensor([0., .01, .02, .1, .3])
    broad = state_ranking_only_loss(output, labels, .05)
    sharp = state_ranking_only_loss(output, labels, .01)
    assert torch.isfinite(broad) and torch.isfinite(sharp) and broad != sharp
    broad.backward()
    assert model.head.weight.grad[0].abs().sum() > 0
    assert model.head.weight.grad[1:].abs().sum() == 0
