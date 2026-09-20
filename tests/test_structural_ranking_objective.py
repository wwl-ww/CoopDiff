"""Analytical objective checks; no benchmark-driven hyperparameter choice."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tricoopdiff.structural_ranker import (
    best_tier_loss, full_list_ranking_loss, structural_ranking_loss,
)


def test_best_candidate_gradient_scales_with_list_length():
    for count in (8, 64, 256):
        scores = torch.zeros(count, dtype=torch.float64, requires_grad=True)
        labels = torch.linspace(1., 0., count, dtype=torch.float64)
        full = torch.autograd.grad(full_list_ranking_loss(scores, labels), scores)[0]
        top = torch.autograd.grad(best_tier_loss(scores, labels), scores)[0]
        assert torch.allclose(full[0], torch.tensor(-1. / count, dtype=torch.float64))
        assert torch.allclose(top[0], torch.tensor(-(count - 1.) / count, dtype=torch.float64))
        assert full[0] < 0, 'original loss does supervise the best candidate'


def test_best_tier_uses_probability_mass_and_not_arbitrary_tie_identity():
    scores = torch.tensor([.7, -.1, .3], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([.9, .90001, .2], dtype=torch.float64)
    expected = -torch.log(torch.softmax(scores, 0)[:2].sum())
    loss = best_tier_loss(scores, labels)
    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, best_tier_loss(scores[[1, 0, 2]], labels[[1, 0, 2]]))
    grad = torch.autograd.grad(loss, scores)[0]
    assert bool((grad[:2] < 0).all()) and grad[2] > 0


def test_all_ties_singleton_and_score_shift():
    for count in (1, 4):
        scores = torch.zeros(count, requires_grad=True)
        loss = best_tier_loss(scores, torch.ones(count) * .5)
        loss.backward()
        assert loss == 0 and torch.equal(scores.grad, torch.zeros_like(scores))
    scores = torch.tensor([-2000., 1500., 2.], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([.9, .5, .1], dtype=torch.float64)
    loss = best_tier_loss(scores, labels)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, best_tier_loss(scores + 10000, labels))
    assert torch.isfinite(torch.autograd.grad(loss, scores)[0]).all()


def test_optional_term_keeps_default_and_lower_rank_supervision():
    labels = torch.tensor([.8, .5, .2, .01])
    good, swapped = torch.tensor([3., 2., 1., 0.]), torch.tensor([3., 2., 0., 1.])
    assert torch.equal(structural_ranking_loss({'score': good}, labels), full_list_ranking_loss(good, labels))
    assert torch.allclose(best_tier_loss(good, labels), best_tier_loss(swapped, labels))
    assert structural_ranking_loss({'score': good}, labels, top_weight=1.) < structural_ranking_loss(
        {'score': swapped}, labels, top_weight=1.)
    combined = structural_ranking_loss({'score': good}, labels, top_weight=.5)
    assert torch.allclose(combined, full_list_ranking_loss(good, labels) + .5 * best_tier_loss(good, labels))


def test_invalid_top_weight_and_labels_rejected():
    scores, labels = torch.zeros(2), torch.tensor([.1, .2])
    for weight in (-1., float('nan'), float('inf')):
        try:
            structural_ranking_loss({'score': scores}, labels, top_weight=weight)
        except ValueError:
            continue
        raise AssertionError('invalid best-tier coefficient accepted')
    try:
        best_tier_loss(scores, torch.tensor([.1, float('nan')]))
    except ValueError:
        return
    raise AssertionError('nonfinite label accepted')


def test_suffix_scan_matches_explicit_tier_loss_and_gradient():
    def reference(scores, labels):
        order = torch.argsort(labels, descending=True, stable=True)
        ordered, truth = scores[order], labels[order].tolist()
        terms, begin = [], 0
        while begin < len(truth):
            end = begin + 1
            while end < len(truth) and truth[begin] - truth[end] <= 1e-4:
                end += 1
            if end < len(truth):
                terms.append(torch.nn.functional.softplus(torch.logsumexp(ordered[end:], 0) - ordered[begin:end]))
            begin = end
        return torch.cat(terms).mean() if terms else scores.sum()*0
    generator = torch.Generator().manual_seed(13)
    for count in (1, 8, 64, 320, 512):
        for scale in (1., 1000.):
            scores = (torch.randn(count, generator=generator, dtype=torch.float64)*scale).requires_grad_()
            labels = torch.randint(0, 10, (count,), generator=generator).double()/10
            actual, expected = full_list_ranking_loss(scores, labels), reference(scores, labels)
            assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
            grad = torch.autograd.grad(actual, scores)[0]
            ref = torch.autograd.grad(expected, scores)[0]
            assert torch.isfinite(grad).all()
            assert torch.allclose(grad, ref, atol=1e-10, rtol=1e-9)


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
    print('Structural objective: 6 analytical tests passed')
