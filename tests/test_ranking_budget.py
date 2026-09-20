"""Budget retention tests use exact known outcomes, not imputed missing labels."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tricoopdiff.structural_ranker import budget_retention_loss


def test_excluded_oracle_gets_gradient_across_budget_boundary():
    scores = torch.cat((torch.tensor([-2.]), torch.linspace(2., 0., 255))).requires_grad_()
    labels = torch.cat((torch.tensor([.8]), torch.zeros(255)))
    before = budget_retention_loss(scores, labels)
    before.backward()
    assert scores.grad[0] < 0
    assert scores.grad[1:].sum() > 0
    assert torch.isfinite(scores.grad).all()
    optimizer = torch.optim.SGD([scores], lr=1.)
    for _ in range(60):
        optimizer.zero_grad()
        budget_retention_loss(scores, labels).backward()
        optimizer.step()
    assert 0 in scores.topk(64).indices.tolist(), 'known oracle must enter the selected budget'
    assert budget_retention_loss(scores, labels) < before


def test_budget_is_not_top1_and_tied_oracles_need_only_one_representative():
    labels = torch.tensor([.8, .8, .2, .1, .0])
    scores = torch.tensor([-10., 2., 4., 1., 0.])
    assert budget_retention_loss(scores, labels, keep=2) < budget_retention_loss(scores, labels, keep=1)
    altered = scores.clone(); altered[0] = -100.
    assert torch.equal(budget_retention_loss(scores, labels, keep=2),
                       budget_retention_loss(altered, labels, keep=2))
    permutation = torch.tensor([4, 2, 0, 3, 1])
    assert torch.equal(budget_retention_loss(scores, labels, keep=2),
                       budget_retention_loss(scores[permutation], labels[permutation], keep=2))


def test_guaranteed_retention_and_score_shift():
    scores = torch.tensor([1., 0., -1.], requires_grad=True)
    labels = torch.tensor([.5, .5, .1])
    loss = budget_retention_loss(scores, labels, keep=2)
    loss.backward()
    assert loss == 0 and torch.equal(scores.grad, torch.zeros_like(scores))
    assert torch.allclose(budget_retention_loss(scores, labels, keep=1),
                          budget_retention_loss(scores + 100., labels, keep=1))


if __name__ == '__main__':
    test_excluded_oracle_gets_gradient_across_budget_boundary()
    test_budget_is_not_top1_and_tied_oracles_need_only_one_representative()
    test_guaranteed_retention_and_score_shift()
    print('Budget retention: 3 tests passed')
