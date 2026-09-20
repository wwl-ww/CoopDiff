"""Decision-aligned supervision: eligible sets, gradients and future semantics."""

from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch

from tricoopdiff.structural_ranker import decision_ranking_loss, structural_ranking_loss
from tricoopdiff.structural_ranking_data import future_utility_case
from scripts.train_structural_ranker import selection_score
from test_structural_future import example


def test_loss_and_gradient_match_actual_three_decisions():
    scores = torch.tensor([3., -1., 2., 0., 8.], requires_grad=True)
    labels = torch.tensor([.2, .4, .1, .3, .95])
    stages = ['initial', 'final', 'initial', 'final', 'initial']
    eligible = torch.tensor([True, True, True, True, False])
    actual = decision_ranking_loss({'score': scores}, labels, stages, eligible, top_weight=1.)
    expected = torch.stack([structural_ranking_loss({'score': scores[ids]}, labels[ids], top_weight=1.)
                            for ids in ([0, 2, 4], [1, 3], [0, 1, 2, 3])]).mean()
    assert torch.allclose(actual, expected)
    assert torch.allclose(torch.autograd.grad(actual, scores)[0],
                          torch.autograd.grad(expected, scores)[0])
    # The unselected best initial competes in Initial, not in Final or Union.
    legacy = structural_ranking_loss({'score': scores}, labels, top_weight=1.)
    assert not torch.allclose(actual, legacy)


def test_initial_only_is_not_counted_twice():
    scores, labels = torch.tensor([0., 1.]), torch.tensor([.5, .1])
    assert torch.equal(decision_ranking_loss({'score': scores}, labels, ['initial'] * 2),
                       structural_ranking_loss({'score': scores}, labels))


def test_union_future_preserves_good_initial_and_never_uses_final_graph():
    case = example()
    future = future_utility_case(case, target='union')
    assert torch.allclose(future['labels'], torch.tensor([.8, .9]))
    assert future['graphs'][0] is case['graphs'][0]
    assert future['identities'] == [(0, 'initial'), (1, 'initial')]
    assert torch.allclose(future_utility_case(case)['labels'], torch.tensor([.01, .9]))


def test_selector_checkpoint_uses_its_own_target_and_requires_full_coverage():
    metrics = {'cases': 2, 'union': {'mean_top1_available': .1},
               'future_refinement_selection': {'complete_pair_coverage_cases': 2,
                   'budgets': {'64': {'mean_final_selected_best': .3, 'mean_union_selected_best': .5}}}}
    assert selection_score(metrics, 'joint') == .1
    assert selection_score(metrics, 'joint', 'final') == .3
    assert selection_score(metrics, 'joint', 'union') == .5
    metrics['future_refinement_selection']['complete_pair_coverage_cases'] = 1
    try:
        selection_score(metrics, 'joint', 'union')
    except ValueError:
        return
    raise AssertionError('partial outcome coverage cannot select a full-pool selector checkpoint')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
    print('Decision ranking: 4 tests passed')
