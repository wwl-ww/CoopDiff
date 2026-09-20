"""Post-ranking supervision must use native-free allocated walker identities."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tricoopdiff.structural_ranker import selected_walker_mask, decision_ranking_loss
from scripts.train_structural_ranker import selection_score


def test_allocation_uses_only_initial_scores_and_canonical_ties():
    identities = [(i, stage) for i in range(3) for stage in ('initial', 'final')]
    mask = selected_walker_mask(torch.tensor([1., 1., 0.]),
                                [(2, 'initial'), (0, 'initial'), (1, 'initial')], identities, keep=1)
    assert mask.tolist() == [True, True, False, False, False, False]


def test_unselected_final_does_not_compete_but_initial_still_trains():
    identities = [(i, stage) for i in range(3) for stage in ('initial', 'final')]
    scores = torch.zeros(6, requires_grad=True)
    labels = torch.tensor([.1, .2, .4, .3, .8, .99])
    mask = selected_walker_mask(torch.tensor([3., 2., 1.]),
                                [(i, 'initial') for i in range(3)], identities, keep=2)
    objective = lambda y: decision_ranking_loss({'score': scores}, y, [s for _, s in identities],
                                               union_eligible=mask, final_eligible=mask, top_weight=1.)
    loss = objective(labels)
    changed = labels.clone(); changed[5] = 0.
    assert torch.equal(loss, objective(changed)), 'unselected final changed training decision'
    loss.backward()
    assert scores.grad[5] == 0
    assert scores.grad[4] != 0, 'full-pool Initial supervision was lost'


def test_quality_checkpoint_matches_selected_output_not_fullpool_oracle():
    metrics = {'cases': 1, 'union': {'mean_top1_available': .9},
               'future_refinement_selection': {'complete_pair_coverage_cases': 1,
                   'budgets': {'64': {'mean_union_top1': .3, 'mean_union_selected_best': .7}}}}
    assert selection_score(metrics, 'archived-pairs', quality_walker_scope='future-top64') == .3
    assert selection_score(metrics, 'archived-pairs', 'union', 'future-top64') == .7


if __name__ == '__main__':
    test_allocation_uses_only_initial_scores_and_canonical_ties()
    test_unselected_final_does_not_compete_but_initial_still_trains()
    test_quality_checkpoint_matches_selected_output_not_fullpool_oracle()
    print('Selector-conditioned quality: 3 tests passed')
