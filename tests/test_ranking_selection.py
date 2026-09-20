"""Selection must be tested against refined outcomes, not initial quality alone."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tricoopdiff.ranking_selection import refinement_selection_metrics


def test_initial_quality_is_not_refined_utility():
    scores = torch.tensor([9., 1., 8., 10.])
    labels = torch.tensor([.8, .01, .1, .9])
    identities = [(0, 'initial'), (0, 'final'), (1, 'initial'), (1, 'final')]
    result = refinement_selection_metrics(scores, labels, identities, ['sdpi'] * 4, (1, 2))
    one, two = result['budgets']['1'], result['budgets']['2']
    assert result['complete_pair_coverage']
    assert one['initial_selected_best'] > .79 and one['final_selected_best'] < .02
    assert one['final_top1'] < .02 and one['union_top1'] > .79
    assert two['final_top1'] > .89 and two['union_top1'] > .89
    # The selected IDs must not change when labels change.
    changed = refinement_selection_metrics(scores, 1-labels, identities, ['sdpi'] * 4, (1,))
    assert changed['budgets']['1']['selected_pool_ids'] == one['selected_pool_ids']


def test_identity_join_permutation_and_incomplete_coverage():
    scores = torch.tensor([4., 3., 2., 1., 9.])
    labels = torch.tensor([.1, .2, .3, .4, .8])
    ids = [(0,'initial'), (1,'final'), (1,'initial'), (0,'final'), (2,'initial')]
    sources = ['sdpi', 'bitseed', 'bitseed', 'sdpi', 'sdpi']
    result = refinement_selection_metrics(scores, labels, ids, sources)
    assert result['paired_states'] == 2 and result['unpaired_initials'] == 1
    assert not result['complete_pair_coverage']
    assert result['budgets']['8']['selected_pool_ids'] == [0, 1]
    permutation = [4, 2, 0, 3, 1]
    moved = refinement_selection_metrics(scores[permutation], labels[permutation],
        [ids[i] for i in permutation], [sources[i] for i in permutation])
    assert result == moved


def test_unobserved_outcomes_are_not_fabricated():
    result = refinement_selection_metrics(torch.tensor([1.]), torch.tensor([.5]),
                                          [(0,'initial')], ['sdpi'])
    assert result['budgets'] == {} and result['unpaired_initials'] == 1
    try:
        refinement_selection_metrics(torch.tensor([1.]), torch.tensor([.5]), [(0,'final')], ['sdpi'])
    except ValueError:
        pass
    else:
        raise AssertionError('unpaired final accepted')


def test_uniform_control_matches_exhaustive_subsets():
    import itertools
    from tricoopdiff.ranking_selection import expected_random_winner
    values = torch.tensor([.2, .8, .1, .9, .3], dtype=torch.float64)
    priority = torch.tensor([1., 3., 3., -1., 4.], dtype=torch.float64)
    for keep in (1, 2, 4, 5):
        outcomes = []
        for subset in itertools.combinations(range(len(values)), keep):
            winner = max(subset, key=lambda i: (float(priority[i]), -i))
            outcomes.append(float(values[winner]))
        assert abs(expected_random_winner(values, priority, keep) - sum(outcomes)/len(outcomes)) < 1e-12


def test_terminal_ties_independent_of_initial_ranking():
    labels = torch.tensor([.1,.9,.2,.3])
    identities = [(0,'initial'),(0,'final'),(1,'initial'),(1,'final')]
    sources = ['sdpi']*4
    quality = torch.tensor([0.,2.,1.,2.])
    a = refinement_selection_metrics(quality,labels,identities,sources,(2,))
    b = refinement_selection_metrics(quality,labels,identities,sources,(2,),
                                    selection_scores=-quality)
    assert a['budgets']['2']['final_top1'] == b['budgets']['2']['final_top1']
    assert abs(a['budgets']['2']['uniform_random_control']['final_top1'] - .9)<1e-6


if __name__ == '__main__':
    test_initial_quality_is_not_refined_utility()
    test_identity_join_permutation_and_incomplete_coverage()
    test_unobserved_outcomes_are_not_fabricated()
    test_uniform_control_matches_exhaustive_subsets()
    test_terminal_ties_independent_of_initial_ranking()
    print('Refinement selection: 5 tests passed')
