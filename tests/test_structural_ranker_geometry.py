"""Controlled geometry/condition diagnostics, not held-out efficacy tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from test_structural_ranker import fixture
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss
from scripts.train_structural_ranker import (
    geometry_shuffle_diagnostic, ordering_metrics, permute_within_groups,
)


def geometry_list():
    graph = fixture()
    graph.pop('stage'); graph.pop('source')
    graphs = []
    for offset in (0., 2., 5., 10.):
        coords = graph['node_coords'].clone()
        coords[5:11, 0] += offset
        graphs.append({**graph, 'node_coords': coords})
    return graphs


def test_geometry_and_conditions_each_reach_score():
    graphs = geometry_list()
    model = StructuralQualityRanker(16, 16).eval()
    with torch.no_grad():
        geometry = model.forward_list(graphs, stages=['initial'] * 4,
                                      sources=['sdpi'] * 4)['score']
        conditions = model.forward_list([graphs[0]] * 4,
                                        stages=['initial', 'initial', 'final', 'final'],
                                        sources=['sdpi', 'bitseed', 'sdpi', 'bitseed'])['score']
    assert geometry.std() > 1e-5, 'geometry cannot affect score under fixed flags'
    assert conditions.std() > 1e-5, 'explicit conditions cannot affect score'


def test_fixed_flags_can_learn_geometry_order():
    graphs = geometry_list()
    model = StructuralQualityRanker(16, 16)
    model.stage_embedding.requires_grad_(False)
    model.source_embedding.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    # Arbitrary non-monotone targets test capacity, not a physical-quality law.
    targets = torch.tensor([.2, .8, .05, .5])
    before = None
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_list(graphs, stages=['initial'] * 4, sources=['sdpi'] * 4)
        loss = structural_ranking_loss(output, targets)
        if before is None:
            before = float(loss.detach())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model.forward_list(graphs, stages=['initial'] * 4, sources=['sdpi'] * 4)
        after = float(structural_ranking_loss(output, targets))
    assert after < before * .25, (before, after)
    assert ordering_metrics(output['score'], targets)['pair_accuracy'] == 1.
    assert model.stage_embedding.weight.grad is None
    assert model.source_embedding.weight.grad is None


def test_flag_only_shortcut_cannot_pass_within_group_metric():
    # Perfect group separation can look good globally without ordering geometry.
    scores = torch.tensor([1., 1., 0., 0.])
    targets = torch.tensor([.9, .8, .2, .1])
    global_metric = ordering_metrics(scores, targets)
    within = ordering_metrics(scores, targets, groups=[('final', 'bitseed')] * 2 + [('initial', 'sdpi')] * 2)
    assert global_metric['pair_accuracy'] > .8
    assert within == {'comparable_pairs': 2, 'pair_accuracy': .5}
    singleton = ordering_metrics(scores[:2], targets[:2], groups=['a', 'b'])
    assert singleton['pair_accuracy'] is None


def test_group_shuffle_preserves_flags_and_global_rng():
    groups = [('initial', 'sdpi'), ('final', 'sdpi'), ('initial', 'sdpi'),
              ('initial', 'bitseed'), ('final', 'sdpi')]
    before = torch.random.get_rng_state().clone()
    a = permute_within_groups(groups, torch.Generator().manual_seed(123))
    b = permute_within_groups(groups, torch.Generator().manual_seed(123))
    assert torch.equal(a, b)
    assert sorted(a.tolist()) == list(range(len(groups)))
    assert all(groups[i] == groups[j] for i, j in enumerate(a.tolist()))
    assert a[3] == 3, 'singleton stratum must not borrow another source'
    assert torch.equal(before, torch.random.get_rng_state())


def test_score_permutation_equals_actual_structure_permutation():
    graphs = geometry_list()
    model = StructuralQualityRanker(16, 16).eval()
    stages, sources = ['initial'] * 4, ['sdpi', 'bitseed', 'sdpi', 'bitseed']
    permutation = [2, 3, 0, 1]
    with torch.no_grad():
        original = model.forward_list(graphs, stages=stages, sources=sources)['score']
        moved = model.forward_list([graphs[i] for i in permutation], stages=stages, sources=sources)['score']
    assert torch.allclose(moved, original[permutation], atol=1e-6)


def test_shuffle_exposes_structure_association_but_not_flag_only_scores():
    stages, sources = ['initial'] * 4, ['sdpi'] * 4
    labels = torch.tensor([.9, .6, .3, .0])
    report = geometry_shuffle_diagnostic(labels, labels, stages, sources, repeats=128, seed=18)
    assert report['union']['observed_minus_shuffled'] > .2
    assert report['within_stage_source']['observed_minus_shuffled'] > .3
    assert report['final'] is None
    flags_only = geometry_shuffle_diagnostic(torch.ones(4), labels, stages, sources, repeats=16, seed=18)
    assert flags_only['union']['observed_minus_shuffled'] == 0
    assert flags_only['within_stage_source']['observed_minus_shuffled'] == 0
    assert flags_only['movable_states'] == 4


if __name__ == '__main__':
    test_geometry_and_conditions_each_reach_score()
    test_fixed_flags_can_learn_geometry_order()
    test_flag_only_shortcut_cannot_pass_within_group_metric()
    test_group_shuffle_preserves_flags_and_global_rng()
    test_score_permutation_equals_actual_structure_permutation()
    test_shuffle_exposes_structure_association_but_not_flag_only_scores()
    print('Structural geometry: 6 controlled tests passed')
