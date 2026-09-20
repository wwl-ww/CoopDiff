"""Whole-list supervision, flags, future isolation and complete ranking."""

from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]
from test_structural_ranker import fixture
from tricoopdiff.structural_ranker import (
    StructuralQualityRanker, full_list_ranking_loss, structural_ranking_loss,
    rank_states,
)


def test_all_rank_levels_contribute_and_ties_do_not_invent_order():
    labels = torch.tensor([.8, .5, .2, .01])
    good = torch.tensor([3., 2., 1., 0.])
    bad = good.flip(0)
    assert full_list_ranking_loss(good, labels) < full_list_ranking_loss(bad, labels)
    # Swapping only the bottom two hurts despite an unchanged top-1.
    assert full_list_ranking_loss(good, labels) < full_list_ranking_loss(
        torch.tensor([3., 2., 0., 1.]), labels)
    scores = torch.tensor([.4, -.2, .7], requires_grad=True)
    ties = torch.tensor([.5, .5, .1])
    loss = full_list_ranking_loss(scores, ties)
    assert torch.allclose(loss, full_list_ranking_loss(scores[[1, 0, 2]], ties))
    same = full_list_ranking_loss(scores, torch.ones(3) * .1)
    same.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_list_forward_ranking_gradients_reach_encoder_and_flags():
    graph = fixture()
    graph.pop('stage'); graph.pop('source')
    model = StructuralQualityRanker(16, 16)
    other = {**graph, 'node_coords': graph['node_coords'].clone()}
    other['node_coords'][5:11] += 3
    output = model.forward_list([graph, other, graph],
                                stages=['initial', 'final', 'initial'],
                                sources=['sdpi', 'sdpi', 'bitseed'])
    structural_ranking_loss(output, torch.tensor([.7, .2, .4])).backward()
    assert model.rank_head.weight.grad.abs().sum() > 0
    assert model.encoder.pair_net[1].weight.grad.abs().sum() > 0
    assert model.stage_embedding.weight.grad.abs().sum() > 0
    assert model.source_embedding.weight.grad.abs().sum() > 0
    assert model.future_head.weight.grad is None


def test_final_union_share_scores_and_full_list_returns_all_valid_states():
    scores = torch.tensor([.4, .8, .6, float('nan')])
    stages = ['initial', 'final', 'initial', 'final']
    valid = torch.tensor([True, True, True, False])
    assert rank_states(scores, stages, valid).tolist() == [1, 2, 0]
    assert rank_states(scores, stages, valid, top_k=2).tolist() == [1, 2]
    assert rank_states(scores, stages, valid, scope='final').tolist() == [1]
    assert rank_states(scores, stages, valid, scope='initial').tolist() == [2, 0]
    assert rank_states(scores, stages, torch.zeros_like(valid)).numel() == 0


def test_future_has_own_head_and_rejects_final_input():
    graph = fixture()
    model = StructuralQualityRanker(16, 16)
    out = model(**graph, task='future')
    assert set(out) == {'score'}
    try:
        model(**{**graph, 'stage': 'final'}, task='future')
    except ValueError:
        pass
    else:
        raise AssertionError('future head accepted final structure')
    try:
        structural_ranking_loss(out, torch.tensor([.5]), quality_weight=1.)
    except ValueError:
        pass
    else:
        raise AssertionError('future target used as current quality supervision')


def test_candidate_permutation_preserves_list_loss_and_scores():
    graph = fixture(); graph.pop('stage'); graph.pop('source')
    other = {**graph, 'node_coords': graph['node_coords'].clone()}
    other['node_coords'][11:] += 4
    model = StructuralQualityRanker(16, 16).eval()
    a = model.forward_list([graph, other], stages=['initial', 'final'],
                           sources=['sdpi', 'bitseed'])
    b = model.forward_list([other, graph], stages=['final', 'initial'],
                           sources=['bitseed', 'sdpi'])
    assert torch.allclose(a['score'], b['score'].flip(0))
    assert torch.allclose(structural_ranking_loss(a, torch.tensor([.6, .1])),
                          structural_ranking_loss(b, torch.tensor([.1, .6])))


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
    print('Full ranking: 5 contract tests passed')
