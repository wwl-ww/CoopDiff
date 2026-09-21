"""Architecture contracts for set-aware candidate ranking."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tricoopdiff.setwise_ranking import (SetwiseCandidateRanker, TrajectorySetRanker,
                                         setwise_quality_loss,
                                         source_indices, stable_topk,
                                         trajectory_ranking_loss)


def _model():
    torch.manual_seed(7)
    return SetwiseCandidateRanker(
        input_dim=5, model_dim=16, num_heads=4, num_layers=2,
        feedforward_dim=32, dropout=0.,
    )


def test_setwise_ranker_is_permutation_equivariant():
    model = _model().eval()
    features = torch.randn(6, 5)
    sources = source_indices(['bitseed', 'sdpi', 'bitseed', 'bitseed', 'sdpi', 'sdpi'])
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.argsort(permutation)
    first = model(features, sources)
    second = model(features[permutation], sources[permutation])
    for key in ('score', 'q_hat', 'p23', 'p49'):
        assert torch.allclose(first[key], second[key][inverse], atol=1e-6)


def test_setwise_ranker_masks_padding_and_orders_thresholds():
    model = _model().eval()
    features = torch.randn(2, 4, 5)
    sources = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    output = model(features, sources, mask)
    assert output['score'].shape == (2, 4)
    assert torch.equal(output['mask'], mask)
    assert bool(((output['p49'] >= 0) & (output['p49'] <= output['p23'])).all())
    assert bool((output['p23'] <= 1).all())

    # Masked values cannot affect valid outputs.
    changed = features.clone(); changed[1, 2:] = 100000
    other = model(changed, sources, mask)
    assert torch.allclose(output['score'][1, :2], other['score'][1, :2], atol=1e-6)


def test_setwise_loss_updates_context_encoder_and_ignores_padding():
    model = _model().train()
    features = torch.randn(2, 4, 5)
    sources = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    labels = torch.tensor([[.1, .7, .3, .2], [.8, .1, 999., 999.]])
    output = model(features, sources, mask)
    # Invalid labels are outside [0,1], so sanitize only padding before loss;
    # the loss contract still validates every supplied tensor value.
    labels[~mask] = 0
    loss = setwise_quality_loss(output, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.set_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_sparse_supervision_keeps_unlabelled_candidates_as_context():
    model = _model().train()
    features = torch.randn(6, 5, requires_grad=True)
    sources = source_indices(['bitseed', 'sdpi'] * 3)
    supervision = torch.tensor([True, False, True, False, False, True])
    labels = torch.tensor([.1, 0., .7, 0., 0., .2])
    output = model(features, sources)
    loss = setwise_quality_loss(output, labels, supervision_mask=supervision)
    gradient, = torch.autograd.grad(loss, features, allow_unused=True)
    # Archived features are detached by contract; context is learned through
    # model parameters, while unlabelled tokens still participate in attention.
    assert gradient is None
    loss.backward()
    assert model.set_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_sources_and_masked_topk_have_explicit_contracts():
    assert source_indices(['bitseed', 'sdpi']).tolist() == [0, 1]
    selected = stable_topk(torch.tensor([.2, .9, .9, 100.]),
                           torch.tensor([True, True, True, False]), 2)
    assert selected.tolist() == [1, 2]
    try:
        source_indices(['unknown'])
    except ValueError:
        pass
    else:
        raise AssertionError('unknown candidate source was accepted')


def test_trajectory_ranker_pairs_states_and_ignores_failed_placeholders():
    torch.manual_seed(11)
    model = TrajectorySetRanker(5, 16, 4, 2, 32, 0.).eval()
    initial, final = torch.randn(6, 5), torch.randn(6, 5)
    sources = source_indices(['bitseed', 'sdpi'] * 3)
    valid = torch.tensor([True, False, True, True, False, True])
    first = model(initial, final, sources, valid)
    changed = final.clone(); changed[~valid] = 100000
    second = model(initial, changed, sources, valid)
    for key in ('final_score', 'union_score', 'choose_final_probability'):
        assert torch.allclose(first[key], second[key], atol=1e-6)
    assert bool((first['final_p49'] <= first['final_p23']).all())
    assert bool((first['union_p49'] <= first['union_p23']).all())


def test_trajectory_loss_updates_pair_and_set_encoders():
    torch.manual_seed(13)
    model = TrajectorySetRanker(5, 16, 4, 1, 32, 0.)
    initial, final = torch.randn(5, 5), torch.randn(5, 5)
    sources = source_indices(['bitseed', 'bitseed', 'sdpi', 'sdpi', 'bitseed'])
    valid = torch.tensor([True, True, False, True, True])
    initial_q = torch.tensor([.1, .4, .2, .7, .05])
    final_q = torch.tensor([.3, .2, 0., .8, .1])
    loss = trajectory_ranking_loss(
        model(initial, final, sources, valid), initial_q, final_q)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pair_encoder[1].weight.grad.abs().sum() > 0
    assert model.set_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


if __name__ == '__main__':
    test_setwise_ranker_is_permutation_equivariant()
    test_setwise_ranker_masks_padding_and_orders_thresholds()
    test_setwise_loss_updates_context_encoder_and_ignores_padding()
    test_sparse_supervision_keeps_unlabelled_candidates_as_context()
    test_sources_and_masked_topk_have_explicit_contracts()
    test_trajectory_ranker_pairs_states_and_ignores_failed_placeholders()
    test_trajectory_loss_updates_pair_and_set_encoders()
