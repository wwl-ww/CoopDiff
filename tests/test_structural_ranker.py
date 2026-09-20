"""Reuse, symmetry, gradient isolation and actual-state decision contracts."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.geometry_encoders import TernaryGeometryEncoder
from tricoopdiff.structural_ranker import (
    StructuralQualityRanker, select_state, structural_quality_loss,
    full_list_ranking_loss, structural_ranking_loss, rank_states,
)


def fixture():
    torch.manual_seed(137)
    features = torch.randn(15, 16)
    features[11:] = 0
    features[11:, 0] = torch.tensor([5., 6., 7., 5.])
    return dict(stage='initial', source='sdpi',
                node_features=features, node_coords=torch.randn(15, 3),
                node_types=torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
                lig_edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]))


def test_quality_is_rigid_invariant_and_node_order_consistent():
    graph = fixture()
    model = StructuralQualityRanker(16, 16).eval()
    base = model(**graph)
    rotation, _ = torch.linalg.qr(torch.tensor([[.1, .2, .8], [.7, -.3, .2], [.2, .6, -.5]]))
    moved = model(**{**graph, 'node_coords': graph['node_coords'] @ rotation + 5})
    for key in ('score', 'p23', 'p49'):
        assert torch.allclose(base[key], moved[key], atol=2e-6), key
    # Reorder all three molecules; bond indices use ligand-local positions.
    ligand_perm = torch.tensor([2, 0, 3, 1])
    permutation = torch.cat((torch.tensor([4, 2, 0, 1, 3]),
                             torch.tensor([10, 8, 6, 9, 7, 5]), ligand_perm + 11))
    reordered = {key: graph[key][permutation] for key in
                 ('node_features', 'node_coords', 'node_types')}
    reordered['lig_edge_index'] = torch.argsort(ligand_perm)[graph['lig_edge_index']]
    reordered.update(stage=graph['stage'], source=graph['source'])
    out = model(**reordered)
    assert torch.allclose(base['score'], out['score'], atol=2e-6)


def test_quality_supervision_reaches_private_encoder_only():
    graph = fixture()
    parent = TernaryGeometryEncoder(16, 16).requires_grad_(False)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    model = StructuralQualityRanker(16, 16, encoder=parent)
    for a, b in zip(parent.parameters(), model.encoder.parameters()):
        assert a.data_ptr() != b.data_ptr(), 'generator parameter storage is shared'
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    graph['node_features'].requires_grad_(True)
    loss = structural_quality_loss(model(**graph), torch.tensor([.35]))
    loss.backward()
    assert model.encoder.pair_net[1].weight.grad.abs().sum() > 0
    assert model.encoder.ligand_contact_fusion[1].weight.grad.abs().sum() > 0
    assert graph['node_features'].grad is None, 'gradient entered feature producer'
    optimizer.step()
    assert all(torch.equal(value, parent.state_dict()[k]) for k, value in before.items())
    assert all(p.grad is None and not p.requires_grad for p in parent.parameters())


def test_distribution_and_threshold_supervision_are_consistent():
    graph = fixture()
    output = StructuralQualityRanker(16, 16)(**graph)
    assert torch.allclose(output['q_hat'],
                          (output['bin_probabilities'] * output['bin_means']).sum(-1))
    assert bool((output['p49'] <= output['p23']).all())
    for target, expected_bin in ((.1, 0), (.23, 1), (.49, 2), (1., 2)):
        logits = torch.zeros(1, 3, requires_grad=True)
        means = torch.tensor([[.1, .3, .7]], requires_grad=True)
        loss = structural_quality_loss(
            {'score': torch.zeros(1), 'bin_logits': logits, 'bin_means': means},
            torch.tensor([target]))
        loss.backward()
        assert logits.grad[0, expected_bin] < 0
        for index in set(range(3)) - {expected_bin}:
            assert means.grad[0, index] == 0, 'conditional mean received wrong-bin label'


def test_selection_returns_scored_state_and_excludes_failed_finals():
    scores = torch.tensor([.2, .8, .6, float('nan')])
    stages = ['initial', 'final', 'initial', 'final']
    valid = torch.tensor([True, True, True, False])
    assert select_state(scores, stages, valid, scope='initial') == 2
    assert select_state(scores, stages, valid, scope='final') == 1
    assert select_state(scores, stages, valid) == 1
    valid[1] = False
    assert select_state(scores, stages, valid, scope='final') is None
    assert select_state(scores, stages, valid) == 2


def test_model_rejects_quality_metadata_and_invalid_categories():
    model = StructuralQualityRanker(16, 16)
    graph = fixture()
    try:
        model(**graph, dockq=.9)
    except TypeError:
        pass
    else:
        raise AssertionError('truth entered ranking inference path')
    graph['node_features'][11, 0] = .5
    try:
        model(**graph)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid category silently clamped')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
    print('Structural ranker contract tests passed')
