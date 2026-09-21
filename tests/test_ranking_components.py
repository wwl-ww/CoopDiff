"""Geometry supervision stays outside inputs and preserves old checkpoints."""
import copy
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from test_structural_ranker import fixture
from test_structural_ranking_data import archive_fixture, rejected
from tricoopdiff.structural_ranker import StructuralQualityRanker, component_quality_loss
from tricoopdiff.structural_ranking_data import dockq_components, adapt_case
from tricoopdiff.ranking_backward import backward_list


def test_decomposition_and_invalid_labels():
    assert dockq_components({'dockq': .2}) is None
    label = {'dockq': 1./3, 'fnat': .3, 'irms': 3., 'lrms': 8.5}
    assert torch.allclose(torch.tensor(dockq_components(label)), torch.tensor([.3, .2, .5]))
    for change in ({'irms': -1}, {'fnat': 2}, {'lrms': float('nan')}, {'dockq': .9}):
        rejected(lambda: dockq_components({**label, **change}))
    rejected(lambda: dockq_components({'dockq': .2, 'fnat': .2}))


def test_old_parameters_and_checkpoint_shape_unchanged_by_optional_head():
    torch.manual_seed(17); old = StructuralQualityRanker(16, 16)
    torch.manual_seed(17); new = StructuralQualityRanker(16, 16, component_aux=True)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    assert new(**fixture())['components'].shape == (1, 3)
    assert 'components' not in old(**fixture())


def test_component_gradient_replay_and_label_isolation():
    graph = fixture(); graph.pop('stage'); graph.pop('source')
    model = StructuralQualityRanker(16, 16, component_aux=True)
    replay = copy.deepcopy(model)
    case = {'graphs': [graph] * 3, 'stages': ['initial', 'final', 'final'], 'sources': ['sdpi'] * 3}
    target = torch.tensor([[.1, .2, .3], [.8, .7, .6], [.4, .5, .6]], requires_grad=True)
    objective = lambda out: component_quality_loss(out, target)
    loss = objective(model.forward_list(case['graphs'], stages=case['stages'], sources=case['sources']))
    loss.backward()
    backward_list(replay, case, 'cpu', objective, chunk_size=2)
    assert target.grad is None
    for (name, a), (_, b) in zip(model.named_parameters(), replay.named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=2e-4), name
    assert model.component_head.weight.grad.abs().sum() > 0
    assert model.encoder.protein_feature_proj.weight.grad.abs().sum() > 0
    archive, labels = archive_fixture()
    plain = adapt_case(archive, labels)
    assert not plain['component_mask'].any()
    labels['labels'][0]['initial'] = {'dockq': 1./3, 'fnat': .3, 'irms': 3., 'lrms': 8.5}
    enriched = adapt_case(archive, labels)
    assert enriched['component_mask'].tolist() == [True, False]
    for a, b in zip(plain['graphs'], enriched['graphs']):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a), 'labels leaked into graph'


if __name__ == '__main__':
    test_decomposition_and_invalid_labels()
    test_old_parameters_and_checkpoint_shape_unchanged_by_optional_head()
    test_component_gradient_replay_and_label_isolation()
    print('Ranking components: 3 tests passed')
