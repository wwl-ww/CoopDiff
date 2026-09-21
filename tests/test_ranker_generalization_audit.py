"""Native-free audit transforms, within-source accounting and ridge baseline."""

from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'scripts')]
import torch
from audit_ranker_generalization import fit_ridge, geometry_features, summarize, transform_graph, local_quality_audit, structural_embeddings
from tricoopdiff.structural_ranker import StructuralQualityRanker
from test_structural_ranker import fixture


def test_geometry_transform_isolated_and_global_invariant():
    graph = fixture()
    original = graph['node_coords'].clone()
    moved = transform_graph(graph, 'global_rigid')
    assert torch.allclose(geometry_features(graph), geometry_features(moved), atol=1e-5)
    separate = transform_graph(graph, 'separate_poi')
    mask = graph['node_types'] == 1
    assert torch.equal(separate['node_coords'][~mask], original[~mask])
    assert torch.equal(graph['node_coords'], original), 'audit must not mutate archived geometry'


def test_source_support_and_selected_union():
    case = {'labels': torch.tensor([.1, .4, .3, .2]),
            'stages': ['initial', 'final', 'initial', 'final'],
            'sources': ['sdpi', 'sdpi', 'bitseed', 'bitseed'],
            'identities': [(0, 'initial'), (0, 'final'), (1, 'initial'), (1, 'final')]}
    report = summarize(case, torch.tensor([0., 1., 3., 2.]))
    assert report['union']['top1_source'] == 'bitseed'
    assert report['union']['sources']['sdpi']['hits23_states'] == 1
    assert abs(report['initial']['selected64_union_oracle'] - .4) < 1e-6


def test_ridge_finite_without_validation_fit():
    cases = [{'labels': torch.tensor([.1, .3])}, {'labels': torch.tensor([.2, .4])}]
    features = [torch.tensor([[0., 1.], [2., 1.]], dtype=torch.float64),
                torch.tensor([[1., 1.], [3., 1.]], dtype=torch.float64)]
    mean, scale, beta = fit_ridge(cases, features)
    assert torch.isfinite(beta).all() and beta[0] > 0
    assert torch.equal(mean, torch.tensor([1.5, 1.], dtype=torch.float64))
    assert bool((scale > 0).all())


def test_nearest_pairs_ignore_labels_and_separate_flags():
    graph = fixture()
    graphs = []
    for offset in (0., 1., 10.):
        coords = graph['node_coords'].clone()
        coords[graph['node_types'] != 0, 0] += offset
        graphs.append({**graph, 'node_coords': coords})
    case = {'graphs': graphs, 'labels': torch.tensor([.1, .3, .7]),
            'sources': ['sdpi'] * 3, 'stages': ['initial'] * 3,
            'identities': [(i, 'initial') for i in range(3)]}
    result = local_quality_audit(case, torch.tensor([0., 1., 2.]))
    assert result[0]['positive_negative_accuracy'] == 1.
    assert torch.allclose(torch.tensor([p['distance_angstrom'] for p in result[0]['nearest_pairs']]),
                          torch.tensor([1., 9.]), atol=1e-5)
    case['labels'] = torch.tensor([.9, .1, .2])
    other = local_quality_audit(case, torch.tensor([0., 1., 2.]))
    assert [(p['a'], p['b']) for p in result[0]['nearest_pairs']] == [(p['a'], p['b']) for p in other[0]['nearest_pairs']]
    case['sources'] = ['sdpi', 'bitseed', 'sdpi']
    assert len(local_quality_audit(case, torch.tensor([0., 1., 2.]))) == 1


def test_frozen_probe_excludes_stage_source_and_removes_hook():
    model = StructuralQualityRanker(16, 16).eval()
    graph = fixture(); graph.pop('stage'); graph.pop('source')
    case = {'graphs': [graph, graph], 'stages': ['initial', 'final'], 'sources': ['sdpi', 'bitseed']}
    features = structural_embeddings(model, case)
    assert features.shape == (2, 64)
    assert torch.equal(features[0], features[1]), 'flags leaked into structure-only probe'
    assert not features.requires_grad
    assert not model.readout._forward_pre_hooks
