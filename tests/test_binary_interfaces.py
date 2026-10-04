"""Metric and native-pose isolation checks for the independent baseline."""

import sys
from pathlib import Path

import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_binary_interfaces import fresh_ligand, interface_metrics


def test_metrics_do_not_align_away_docking_error():
    native = torch.tensor([[[0., 0, 0], [20., 0, 0]]])
    receptor = torch.tensor([[[0., 2, 0], [0., -2, 0]]])
    result = interface_metrics(native + 3, native, receptor)
    assert result['interface_atom_count'] == 1
    assert abs(result['ligand_rmsd'] - 27 ** 0.5) < 1e-5
    assert abs(result['interface_atom_rmsd'] - 27 ** 0.5) < 1e-5


def test_initialization_ignores_native_coordinates():
    mol = Chem.MolFromSmiles('CC')
    ligand = {'mol': mol, 'features': torch.zeros(1, 2, 1),
              'coords': torch.zeros(1, 2, 3),
              'edge_index': torch.tensor([[0, 1], [1, 0]]),
              'edge_mask': torch.zeros(2, dtype=torch.bool),
              'mask_rotate': torch.zeros(0, 2, dtype=torch.bool)}
    receptor = torch.tensor([[0., 2, 0], [0., -2, 0]])
    first, _ = fresh_ligand(ligand, receptor, 42)
    ligand['coords'][:] = float('nan')
    second, _ = fresh_ligand(ligand, receptor, 42)
    assert torch.equal(first, second), 'native coordinates leaked into blind initialization'


def test_receptor_crop_preserves_all_geometrically_visible_nodes_with_empty_edges():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords = torch.arange(10).float()[:, None].expand(-1, 3) / 10
    features = torch.arange(10).float()[:, None]
    result = crop_receptor_to_ligand(coords, features, torch.zeros(1, 3),
                                     torch.empty(2, 0, dtype=torch.long), cutoff=10.,
                                     require_internal_edges=False)
    assert result['mask'].all(), 'fallback must not discard nodes already inside the radius'
    assert torch.equal(result['coords'], coords)
    assert torch.equal(result['features'], features)
    assert result['edge_index'].shape == (2, 0)


def test_receptor_crop_relabels_edges_without_reordering_nodes_or_detaching_inputs():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords = torch.tensor([[5., 0., 0.], [0., 0., 0.], [10., 0., 0.],
                           [1., 0., 0.], [20., 0., 0.]], requires_grad=True)
    features = torch.arange(5).float()[:, None].requires_grad_()
    edges = torch.tensor([[1, 3, 0, 2], [3, 1, 2, 0]])
    result = crop_receptor_to_ligand(coords, features, torch.zeros(1, 3), edges,
                                     cutoff=2., min_residues_with_edges=2)
    assert result['mask'].tolist() == [False, True, False, True, False]
    assert result['features'][:, 0].tolist() == [1., 3.]
    assert torch.equal(result['edge_index'], torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(edges, torch.tensor([[1, 3, 0, 2], [3, 1, 2, 0]]))
    (result['coords'].sum() + result['features'].sum()).backward()
    assert torch.equal(coords.grad, result['mask'][:, None].expand(-1, 3).float())
    assert torch.equal(features.grad, result['mask'][:, None].float())


def test_graph_entry_rejects_empty_components_but_preserves_singleton_inputs():
    from tricoopdiff.inputs.graphs import build_binary_graph, build_trifuse_graph, crop_receptor_to_ligand

    coords, features = torch.zeros(1, 3), torch.ones(1, 2)
    empty_coords, empty_features = torch.empty(0, 3), torch.empty(0, 2)
    invalid_calls = (
        lambda: build_binary_graph(empty_coords, empty_features, coords, features),
        lambda: build_binary_graph(coords, features, empty_coords, empty_features),
        lambda: build_binary_graph(coords, empty_features, coords, features),
        lambda: build_trifuse_graph(coords, features, empty_coords, empty_features, coords, features),
        lambda: crop_receptor_to_ligand(empty_coords, empty_features, coords, cutoff=None),
        lambda: crop_receptor_to_ligand(coords, features, empty_coords, cutoff=1.),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError as error:
            assert 'coordinates' in str(error) or 'features' in str(error)
        else:
            raise AssertionError('empty/mismatched nodes must fail at the shared graph boundary')
    graph = build_binary_graph(coords, features, coords, features,
                               rec_edge_index=torch.empty(2, 0, dtype=torch.long))
    assert graph['receptor'].pos.shape == graph['ligand'].pos.shape == (1, 3)
    crop = crop_receptor_to_ligand(coords, features, coords,
                                   rec_edge_index=torch.empty(2, 0, dtype=torch.long), cutoff=0.)
    assert crop['mask'].tolist() == [True] and crop['edge_index'].shape == (2, 0)


def test_receptor_crop_rejects_invalid_radius_and_fallback_counts():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords, features = torch.zeros(2, 3), torch.ones(2, 1)
    for invalid in (-1., float('nan'), float('inf'), True, torch.tensor([1., 2.]), torch.tensor(True)):
        try:
            crop_receptor_to_ligand(coords, features, coords, cutoff=invalid)
        except ValueError as error:
            assert 'cutoff' in str(error)
        else:
            raise AssertionError('an invalid radius must not silently trigger a different crop')
    for name in ('min_residues', 'min_residues_with_edges'):
        for invalid in (0, -1, True, 1.5):
            try:
                crop_receptor_to_ligand(coords, features, coords, cutoff=1., **{name: invalid})
            except ValueError as error:
                assert 'minimum counts' in str(error)
            else:
                raise AssertionError('invalid fallback counts must not silently become another count')
    assert torch.equal(coords, torch.zeros(2, 3)) and torch.equal(features, torch.ones(2, 1))


if __name__ == '__main__':
    test_metrics_do_not_align_away_docking_error()
    test_initialization_ignores_native_coordinates()
    test_receptor_crop_preserves_all_geometrically_visible_nodes_with_empty_edges()
    test_receptor_crop_relabels_edges_without_reordering_nodes_or_detaching_inputs()
    test_graph_entry_rejects_empty_components_but_preserves_singleton_inputs()
    test_receptor_crop_rejects_invalid_radius_and_fallback_counts()
    print('Independent binary baseline checks passed')
