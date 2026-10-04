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


if __name__ == '__main__':
    test_metrics_do_not_align_away_docking_error()
    test_initialization_ignores_native_coordinates()
    test_receptor_crop_preserves_all_geometrically_visible_nodes_with_empty_edges()
    test_receptor_crop_relabels_edges_without_reordering_nodes_or_detaching_inputs()
    print('Independent binary baseline checks passed')
