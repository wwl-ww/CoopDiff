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


if __name__ == '__main__':
    test_metrics_do_not_align_away_docking_error()
    test_initialization_ignores_native_coordinates()
    print('Independent binary baseline checks passed')
