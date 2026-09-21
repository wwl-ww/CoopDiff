"""A corrected chemical graph cannot silently move or reorder training atoms."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_initialization import ligand_input
from tricoopdiff.ligand_chemistry import rebuild_ligand_graph


def test_rebuild_preserves_frame_and_rejects_coordinate_changes():
    data = ligand_input('CC(=O)NCCC')
    data['coords'] = data['coords'][0] + 10
    molecule = Chem.Mol(data['mol'])
    restored, audit = rebuild_ligand_graph(molecule, data)
    assert torch.equal(restored['coords'], data['coords'])
    assert audit['coordinates_changed'] is False
    assert torch.equal(restored['edge_mask'], data['edge_mask'])
    edited = Chem.Mol(molecule)
    point = edited.GetConformer().GetAtomPosition(0)
    point.x += .2
    edited.GetConformer().SetAtomPosition(0, point)
    try:
        rebuild_ligand_graph(edited, data)
    except ValueError as error:
        assert 'moved/reordered' in str(error)
    else:
        raise AssertionError('coordinate changes must not be accepted as chemistry repair')


def test_rebuild_replaces_old_bond_features_not_only_serialized_molecule():
    data = ligand_input('CC(=O)NCCC')
    data['coords'] = data['coords'][0]
    corrected = Chem.Mol(data['mol'])
    wrong = Chem.RWMol(corrected)
    wrong.GetBondBetweenAtoms(1, 2).SetBondType(Chem.BondType.SINGLE)
    wrong = wrong.GetMol()
    Chem.SanitizeMol(wrong)
    old, _ = rebuild_ligand_graph(wrong, data)
    new, audit = rebuild_ligand_graph(corrected, old)
    assert audit['old_smiles'] != audit['new_smiles']
    assert not torch.equal(old['edge_attr'], new['edge_attr'])
    assert not torch.equal(old['features'], new['features'])
    assert torch.equal(new['coords'], old['coords'])


if __name__ == '__main__':
    test_rebuild_preserves_frame_and_rejects_coordinate_changes()
    test_rebuild_replaces_old_bond_features_not_only_serialized_molecule()
    print('Ligand chemistry migration checks passed')
