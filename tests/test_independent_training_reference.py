"""Reference construction must not see native ligand coordinates/conformers."""

from pathlib import Path
import sys
from unittest.mock import patch

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_bounded_binary_flow import bounded
from tricoopdiff.independent_training_reference import build_independent_training_reference


def test_target_coordinates_are_stripped_before_deployment_builder():
    flow = bounded()
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCCCC'))
    AllChem.EmbedMolecule(mol, randomSeed=7315)
    mol = Chem.RemoveHs(mol)
    ligand = dict(mol=mol, coords=torch.full((1, 8, 3), float('nan')),
                  edge_index=torch.empty(2, 0, dtype=torch.long), edge_mask=torch.zeros(0, dtype=torch.bool),
                  mask_rotate=torch.empty(0, 8, dtype=torch.bool))

    def builder(initializer, e3, poi, data, **kwargs):
        assert set(data) == {'mol', 'edge_index', 'edge_mask', 'mask_rotate'}
        assert data['mol'].GetNumConformers() == 0
        return [(flow.strict, {'selected_atom_maps': [flow.end1, flow.end2]})], {'native_information_used': False}

    with patch('tricoopdiff.independent_training_reference.build_binary_chart_bank', side_effect=builder):
        result, audit = build_independent_training_reference(
            (object(), object()), flow.strict.poi, flow.strict.poi, ligand,
            seed=7315, generator=torch.Generator().manual_seed(7315),
            binary_bounds={'translation': [.5, .5], 'rotation': [.2, .2]},
            evidence_provenance='synthetic_native_local_pairs')
    assert torch.equal(result.strict.ligand, flow.strict.ligand)
    assert audit['native_information_used'] is True
    assert audit['full_ligand_input_coordinates_used'] is False
    assert mol.GetNumConformers() == 1, 'input chemistry object was mutated'


if __name__ == '__main__':
    test_target_coordinates_are_stripped_before_deployment_builder()
    print('Independent reference boundary checks passed')
