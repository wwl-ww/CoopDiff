"""All legal references are available without selecting by native residual."""

from pathlib import Path
import sys
from unittest.mock import patch

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_bounded_binary_flow import bounded
from tricoopdiff.independent_training_reference import build_independent_training_reference_bank


def test_bank_returns_all_candidates_and_selection_is_external():
    flow = bounded()
    mol = Chem.MolFromSmiles('CCCCCCCC')
    ligand = dict(mol=mol, edge_index=torch.empty(2, 0, dtype=torch.long),
                  edge_mask=torch.zeros(0, dtype=torch.bool), mask_rotate=torch.empty(0, 8, dtype=torch.bool))
    selected = [(flow.strict, {'selected_atom_maps': [flow.end1, flow.end2]}),
                (flow.strict, {'selected_atom_maps': [flow.end1, flow.end2]})]
    with patch('tricoopdiff.independent_training_reference.build_binary_chart_bank', return_value=(selected, {'native_information_used': False})):
        bank, audit = build_independent_training_reference_bank(
            (object(), object()), flow.strict.poi, flow.strict.poi, ligand, seed=7315,
            binary_bounds={'translation': [.5, .5], 'rotation': [.2, .2]},
            evidence_provenance='synthetic_native_local_pairs')
    assert len(bank) == 2 and audit['reference_selection'].startswith('uniform')
    assert audit['native_information_used'] is True


if __name__ == '__main__':
    test_bank_returns_all_candidates_and_selection_is_external()
    print('Reference bank checks passed')
