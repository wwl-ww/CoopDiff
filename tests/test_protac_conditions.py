"""Binary-frame and chemical atom-mapping gates independent of ternary native."""

from pathlib import Path
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.protac_conditions import attachment_bonds, fragment_matches, local_pocket, mapping_diagnostics


def test_fragment_atom_permutation_preserves_equivalent_mappings():
    full, fragment = Chem.MolFromSmiles('CCOCCN'), Chem.MolFromSmiles('CO')
    original = fragment_matches(full, fragment)
    order = list(reversed(range(full.GetNumAtoms())))
    reordered = fragment_matches(Chem.RenumberAtoms(full, order), fragment)
    recovered = sorted(tuple(order[i] for i in match) for match in reordered['atom_maps'])
    assert recovered == sorted(map(tuple, original['atom_maps']))
    assert original['mapping_status'] == 'ambiguous'
    assert attachment_bonds(full, [0, 1]) == [[1, 2]]


def test_binary_pocket_is_invariant_under_its_own_rigid_transform():
    protein = np.array([[0., 0., 0.], [3., 1., 2.], [20., 0., 0.]])
    fragment = np.array([[1., 0., 0.], [1., 2., 0.]])
    residues = ['A:1:', 'A:2:', 'A:3:']
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    first = local_pocket(protein, residues, fragment)
    transformed = local_pocket(protein @ rotation + 13, residues, fragment @ rotation + 13)
    assert first == transformed
    assert first['residue_ids'] == ['A:1:', 'A:2:']


def test_relaxed_chemistry_is_diagnostic_not_a_valid_mapping():
    aromatic, saturated = Chem.MolFromSmiles('c1ccccc1'), Chem.MolFromSmiles('C1CCCCC1')
    report = mapping_diagnostics(aromatic, saturated)
    assert report['element_connectivity_match']
    assert not report['chemistry_match_ignoring_stereo']
    assert not report['relaxed_mapping_usable_for_inference']
    assert fragment_matches(aromatic, saturated)['mapping_status'] == 'unmatched'
    unrelated = mapping_diagnostics(aromatic, Chem.MolFromSmiles('C1CCNCC1'))
    assert not unrelated['element_connectivity_match']
