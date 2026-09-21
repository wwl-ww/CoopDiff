"""Tautomer correspondence must not become unrestricted chemistry matching."""

from pathlib import Path
import sys

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.tautomer_binary_cores import canonical_tautomer_with_identity, tautomer_conserved_binary_cores


def test_lactam_lactim_core_matches_without_changing_atom_identity():
    left = Chem.MolFromSmiles('Oc1ccccn1')
    right = Chem.MolFromSmiles('O=c1cccc[nH]1')
    a, audit = canonical_tautomer_with_identity(left)
    b, _ = canonical_tautomer_with_identity(right)
    assert Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
    cores, report = tautomer_conserved_binary_cores(left, right, min_atoms=6)
    assert cores and report['coordinates_used_for_mapping'] is False
    assert audit['atom_order_and_stereocenters_preserved']


def test_reduced_ring_is_not_a_tautomer_match():
    aromatic = Chem.MolFromSmiles('O=c1cccc[nH]1')
    reduced = Chem.MolFromSmiles('O=C1CCCCN1')
    assert not tautomer_conserved_binary_cores(aromatic, reduced, min_atoms=6)[0]
