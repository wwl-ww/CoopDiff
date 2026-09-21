"""Conserved-core discovery must not bypass parent chemistry/stereochemistry."""

from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.conserved_binary_cores import conserved_binary_cores, _parent_stereo_agrees
from tricoopdiff.conserved_core_geometry import core_geometry_evidence


def test_parent_stereo_is_permutation_invariant_and_rejects_opposite_enantiomer():
    source = Chem.MolFromSmiles('N[C@@H](C)C(=O)O')
    order = list(reversed(range(source.GetNumAtoms())))
    target = Chem.RenumberAtoms(source, order)
    mapping = {old: new for new, old in enumerate(order)}
    assert _parent_stereo_agrees(source, target, mapping)
    mirror = Chem.MolFromSmiles('N[C@H](C)C(=O)O')
    assert not _parent_stereo_agrees(source, mirror, dict(enumerate(range(source.GetNumAtoms()))))


def test_analogue_core_is_exact_not_a_relaxed_bond_match():
    source = Chem.MolFromSmiles('Clc1ccc2ncccc2c1')
    target = Chem.MolFromSmiles('Cc1ccc2ncccc2c1')
    cores, audit = conserved_binary_cores(target, source)
    assert cores and audit['core_atoms'] == 10 and audit['deployment_eligible'] is False
    for item in cores:
        assert item['source_coverage'] < 1
        assert all(mapping in target.GetSubstructMatches(item['fragment'], useChirality=True, uniquify=False)
                   for mapping in item['atom_maps'])
    incompatible = Chem.MolFromSmiles('C1CCC2NCCCC2C1')
    assert not conserved_binary_cores(incompatible, source)[0], 'aromatic core was transferred into saturated rings'


def test_canonical_core_geometry_keeps_maps_and_binary_coordinates_aligned():
    source = Chem.MolFromSmiles('Clc1ccc2ncccc2c1')
    target = Chem.MolFromSmiles('Cc1ccc2ncccc2c1')
    AllChem.Compute2DCoords(source)
    core = conserved_binary_cores(target, source)[0][0]
    protein = [[0., 0., 3.], [1., 0., 3.], [0., 1., 3.]]
    result = core_geometry_evidence(core, source, protein)
    query = Chem.MolFromSmiles(result['fragment_smiles'])
    matches = target.GetSubstructMatches(query, uniquify=False, useChirality=True)
    assert all(tuple(mapping) in matches for mapping in result['atom_maps'])
    original = source.GetConformer().GetPositions()
    for index, position in zip(result['source_atom_indices_in_smiles_order'], result['binary_fragment_coords']):
        assert list(original[index]) == position
    assert result['orientation_defined'] and result['core_contact_atoms'] > 0
    assert not result['deployment_eligible']
