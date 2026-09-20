"""Deploy-time PROTAC manifest identity and leakage boundaries."""

import json
from pathlib import Path
import sys
import tempfile

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tricoopdiff.condition_manifest import (MANIFEST_VERSION, canonical_isomeric_smiles,
                                            load_binary_pose_sides, load_condition_manifest)


def payload(molecule):
    return {
        'schema': MANIFEST_VERSION, 'case': 'fixture',
        'ligand_atom_count': molecule.GetNumAtoms(),
        'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(molecule),
        'sides': [
            {'provenance': 'chemical_definition', 'evidence_id': 'design:left',
             'fragment_smiles': 'CC', 'atom_map': [0, 1]},
            {'provenance': 'template', 'evidence_id': 'template:right',
             'fragment_smiles': 'ON', 'atom_map': [2, 3],
             'pocket_probabilities': [0.1, 0.9, 0.2]},
        ],
        'linker_indices': [], 'protac': True,
    }


def write(value, directory):
    path = Path(directory) / 'condition.json'
    path.write_text(json.dumps(value))
    return path


def test_manifest_builds_aligned_features_without_native_geometry():
    molecule = Chem.MolFromSmiles('CCON')
    with tempfile.TemporaryDirectory() as directory:
        features, audit = load_condition_manifest(
            write(payload(molecule), directory), case='fixture', molecule=molecule,
            protein_sizes=(2, 3))
    assert features['protein1'].shape == (2, 6)
    assert features['protein2'].shape == (3, 6)
    assert features['ligand'][:, :2].tolist() == [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]
    assert audit['native_ternary_geometry_read'] is False


def test_manifest_rejects_case_chemistry_unknown_fields_and_native_provenance():
    molecule = Chem.MolFromSmiles('CCON')
    variants = []
    wrong_case = payload(molecule); wrong_case['case'] = 'other'; variants.append(wrong_case)
    wrong_ligand = payload(molecule); wrong_ligand['ligand_atom_count'] = 5; variants.append(wrong_ligand)
    unknown = payload(molecule); unknown['native_coords'] = []; variants.append(unknown)
    native = payload(molecule); native['sides'][0]['provenance'] = 'training_native_supervision'; variants.append(native)
    with tempfile.TemporaryDirectory() as directory:
        for index, value in enumerate(variants):
            try:
                load_condition_manifest(write(value, directory), case='fixture', molecule=molecule,
                                        protein_sizes=(2, 3))
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid manifest variant {index} accepted')


def test_binary_pose_loader_returns_only_complete_local_pose_sides():
    molecule = Chem.MolFromSmiles('CCON')
    value = payload(molecule)
    value['sides'][0].update(
        provenance='experimental_binary',
        atom_maps=[[0, 1]],
        binary_fragment_coords=[[0., 0., 0.], [1., 0., 0.]],
        binary_protein_coords=[[0., 1., 0.], [1., 1., 0.]],
    )
    with tempfile.TemporaryDirectory() as directory:
        sides, audit = load_binary_pose_sides(
            write(value, directory), case='fixture', molecule=molecule,
            protein_sizes=(2, 3))
    assert sides[0] is not None and sides[0].atom_maps == ((0, 1),)
    assert sides[1] is None
    assert audit['binary_pose_sides'] == [True, False]


if __name__ == '__main__':
    test_manifest_builds_aligned_features_without_native_geometry()
    test_manifest_rejects_case_chemistry_unknown_fields_and_native_provenance()
