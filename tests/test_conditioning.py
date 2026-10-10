"""Conditioning regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from helpers import chemistry_example
from make_protac_pocket_manifests import (
    build_case,
    pdb_coordinates,
    pocket_probabilities,
    receptor_node_coordinates,
)
from prepare_protac_binary_chemistry import restore_side
from tricoopdiff.conditioning.condition_augmentation import (
    ConditionAugmentation,
    ConditionFeatureDropout,
    augment_conditions,
    dropout_condition_features,
    protein_local_basis,
)
from tricoopdiff.conditioning.condition_features import (
    MANIFEST_VERSION,
    ProtacConditionEncoder,
    SideCondition,
    attachment_bonds,
    build_condition_features,
    build_supervised_contact_features,
    canonical_isomeric_smiles,
    fragment_matches,
    load_condition_manifest,
    local_pocket,
    mapping_diagnostics,
)


# Condition features


def condition_features_binary_side():
    return SideCondition(provenance='experimental_binary', evidence_id='fixture:binary',
                         fragment=Chem.MolFromSmiles('CO'), atom_map=(1, 2),
                         pocket_probabilities=[.8, .1],
                         binary_fragment_coords=[[1., 0., 0.], [2., 1., 0.]],
                         binary_protein_coords=[[0., 0., 0.], [5., 1., 0.]])


def test_absent_conditions_are_exact_zero_even_with_encoder_bias():
    features = build_condition_features(Chem.MolFromSmiles('CCOCCN'), (2, 3))
    encoder = ProtacConditionEncoder(8)
    assert torch.equal(features['ligand'][:, 3], torch.ones(6))
    assert all(torch.count_nonzero(value) == 0 for value in encoder(features).values())


def test_two_binary_frames_transform_independently_without_changing_features():
    molecule = Chem.MolFromSmiles('CCOCCN')
    left = condition_features_binary_side()
    right = replace(left, fragment=Chem.MolFromSmiles('CN'), atom_map=(4, 5))
    original = build_condition_features(molecule, (2, 2), (left, right))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = replace(left, binary_fragment_coords=torch.tensor(left.binary_fragment_coords) @ rotation + 17,
                    binary_protein_coords=torch.tensor(left.binary_protein_coords) @ rotation + 17)
    changed = build_condition_features(molecule, (2, 2), (moved, right))
    assert all(torch.allclose(original[k], changed[k], atol=1e-6) for k in original)


def test_unknown_atoms_are_not_automatically_linker_and_explicit_roles_permute():
    molecule = Chem.MolFromSmiles('CCOCCN')
    side = replace(condition_features_binary_side(), provenance='chemical_definition', pocket_probabilities=None,
                   binary_fragment_coords=None, binary_protein_coords=None)
    original = build_condition_features(molecule, (2, 2), (side, None), linker_indices=(3,))
    assert original['ligand'][:, 2].sum() == 1
    assert original['ligand'][0, 3] == 1  # Unannotated decoration stays unknown.
    order = list(reversed(range(6)))
    remapped = replace(side, atom_map=tuple(order.index(i) for i in side.atom_map))
    changed = build_condition_features(Chem.RenumberAtoms(molecule, order), (2, 2),
                                       (remapped, None), linker_indices=(order.index(3),))
    assert torch.equal(original['ligand'][order], changed['ligand'])


def test_missing_pocket_differs_from_known_negative_and_gradients_flow():
    molecule = Chem.MolFromSmiles('CCOCCN')
    side = condition_features_binary_side()
    missing = build_condition_features(molecule, (2, 2), (replace(side, pocket_probabilities=None), None))
    negative = build_condition_features(molecule, (2, 2), (replace(side, pocket_probabilities=[0., 0.]), None))
    assert missing['protein1'][:, 1].sum() == 0
    assert negative['protein1'][:, 1].sum() == 2
    encoder = ProtacConditionEncoder(8)
    output = encoder(negative)
    sum(value.square().sum() for value in output.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert any(torch.count_nonzero(p.grad) for p in encoder.parameters())


def test_provenance_is_audited_but_not_an_untrained_model_feature():
    molecule = Chem.MolFromSmiles('CCOCCN')
    side = condition_features_binary_side()
    variants = [build_condition_features(
        molecule, (2, 2), (replace(side, provenance=provenance), None))
        for provenance in ('experimental_binary', 'template', 'docked', 'predicted')]
    assert all(torch.equal(variants[0][name], variant[name])
               for variant in variants[1:] for name in variants[0])
    assert variants[0]['sides'].shape == (2, 4)


def test_invalid_chemistry_unverified_provenance_and_overlapping_ends_rejected():
    molecule, side = Chem.MolFromSmiles('CCOCCN'), condition_features_binary_side()
    invalid = [replace(side, atom_map=(0, 1)), replace(side, provenance='native'),
               replace(side, provenance='unverified'), replace(side, binary_protein_coords=None),
               replace(side, pocket_probabilities=[2., 0.]), replace(side, confidence=float('nan')),
               replace(side, provenance='chemical_definition', binary_fragment_coords=None, binary_protein_coords=None)]
    for item in invalid:
        try:
            build_condition_features(molecule, (2, 2), (item, None))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid or unverified condition accepted')
    try:
        build_condition_features(molecule, (2, 2), (side, side))
    except ValueError:
        pass
    else:
        raise AssertionError('PROTAC ends must not overlap')
    generic = build_condition_features(molecule, (2, 2), (side, side), protac=False)
    assert generic['ligand'][1, :2].tolist() == [1., 1.]


def test_symmetry_can_fix_fragment_roles_without_fixing_atom_correspondence():
    report = fragment_matches(Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC'))
    assert report['mapping_status'] == 'ambiguous'
    assert report['unambiguous_role_atoms'] == [0, 1]
    truncated = fragment_matches(Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC'), limit=1)
    assert truncated['unambiguous_role_atoms'] is None


def test_ccd_restoration_wrapper_preserves_geometry_and_does_not_enable_pose(tmp_path):
    pdb, ccd, _ = chemistry_example(tmp_path)
    full = Chem.MolFromSmiles('CC(=O)[O-]')
    result = restore_side(pdb, ccd.parent, tmp_path / 'restored.sdf', full)
    assert result['coordinates_preserved']
    assert result['mapping']['unambiguous_role_atoms'] == [0, 1, 2, 3]
    assert not result['binary_pose_provenance_verified']
    assert not result['ready_for_conditioned_inference']


def test_native_training_contacts_make_deepternary_style_masks_without_coordinates():
    molecule = Chem.MolFromSmiles('CCO')
    ligand = torch.tensor([[0., 0., 0.], [6., 0., 0.], [20., 0., 0.]])
    proteins = (torch.tensor([[3., 0., 0.], [30., 0., 0.]]),
                torch.tensor([[3., 1., 0.], [40., 0., 0.]]))
    features = build_supervised_contact_features(molecule, proteins, ligand, cutoff=6.)
    assert features['ligand'][:, 0].tolist() == [1., 1., 0.]
    assert features['ligand'][:, 1].tolist() == [1., 1., 0.]
    assert features['protein1'][:, 0].tolist() == [1., 0.]
    assert features['protein2'][:, 0].tolist() == [1., 0.]
    assert features['ligand'][0, :2].sum() == 2, 'generic ternary contact roles may overlap'
    assert features['ligand'][:2, 6:8].tolist() == [[1., 1.], [1., 1.]]
    assert features['sides'][:, 2].tolist() == [1., 1.]
    assert set(features) == {'ligand', 'protein1', 'protein2', 'sides'}
    assert all(value.ndim == 2 for value in features.values())
    invalid = SideCondition('training_native_supervision', 'must-not-enter-inference',
                            pocket_probabilities=[1., 0.])
    try:
        build_condition_features(molecule, (2, 2), (invalid, None), protac=False)
    except ValueError:
        pass
    else:
        raise AssertionError('native-derived supervision entered deployable condition API')


# Condition augmentation


def condition_augmentation_side():
    return SideCondition(provenance='experimental_binary', evidence_id='fixture:binary',
                         fragment=Chem.MolFromSmiles('CO'), atom_map=(1, 2), confidence=.8,
                         pocket_probabilities=np.array([.9, .2, .1, 0.]),
                         binary_fragment_coords=np.array([[1., 0., 0.], [2., 1., .3]]),
                         binary_protein_coords=np.array([[0., 0., 0.], [5., 0., 0.], [1., 2., 0.], [0., 1., 3.]]))


def test_augmentation_identity_no_mutation_and_global_rng_preserved():
    original = condition_augmentation_side()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    sides, linker, audit = augment_conditions((original, None), seed=31, linker_indices=(3,))
    assert linker == (3,) and audit['events'] == []
    assert np.array_equal(sides[0].binary_fragment_coords, original.binary_fragment_coords)
    sides[0].binary_fragment_coords[0] += 100
    assert not np.array_equal(sides[0].binary_fragment_coords, original.binary_fragment_coords)
    assert python_rng == random.getstate()
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert torch.equal(torch_rng, torch.random.get_rng_state())


def test_drop_all_removes_linker_and_encodes_exact_zero():
    sides, linker, audit = augment_conditions((condition_augmentation_side(), condition_augmentation_side()), seed=7, linker_indices=(3,),
                                            config=ConditionAugmentation(drop_all_probability=1.))
    assert sides == (None, None) and not linker
    features = build_condition_features(Chem.MolFromSmiles('CCOC'), (4, 4), sides, linker_indices=linker)
    assert all(torch.count_nonzero(v) == 0 for v in ProtacConditionEncoder(8)(features).values())
    assert not audit['flow_state_modified'] and not audit['candidate_source_modified']


def test_pocket_and_binary_dropout_do_not_erase_chemical_roles():
    config = ConditionAugmentation(drop_pocket_probability=1., drop_binary_probability=1.)
    sides, _, _ = augment_conditions((condition_augmentation_side(), None), seed=9, config=config)
    assert sides[0].atom_map == (1, 2)
    assert sides[0].pocket_probabilities is None and sides[0].binary_fragment_coords is None
    features = build_condition_features(Chem.MolFromSmiles('CCOC'), (4, 4), sides)
    assert features['ligand'][:, 0].sum() == 2
    assert features['protein1'][:, 1].sum() == 0


def test_binary_noise_is_replayable_and_preserves_internal_distances():
    original = condition_augmentation_side()
    config = ConditionAugmentation(binary_translation_std=.5, binary_rotation_std=.2,
                                   pocket_noise_std=.1, perturbed_confidence_scale=.5)
    first, _, record = augment_conditions((original, None), seed=13, config=config)
    second, _, replay = augment_conditions((original, None), seed=13, config=config)
    assert record == replay
    assert np.array_equal(first[0].binary_fragment_coords, second[0].binary_fragment_coords)
    before, after = original.binary_fragment_coords, first[0].binary_fragment_coords
    assert not np.allclose(before, after)
    assert np.allclose(np.linalg.norm(before[:, None] - before[None], axis=-1),
                       np.linalg.norm(after[:, None] - after[None], axis=-1), atol=1e-8)
    assert np.array_equal(original.binary_protein_coords, first[0].binary_protein_coords)
    assert first[0].confidence == .4 and '#synthetic-condition-noise' in first[0].evidence_id


def test_fixed_seed_perturbation_is_equivariant_under_independent_binary_frame_change():
    original = condition_augmentation_side()
    config = ConditionAugmentation(binary_translation_std=.5, binary_rotation_std=.2)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = replace(original, binary_fragment_coords=original.binary_fragment_coords @ rotation + 11,
                    binary_protein_coords=original.binary_protein_coords @ rotation + 11)
    base, _, _ = augment_conditions((original, None), seed=73, config=config)
    changed, _, _ = augment_conditions((moved, None), seed=73, config=config)
    assert np.allclose(base[0].binary_fragment_coords @ rotation + 11, changed[0].binary_fragment_coords)


def test_side_decisions_are_independent_and_invalid_frames_fail_closed():
    config = ConditionAugmentation(drop_side_probability=.5, pocket_noise_std=.1)
    first, _, _ = augment_conditions((condition_augmentation_side(), condition_augmentation_side()), seed=91, config=config)
    second, _, _ = augment_conditions((None, condition_augmentation_side()), seed=91, config=config)
    assert (first[1] is None) == (second[1] is None)
    if first[1] is not None:
        assert np.array_equal(first[1].pocket_probabilities, second[1].pocket_probabilities)
    for coords in (np.zeros((4, 3)), np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])):
        try:
            protein_local_basis(coords)
        except ValueError:
            pass
        else:
            raise AssertionError('degenerate frame cannot use an arbitrary world-axis fallback')
    try:
        ConditionAugmentation(drop_all_probability=1.1).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid probability accepted')


def test_training_feature_dropout_marks_unknown_and_never_returns_native_coordinates():
    molecule = Chem.MolFromSmiles('CCO')
    features = build_supervised_contact_features(
        molecule, (torch.tensor([[0., 1., 0.], [5., 0., 0.]]),
                   torch.tensor([[1., 1., 0.], [8., 0., 0.]])),
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [20., 0., 0.]]))
    original = {key: value.clone() for key, value in features.items()}
    dropped, audit = dropout_condition_features(features, seed=17,
        config=ConditionFeatureDropout(drop_pocket_probability=1., drop_ligand_role_probability=1.,
                                       drop_binary_probability=1.))
    assert dropped['protein1'].count_nonzero() == dropped['protein2'].count_nonzero() == 0
    assert dropped['ligand'][:, :2].count_nonzero() == 0
    assert torch.equal(dropped['ligand'][:, 3], torch.ones(3))
    assert dropped['sides'].count_nonzero() == 0
    assert all(torch.equal(features[k], original[k]) for k in features)
    assert not audit['native_coordinates_returned'] and not audit['flow_state_modified']
    replay, replay_audit = dropout_condition_features(features, seed=17,
        config=ConditionFeatureDropout(drop_pocket_probability=1., drop_ligand_role_probability=1.,
                                       drop_binary_probability=1.))
    assert audit == replay_audit and all(torch.equal(dropped[k], replay[k]) for k in dropped)


def test_training_feature_dropout_has_explicit_two_sided_pocket_only_mode():
    molecule = Chem.MolFromSmiles('CCO')
    features = build_supervised_contact_features(
        molecule, (torch.tensor([[0., 1., 0.], [5., 0., 0.]]),
                   torch.tensor([[1., 1., 0.], [8., 0., 0.]])),
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [20., 0., 0.]]))
    result, audit = dropout_condition_features(
        features, seed=3, config=ConditionFeatureDropout(
            0., 0., 0., 0., 0., pocket_only_probability=1.))
    assert result['sides'][:, 0].sum() == 0
    assert result['sides'][:, 1].sum() == 2
    assert result['sides'][:, 2].sum() == 0
    assert result['protein1'].sum() > 0 and result['protein2'].sum() > 0
    assert audit['events'] == [{'action': 'keep_pocket_only_both_sides'}]


# Condition manifest


def condition_manifest_payload(molecule):
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


def condition_manifest_write(value, directory):
    path = Path(directory) / 'condition.json'
    path.write_text(json.dumps(value))
    return path


def test_manifest_builds_aligned_features_without_native_geometry():
    molecule = Chem.MolFromSmiles('CCON')
    with tempfile.TemporaryDirectory() as directory:
        features, audit = load_condition_manifest(
            condition_manifest_write(condition_manifest_payload(molecule), directory), case='fixture', molecule=molecule,
            protein_sizes=(2, 3))
    assert features['protein1'].shape == (2, 3)
    assert features['protein2'].shape == (3, 3)
    assert features['ligand'][:, :2].tolist() == [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]
    assert audit['native_ternary_geometry_read'] is False


def test_manifest_rejects_case_chemistry_unknown_fields_and_native_provenance():
    molecule = Chem.MolFromSmiles('CCON')
    variants = []
    wrong_case = condition_manifest_payload(molecule); wrong_case['case'] = 'other'; variants.append(wrong_case)
    wrong_ligand = condition_manifest_payload(molecule); wrong_ligand['ligand_atom_count'] = 5; variants.append(wrong_ligand)
    unknown = condition_manifest_payload(molecule); unknown['native_coords'] = []; variants.append(unknown)
    native = condition_manifest_payload(molecule); native['sides'][0]['provenance'] = 'training_native_supervision'; variants.append(native)
    with tempfile.TemporaryDirectory() as directory:
        for index, value in enumerate(variants):
            try:
                load_condition_manifest(condition_manifest_write(value, directory), case='fixture', molecule=molecule,
                                        protein_sizes=(2, 3))
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid manifest variant {index} accepted')


# Protac conditions

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


# Protac pocket manifests


def protac_pocket_manifests_pdb_line(record, serial, name, residue, x, y, z, element, residue_number=1):
    return (f'{record:<6}{serial:5d} {name:^4} {residue:>3} A{residue_number:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {element:>2}\n')


def test_binary_template_pockets_are_local_and_node_aligned(tmp_path):
    case_dir = tmp_path / 'case'
    case_dir.mkdir()
    protein = ''.join([
        protac_pocket_manifests_pdb_line('ATOM', 1, 'CA', 'ALA', 0, 0, 0, 'C', 1),
        protac_pocket_manifests_pdb_line('ATOM', 2, 'CA', 'GLY', 10, 0, 0, 'C', 2),
        'END\n',
    ])
    fragment = protac_pocket_manifests_pdb_line('HETATM', 1, 'C1', 'LIG', 1, 0, 0, 'C') + 'END\n'
    for side in (1, 2):
        (case_dir / f'protein{side}.pdb').write_text(protein)
        (case_dir / f'unbound_lig{side}.pdb').write_text(fragment)
    molecule = Chem.AddHs(Chem.MolFromSmiles('CC'))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    Chem.MolToMolFile(molecule, str(case_dir / 'ligand.sdf'))
    entry = {
        'case': 'fixture', 'protein1': 'case/protein1.pdb',
        'protein2': 'case/protein2.pdb', 'ligand': 'case/ligand.sdf',
    }
    pocket, absent, report = build_case(entry, tmp_path, 6.)
    assert pocket['sides'][0]['pocket_probabilities'] == [1., 0.]
    assert pocket['sides'][1]['pocket_probabilities'] == [1., 0.]
    assert absent['sides'] == [None, None]
    assert report['sides'][0]['pocket_ca_nodes'] == 1
    assert pdb_coordinates(case_dir / 'protein1.pdb', alpha_carbons=True).shape == (2, 3)
    values, _, _ = pocket_probabilities(
        case_dir / 'protein1.pdb', case_dir / 'unbound_lig1.pdb', 6.)
    assert np.array_equal(values, [1., 0.])


def test_modified_residue_is_excluded_like_diffdock(tmp_path):
    protein = ''.join([
        protac_pocket_manifests_pdb_line('ATOM', 1, 'CA', 'ALA', 0, 0, 0, 'C', 1),
        protac_pocket_manifests_pdb_line('ATOM', 2, 'CA', 'CAS', 2, 0, 0, 'C', 2),
        protac_pocket_manifests_pdb_line('ATOM', 3, 'CA', 'GLY', 10, 0, 0, 'C', 3),
        'END\n',
    ])
    path = tmp_path / 'modified.pdb'
    path.write_text(protein)
    # The raw PDB has three CA rows, but DiffDock/ProDy constructs two protein
    # nodes because CAS is a modified residue outside the protein selection.
    assert pdb_coordinates(path, alpha_carbons=True).shape == (3, 3)
    assert receptor_node_coordinates(path).shape == (2, 3)
