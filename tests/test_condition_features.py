"""Chemical roles, missing conditions, independent binary frames and encoder gradients."""

from dataclasses import replace
from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from tricoopdiff.condition_features import (SideCondition, ProtacConditionEncoder,
                                            build_condition_features, build_supervised_contact_features)
from tricoopdiff.protac_conditions import fragment_matches
from prepare_protac_binary_chemistry import restore_side
from test_restore_ligand_chemistry import fixture


def binary_side():
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
    left = binary_side()
    right = replace(left, fragment=Chem.MolFromSmiles('CN'), atom_map=(4, 5))
    original = build_condition_features(molecule, (2, 2), (left, right))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = replace(left, binary_fragment_coords=torch.tensor(left.binary_fragment_coords) @ rotation + 17,
                    binary_protein_coords=torch.tensor(left.binary_protein_coords) @ rotation + 17)
    changed = build_condition_features(molecule, (2, 2), (moved, right))
    assert all(torch.allclose(original[k], changed[k], atol=1e-6) for k in original)


def test_unknown_atoms_are_not_automatically_linker_and_explicit_roles_permute():
    molecule = Chem.MolFromSmiles('CCOCCN')
    side = replace(binary_side(), provenance='chemical_definition', pocket_probabilities=None,
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
    side = binary_side()
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
    side = binary_side()
    variants = [build_condition_features(
        molecule, (2, 2), (replace(side, provenance=provenance), None))
        for provenance in ('experimental_binary', 'template', 'docked', 'predicted')]
    assert all(torch.equal(variants[0][name], variant[name])
               for variant in variants[1:] for name in variants[0])
    assert variants[0]['sides'].shape == (2, 5)


def test_invalid_chemistry_unverified_provenance_and_overlapping_ends_rejected():
    molecule, side = Chem.MolFromSmiles('CCOCCN'), binary_side()
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


def test_conservative_mcs_requires_explicit_low_confidence_audit_marker():
    molecule = Chem.MolFromSmiles('CCO')
    side = SideCondition(
        provenance='template', evidence_id='fixture:mcs', confidence=.5,
        fragment=Chem.MolFromSmiles('CC'), atom_map=(0, 1),
        mapping_method='conservative_charge_normalized_mcs')
    for unsafe in (side, replace(side, audit_only=True, confidence=.6)):
        try:
            build_condition_features(molecule, (2, 2), (unsafe, None))
        except ValueError:
            pass
        else:
            raise AssertionError('conservative MCS entered deployment without both audit gates')
    features = build_condition_features(
        molecule, (2, 2), (replace(side, audit_only=True), None))
    assert features['ligand'][:, 0].tolist() == [1., 1., 0.]


def test_symmetry_can_fix_fragment_roles_without_fixing_atom_correspondence():
    report = fragment_matches(Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC'))
    assert report['mapping_status'] == 'ambiguous'
    assert report['unambiguous_role_atoms'] == [0, 1]
    truncated = fragment_matches(Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC'), limit=1)
    assert truncated['unambiguous_role_atoms'] is None


def test_binary_geometry_averages_all_chemical_symmetry_maps():
    molecule = Chem.MolFromSmiles('CCO')
    fragment = Chem.MolFromSmiles('CC')
    maps = tuple(fragment_matches(molecule, fragment)['atom_maps'])
    side = SideCondition(
        provenance='experimental_binary', evidence_id='fixture:symmetric',
        fragment=fragment, atom_maps=maps, pocket_probabilities=[1., 0.],
        binary_fragment_coords=[[0., 0., 0.], [2., 0., 0.]],
        binary_protein_coords=[[0., 1., 0.], [5., 0., 0.]],
    )
    first = build_condition_features(molecule, (2, 2), (side, None))
    swapped = replace(side, binary_fragment_coords=list(reversed(side.binary_fragment_coords)))
    second = build_condition_features(molecule, (2, 2), (swapped, None))
    assert torch.allclose(first['ligand'], second['ligand'])
    assert first['ligand'][:2, 0].tolist() == [1., 1.]


def test_ccd_restoration_wrapper_preserves_geometry_and_does_not_enable_pose(tmp_path):
    pdb, ccd, _ = fixture(tmp_path)
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
    assert features['ligand'][:2, [7, 12]].tolist() == [[1., 1.], [1., 1.]]
    assert features['protein1'].shape[1] == 6
    assert features['ligand'].shape[1] == 14
    assert features['sides'][:, 2].tolist() == [1., 1.]
    assert set(features) == {'ligand', 'protein1', 'protein2', 'sides', 'binary1', 'binary2'}
    assert features['binary1'].shape == (1, 3, 2, 3)
    expanded = build_supervised_contact_features(
        molecule, proteins, ligand, cutoff=6., role_graph_radius=1)
    assert expanded['ligand'][:, 0].tolist() == [1., 1., 1.]
    assert expanded['ligand'][:, 7].tolist() == [1., 1., 1.]
    assert expanded['sides'][0, 4] == 1., 'role coverage must record expanded warhead proxy'
    assert torch.equal(expanded['protein1'][:, 0], features['protein1'][:, 0]), (
        'role expansion must not alter the physical pocket mask')
    try:
        build_supervised_contact_features(
            molecule, proteins, ligand, role_graph_radius=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('negative training role radius accepted')
    invalid = SideCondition('training_native_supervision', 'must-not-enter-inference',
                            pocket_probabilities=[1., 0.])
    try:
        build_condition_features(molecule, (2, 2), (invalid, None), protac=False)
    except ValueError:
        pass
    else:
        raise AssertionError('native-derived supervision entered deployable condition API')
