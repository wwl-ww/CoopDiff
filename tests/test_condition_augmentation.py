"""Corrupt conditions without altering chemistry, protein frames or global RNG."""

from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tricoopdiff.conditioning.condition_augmentation import (ConditionAugmentation, ConditionFeatureDropout,
                                                augment_conditions, dropout_condition_features,
                                                protein_local_basis)
from tricoopdiff.conditioning.condition_features import (SideCondition, build_condition_features,
                                            build_supervised_contact_features, ProtacConditionEncoder)


def side():
    return SideCondition(provenance='experimental_binary', evidence_id='fixture:binary',
                         fragment=Chem.MolFromSmiles('CO'), atom_map=(1, 2), confidence=.8,
                         pocket_probabilities=np.array([.9, .2, .1, 0.]),
                         binary_fragment_coords=np.array([[1., 0., 0.], [2., 1., .3]]),
                         binary_protein_coords=np.array([[0., 0., 0.], [5., 0., 0.], [1., 2., 0.], [0., 1., 3.]]))


def test_augmentation_identity_no_mutation_and_global_rng_preserved():
    original = side()
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
    sides, linker, audit = augment_conditions((side(), side()), seed=7, linker_indices=(3,),
                                            config=ConditionAugmentation(drop_all_probability=1.))
    assert sides == (None, None) and not linker
    features = build_condition_features(Chem.MolFromSmiles('CCOC'), (4, 4), sides, linker_indices=linker)
    assert all(torch.count_nonzero(v) == 0 for v in ProtacConditionEncoder(8)(features).values())
    assert not audit['flow_state_modified'] and not audit['candidate_source_modified']


def test_pocket_and_binary_dropout_do_not_erase_chemical_roles():
    config = ConditionAugmentation(drop_pocket_probability=1., drop_binary_probability=1.)
    sides, _, _ = augment_conditions((side(), None), seed=9, config=config)
    assert sides[0].atom_map == (1, 2)
    assert sides[0].pocket_probabilities is None and sides[0].binary_fragment_coords is None
    features = build_condition_features(Chem.MolFromSmiles('CCOC'), (4, 4), sides)
    assert features['ligand'][:, 0].sum() == 2
    assert features['protein1'][:, 1].sum() == 0


def test_binary_noise_is_replayable_and_preserves_internal_distances():
    original = side()
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
    original = side()
    config = ConditionAugmentation(binary_translation_std=.5, binary_rotation_std=.2)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = replace(original, binary_fragment_coords=original.binary_fragment_coords @ rotation + 11,
                    binary_protein_coords=original.binary_protein_coords @ rotation + 11)
    base, _, _ = augment_conditions((original, None), seed=73, config=config)
    changed, _, _ = augment_conditions((moved, None), seed=73, config=config)
    assert np.allclose(base[0].binary_fragment_coords @ rotation + 11, changed[0].binary_fragment_coords)


def test_side_decisions_are_independent_and_invalid_frames_fail_closed():
    config = ConditionAugmentation(drop_side_probability=.5, pocket_noise_std=.1)
    first, _, _ = augment_conditions((side(), side()), seed=91, config=config)
    second, _, _ = augment_conditions((None, side()), seed=91, config=config)
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
