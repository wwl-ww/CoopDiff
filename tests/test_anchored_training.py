"""Native-supervised sample consistency without contact-inferred chemical roles."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_initialization import ligand_input
from tricoopdiff.anchored_training import prepare_anchored_training_sample


def fixture():
    ligand = ligand_input('CCCCCCCC')
    protein = torch.tensor([[[0., 2., 0.], [1., 3., 1.], [-1., 2., 1.]]])
    return {'coords': protein}, {'coords': protein + 6}, ligand


def prepare(e3, poi, ligand, **extra):
    return prepare_anchored_training_sample(
        e3, poi, ligand, fragments=(Chem.MolFromSmiles('CCC'), Chem.MolFromSmiles('CCC')),
        atom_maps=((0, 1, 2), (5, 6, 7)), generator=torch.Generator().manual_seed(7315),
        evidence_id='synthetic-training-fixture', **extra)


def test_training_sample_reaches_native_with_consistent_binary_conditions():
    e3, poi, ligand = fixture()
    sample = prepare(e3, poi, ligand)
    recovered_ligand, recovered_poi, recovered_body = sample.flow.decode(sample.data_angles)
    assert torch.allclose(recovered_ligand, ligand['coords'][0], atol=1e-4)
    assert torch.allclose(recovered_poi, poi['coords'][0], atol=1e-4)
    assert torch.allclose(recovered_body, torch.eye(3), atol=1e-4)
    assert sample.data_angles.abs().max() > .1, 'target should not always be q=0'
    conditions = sample.conditions
    assert torch.equal(conditions['ligand'][3:5, 3], torch.ones(2)), 'unmapped is unknown'
    assert torch.count_nonzero(conditions['ligand'][:, 2]) == 0, 'do not invent linker labels'
    for index, (data, mapped) in enumerate(((e3, [0, 1, 2]), (poi, [5, 6, 7])), start=1):
        expected = torch.cdist(ligand['coords'][0, mapped], data['coords'][0])
        assert torch.allclose(conditions[f'binary{index}'][0, mapped, :, 0], expected, atol=1e-5)
    repeated = prepare(e3, poi, ligand)
    assert torch.equal(repeated.prior_angles, sample.prior_angles)
    assert torch.equal(repeated.flow.ligand, sample.flow.ligand)


def test_training_features_and_chart_are_rigid_frame_invariant():
    e3, poi, ligand = fixture()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = torch.tensor([7., -3., 4.])

    def move(data):
        return {**data, 'coords': data['coords'] @ rotation.T + translation}

    first = prepare(e3, poi, ligand)
    second = prepare(move(e3), move(poi), move(ligand))
    for name in first.conditions:
        assert torch.allclose(first.conditions[name], second.conditions[name], atol=1e-5), name
    assert torch.allclose(first.flow.ligand @ rotation.T + translation,
                          second.flow.ligand, atol=1e-4)
    assert torch.allclose(first.data_angles, second.data_angles, atol=1e-4)


if __name__ == '__main__':
    test_training_sample_reaches_native_with_consistent_binary_conditions()
    test_training_features_and_chart_are_rigid_frame_invariant()
    print('Anchored training sample checks passed')
