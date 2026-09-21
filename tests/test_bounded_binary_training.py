"""Perturbed templates must teach nonzero, reachable binary corrections."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_training import fixture, prepare
from tricoopdiff.bounded_binary_training import augment_binary_uncertainty


def test_perturbed_binary_training_has_nonzero_reachable_offsets():
    e3, poi, ligand = fixture()
    strict = prepare(e3, poi, ligand)
    bounds = {'translation': [.5, .5], 'rotation': [.2, .2]}
    def build():
        return augment_binary_uncertainty(
            strict, e3, poi, ligand,
            fragments=(Chem.MolFromSmiles('CCC'), Chem.MolFromSmiles('CCC')),
            atom_maps=((0, 1, 2), (5, 6, 7)),
            generator=torch.Generator().manual_seed(7315), evidence_id='test', bounds=bounds)
    sample = build()
    decoded = sample.flow.decode(sample.data_angles)
    assert torch.allclose(decoded[0], ligand['coords'][0], atol=1e-4)
    assert torch.allclose(decoded[1], poi['coords'][0], atol=1e-4)
    assert sample.data_angles[sample.flow.torsion_count:].abs().sum() > .1
    assert not torch.allclose(sample.conditions['binary1'], strict.conditions['binary1'])
    assert not torch.allclose(sample.conditions['binary2'], strict.conditions['binary2'])
    repeated = build()
    assert torch.equal(sample.prior_angles, repeated.prior_angles)
    assert torch.equal(sample.data_angles, repeated.data_angles)
    assert sample.audit['binary_bounds'] == bounds
