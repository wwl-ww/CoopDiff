"""Missing-side task has a free pose target, not a dropped feature on a dual path."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_training import fixture
from tricoopdiff.single_anchor_training import prepare_single_anchor_sample


def test_single_training_target_and_missing_side_features_are_consistent():
    e3, poi, ligand = fixture()
    sample = prepare_single_anchor_sample(e3, poi, ligand,
        generator=torch.Generator().manual_seed(7315), evidence_id='single-test')
    assert sample is not None
    state = sample.flow.decode(sample.data_angles)
    assert torch.allclose(state[0], ligand['coords'][0], atol=1e-4)
    assert torch.allclose(state[1], poi['coords'][0], atol=1e-4)
    assert torch.allclose(state[2], torch.eye(3), atol=1e-4)
    assert sample.conditions['sides'][0, 2] > 0
    assert sample.conditions['sides'][1, 2] == 0
    assert torch.count_nonzero(sample.conditions['binary2']) == 0
    assert not any(sample.flow.plan.moves_poi)
    assert sample.data_angles[-3:].norm() > .01, 'free rotation target was always identity'
    assert sample.audit['selected_atom_maps'][1] is None
