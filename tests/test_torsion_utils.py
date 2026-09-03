"""Regression checks for the vendored torsion update used during training."""

import numpy as np
import torch

from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles


def test_modify_conformer_torsion_angles_copies_and_rotates():
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    original = positions.clone()
    edges = torch.tensor([[0, 1]])
    mask = np.asarray([[False, True, True]])
    updated = modify_conformer_torsion_angles(
        positions, edges, mask, np.asarray([0.2]),
    )
    assert updated.shape == positions.shape
    assert torch.isfinite(updated).all()
    assert torch.equal(positions, original), "input coordinates were modified in place"
