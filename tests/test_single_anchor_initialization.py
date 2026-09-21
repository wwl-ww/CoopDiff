"""One-sided initialization consumes no ternary ligand conformer or second pose."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_initialization import ligand_input
from tricoopdiff.binary_pose_constraints import BinaryPoseSide
from tricoopdiff.single_anchor_initialization import initialize_single_anchor


def test_single_initializer_ignores_input_ligand_coordinates():
    data = ligand_input('CCCCCCCC')
    e3 = torch.tensor([[0., 2., 0.], [1., 3., 1.], [-1., 2., 1.]])
    poi = e3 + 6
    side = BinaryPoseSide(e3, data['coords'][0, :3], ((0, 1, 2),))
    first, audit = initialize_single_anchor(side, e3, poi, data, seed=7315, max_candidates=3)
    changed = {**data, 'coords': data['coords'] + 1000}
    second, _ = initialize_single_anchor(side, e3, poi, changed, seed=7315, max_candidates=3)
    assert torch.equal(first.strict.ligand, second.strict.ligand)
    assert torch.equal(first.strict.poi, poi)
    assert audit['native_information_used'] is False
    assert audit['selected_atom_maps'] == [[0, 1, 2], None]
    assert audit['conditioned_sides'] == [True, False]
    assert not any(first.plan.moves_poi)
    assert torch.allclose(first.strict.ligand[:3], data['coords'][0, :3], atol=1e-5)
