"""Explicit local torsions and coupled geometry must obey graph/frame symmetries."""

from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.linker_action_geometry import linker_geometry_features


def test_linker_features_obey_rigid_and_atom_permutation_invariance():
    ligand, _, _, plan = fixture()
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    local, pairs = linker_geometry_features(ligand, plan, edges, 5.)
    rotation = ligand.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed = linker_geometry_features(ligand @ rotation.T + 11., plan, edges, 5.)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip((local, pairs), transformed))
    permutation = torch.tensor([4, 1, 7, 0, 6, 2, 5, 3])
    inverse = torch.argsort(permutation)
    renamed = replace(plan, bonds=tuple(tuple(int(inverse[i]) for i in bond) for bond in plan.bonds),
                      moving_atoms=tuple(tuple(int(inverse[i]) for i in group) for group in plan.moving_atoms))
    other = linker_geometry_features(ligand[permutation], renamed, inverse[edges], 5.)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip((local, pairs), other))


def test_features_read_torsion_not_chart_origin():
    flow = AnchoredTorsionFlow(*fixture())
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    zero = flow.ligand.new_zeros(3)
    state = zero + .4
    ligand, poi, body = flow.decode(state)
    initial, _ = linker_geometry_features(flow.ligand, flow.plan, edges, 5.)
    current, _ = linker_geometry_features(ligand, flow.plan, edges, 5.)
    assert not torch.allclose(initial[:, :6], current[:, :6]), 'torsion features only read bond length'
    rebased = AnchoredTorsionFlow(ligand, poi, body, flow.plan)
    same, _ = linker_geometry_features(rebased.decode(zero)[0], rebased.plan, edges, 5.)
    assert torch.allclose(current, same), 'features leak the arbitrary reference chart origin'
