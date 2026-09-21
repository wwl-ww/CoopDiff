"""Geometry invariants and derivatives of the experimental anchored update."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.anchored_torsion import compile_anchored_torsions, anchored_torsion_step


def fixture():
    torch.manual_seed(19)
    ligand = torch.randn(8, 3, dtype=torch.float64)
    poi = torch.randn(5, 3, dtype=torch.float64) + 4
    # Deliberately alternate stored orientations, including anchor-side masks.
    pairs = [(i, i + 1) if i % 2 else (i + 1, i) for i in range(7)]
    edges = torch.tensor(pairs, dtype=torch.long).T
    plan = compile_anchored_torsions(8, edges, torch.ones(7, dtype=torch.bool),
                                    (0, 1, 2), (5, 6, 7))
    return ligand, poi, torch.eye(3, dtype=torch.float64), plan


def test_binary_geometry_and_bond_lengths_survive_multiple_updates():
    ligand, poi, body, plan = fixture()
    assert plan.original_indices == (2, 3, 4)
    assert all(plan.moves_poi)
    assert len(plan.rejected) == 4
    moved, receptor, frame = anchored_torsion_step(
        ligand, poi, body, torch.tensor([.7, -.4, .8], dtype=ligand.dtype), plan)
    assert torch.equal(moved[:3], ligand[:3]), 'anchored warhead moved'
    assert torch.allclose(torch.cdist(moved[5:], receptor), torch.cdist(ligand[5:], poi))
    assert torch.allclose((moved[1:] - moved[:-1]).norm(dim=-1),
                          (ligand[1:] - ligand[:-1]).norm(dim=-1))
    assert torch.allclose((receptor - receptor.mean(0)),
                          (poi - poi.mean(0)) @ frame.T)
    assert not torch.allclose(receptor, poi), 'POI did not follow the moving end'


def test_equivariance_and_cartesian_derivatives():
    ligand, poi, body, plan = fixture()
    angles = torch.tensor([.3, -.5, .2], dtype=ligand.dtype, requires_grad=True)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=ligand.dtype)
    shift = torch.tensor([10., -2., 5.], dtype=ligand.dtype)
    first = anchored_torsion_step(ligand, poi, body, angles, plan)
    second = anchored_torsion_step(ligand @ rotation.T + shift,
                                  poi @ rotation.T + shift, rotation @ body, angles, plan)
    for original, transformed in zip(first[:2], second[:2]):
        assert torch.allclose(transformed, original @ rotation.T + shift)
    assert torch.allclose(second[2], rotation @ first[2])

    def decode(value):
        left, right, _ = anchored_torsion_step(ligand, poi, body, value, plan)
        return torch.cat((left, right))

    assert torch.autograd.gradcheck(decode, (angles,), atol=1e-5, rtol=1e-4)
    jacobian = torch.autograd.functional.jacobian(decode, angles)
    assert jacobian[8:].abs().sum() > 0, 'POI motion missing from derivatives'


def test_ring_is_rejected_and_disconnected_input_fails():
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    plan = compile_anchored_torsions(6, edges, torch.ones(6, dtype=torch.bool),
                                    (0, 1), (3, 4))
    assert not plan.bonds and all(reason == 'ring_closure' for _, reason in plan.rejected)
    try:
        compile_anchored_torsions(7, edges, torch.ones(6, dtype=torch.bool), (0, 1), (3, 4))
    except ValueError as error:
        assert 'connected' in str(error)
    else:
        raise AssertionError('disconnected ligand accepted')


if __name__ == '__main__':
    test_binary_geometry_and_bond_lengths_survive_multiple_updates()
    test_equivariance_and_cartesian_derivatives()
    test_ring_is_rejected_and_disconnected_input_fails()
    print('Anchored torsion checks passed')
