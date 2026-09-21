"""Path/integrator parity, Cartesian targets, and consensus gradient routing."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_flow import AnchoredTorsionFlow, wrap_angles


def test_bond_audit_rejects_geometry_a_rigid_chart_cannot_repair():
    from rdkit import Chem
    from tricoopdiff.anchored_flow import audit_bond_geometry

    molecule = Chem.MolFromSmiles('CCC')
    ligand = torch.tensor([[0., 0., 0.], [1.52, 0., 0.], [2.28, 1.316, 0.]])
    assert audit_bond_geometry(molecule, ligand)['bonds_checked'] == 2
    ligand[2] += 5
    try:
        audit_bond_geometry(molecule, ligand)
    except ValueError as error:
        assert 'distorted covalent bonds' in str(error)
    else:
        raise AssertionError('stretched linker must not be frozen into a chart')


def test_path_reverses_to_endpoint_across_periodic_boundary():
    flow = AnchoredTorsionFlow(*fixture())
    data = torch.tensor([3., -.4, -3.], dtype=torch.float64)
    prior = torch.tensor([-3., .8, 3.], dtype=torch.float64)
    q, state, target = flow.path(data, prior, 1.)
    for index in range(20):
        q, state = flow.step(q, target, .05)
        expected_q, expected_state, _ = flow.path(data, prior, 1 - (index + 1) / 20)
        assert torch.allclose(wrap_angles(q - expected_q), torch.zeros_like(q), atol=1e-12)
        for actual, expected in zip(state, expected_state):
            assert torch.allclose(actual, expected, atol=1e-10)
        assert torch.equal(state[0][:3], flow.ligand[:3])
        assert torch.allclose(torch.cdist(state[0][5:], state[1]),
                              torch.cdist(flow.ligand[5:], flow.poi), atol=1e-10)


def test_cartesian_target_matches_time_difference_and_trains_allowed_outputs():
    flow = AnchoredTorsionFlow(*fixture())
    data = flow.ligand.new_zeros(3)
    prior = flow.sample_prior(generator=torch.Generator().manual_seed(73))
    q, _, target = flow.path(data, prior, .4)
    tangent = flow.cartesian_velocity(q, target)
    plus = flow.path(data, prior, .400001)[1]
    minus = flow.path(data, prior, .399999)[1]
    for value, left, right in zip(tangent, plus, minus):
        assert torch.allclose(value, (left - right) / .000002, atol=1e-7, rtol=1e-5)
    output = torch.zeros(1, 7, dtype=torch.float64, requires_grad=True)
    predicted = flow.consensus_velocity(output)
    cartesian = flow.cartesian_velocity(q, predicted, create_graph=True)
    loss = (predicted - target).square().mean()
    loss = loss + sum((p - t).square().mean() for p, t in zip(cartesian, tangent))
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert (output.grad[0, [2, 3, 4]].abs() > 0).all()
    assert torch.count_nonzero(output.grad[0, [0, 1, 5, 6]]) == 0


def test_target_encoding_recovers_angles_and_rejects_incompatible_geometry():
    flow = AnchoredTorsionFlow(*fixture())
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    angles = flow.ligand.new_tensor([2.8, -2.9, .7])
    ligand, poi, _ = flow.decode(angles)
    recovered, audit = flow.encode_target(ligand, poi, edges)
    assert torch.allclose(wrap_angles(recovered - angles), torch.zeros_like(angles), atol=1e-10)
    assert max(audit.values()) < 1e-10
    bad_ligand = ligand.clone()
    bad_ligand[0, 0] += .1
    for left, right in ((bad_ligand, poi), (ligand, poi + .1)):
        try:
            flow.encode_target(left, right, edges)
        except ValueError as error:
            assert 'outside anchored chart' in str(error)
        else:
            raise AssertionError('incompatible binary geometry accepted as a target')


def test_float32_geometry_uses_high_precision_internal_rotations():
    ligand, poi, body, plan = fixture()
    ligand, poi, body = ligand.float(), (poi + 1000).float(), body.float()
    single = AnchoredTorsionFlow(ligand, poi, body, plan)
    double = AnchoredTorsionFlow(ligand.double(), poi.double(), body.double(), plan)
    angles = torch.tensor([2.7, -2.9, 2.8], requires_grad=True)
    actual = single.decode(angles)
    expected = double.decode(angles.double())
    for left, right in zip(actual, expected):
        assert left.dtype == torch.float32
        assert torch.equal(left, right.float()), 'rotation drift was accumulated in float32'
    sum(value.square().mean() for value in actual).backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


if __name__ == '__main__':
    test_float32_geometry_uses_high_precision_internal_rotations()
    test_bond_audit_rejects_geometry_a_rigid_chart_cannot_repair()
    test_path_reverses_to_endpoint_across_periodic_boundary()
    test_cartesian_target_matches_time_difference_and_trains_allowed_outputs()
    test_target_encoding_recovers_angles_and_rejects_incompatible_geometry()
    print('Anchored flow checks passed')
