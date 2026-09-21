"""Unknown-side freedom and exact interface derivative checks."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_single_anchor_flow import fixture
from test_anchored_joint_motion import observed, proposals
from tricoopdiff.single_anchor_joint_motion import single_interface_jacobian, single_joint_velocity


def test_single_joint_jacobian_matches_all_directions():
    for bounded in (False, True):
        flow, _ = fixture(bounded)
        state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
        scales = state.new_tensor([4., 7.])
        matrix = single_interface_jacobian(flow, state, scales)
        ligand, poi, _ = flow.decode(state)
        for index in range(flow.size):
            velocity = torch.eye(flow.size, dtype=state.dtype)[index]
            if index < flow.torsion_count:
                velocity *= math.pi
            vl, vp = flow.cartesian_velocity(state, velocity)
            assert torch.allclose(matrix[:, index], observed(ligand, poi, vl, vp, scales), atol=1e-9), index


def test_single_joint_translates_unknown_poi_without_disturbing_known_side():
    flow, _ = fixture()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    rigid = state.new_zeros(12, requires_grad=True)
    with torch.no_grad():
        rigid[6] = 1
    torsion = state.new_zeros(flow.torsion_count, requires_grad=True)
    velocity, _ = single_joint_velocity(flow, state, proposals(flow.strict, rigid, torsion),
                                        state.new_tensor([4., 7.]))
    assert velocity[flow.torsion_count + 6:flow.torsion_count + 9].abs().sum() > .1
    assert velocity[flow.torsion_count:flow.torsion_count + 6].abs().max() < 1e-10
    velocity.square().sum().backward()
    assert torch.isfinite(rigid.grad).all() and rigid.grad.abs().sum() > 0
    assert torch.isfinite(torsion.grad).all()
    _, (ligand, poi, _) = flow.step(state, velocity.detach(), .1)
    assert torch.equal(ligand[:3], flow.strict.ligand[:3])
