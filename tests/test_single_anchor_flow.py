"""One known side must not invent a constraint on the unknown receptor."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_initialization import ligand_input
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.anchored_torsion import compile_anchored_torsions
from tricoopdiff.single_anchor_flow import SingleAnchorTorsionFlow, compile_single_anchor_torsions


def fixture(bounded=False):
    ligand = ligand_input('CCCCCCCC')
    coords = ligand['coords'][0].double()
    poi = torch.tensor([[0., 2., 0.], [1., 3., 1.], [-1., 2., 1.]], dtype=torch.float64) + 6
    plan = compile_single_anchor_torsions(ligand['mol'], ligand['edge_index'], ligand['edge_mask'], (0, 1, 2))
    strict = AnchoredTorsionFlow(coords, poi, torch.eye(3, dtype=coords.dtype), plan)
    return SingleAnchorTorsionFlow(strict, (0, 1, 2), coords.new_zeros(3), 10.,
                                   translation_bound=.5 if bounded else 0.,
                                   rotation_bound=.2 if bounded else 0.), ligand


def test_single_anchor_keeps_unknown_side_free_and_more_torsions():
    flow, ligand = fixture()
    dual = compile_anchored_torsions(8, ligand['edge_index'], ligand['edge_mask'],
                                     (0, 1, 2), (5, 6, 7), molecule=ligand['mol'])
    assert len(flow.plan.bonds) > len(dual.bonds)
    assert not any(flow.plan.moves_poi)
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    first = flow.decode(state)
    assert torch.equal(first[0][:3], flow.strict.ligand[:3])
    moved_state = state.clone()
    moved_state[flow.torsion_count + 6:flow.torsion_count + 9] += 10
    second = flow.decode(moved_state)
    assert torch.equal(first[0], second[0])
    assert torch.allclose(second[1] - first[1], torch.full_like(first[1], 100.))
    torsion_state = state.clone()
    torsion_state[0] += .7
    assert torch.equal(flow.decode(torsion_state)[1], first[1]), 'unknown POI followed a fictitious warhead'


def test_single_anchor_target_encoding_paths_and_tangents():
    for bounded in (False, True):
        flow, ligand = fixture(bounded)
        generator = torch.Generator().manual_seed(7315)
        target, prior = (flow.sample_prior(generator=generator) for _ in range(2))
        target_ligand, target_poi, _ = flow.decode(target)
        recovered, audit = flow.encode_target(target_ligand, target_poi, ligand['edge_index'])
        assert torch.allclose(recovered, target, atol=1e-9)
        assert max(audit.values()) < 1e-9
        state, _, velocity = flow.path(target, prior, 1.)
        for _ in range(20):
            state, _ = flow.step(state, velocity, .05)
        assert torch.allclose(state, target, atol=1e-10)
        state, _, velocity = flow.path(target, prior, .4)
        vl, vp = flow.cartesian_velocity(state, velocity)
        plus, minus = flow.decode(state + velocity * 1e-6), flow.decode(state - velocity * 1e-6)
        assert torch.allclose(vl, (plus[0] - minus[0]) / 2e-6, atol=1e-7)
        assert torch.allclose(vp, (plus[1] - minus[1]) / 2e-6, atol=1e-7)
