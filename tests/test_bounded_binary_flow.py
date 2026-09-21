"""Total uncertainty bounds, target reachability, and path/decoder parity."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_anchored_torsion import fixture
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow, ball_map
from tricoopdiff.pose_anchors import proper_rigid_fit


def bounded():
    return BoundedBinaryTorsionFlow(AnchoredTorsionFlow(*fixture()), (0, 1, 2), (5, 6, 7),
                                    translation_bounds=(.5, .8), rotation_bounds=(.15, .2))


def test_bounded_target_roundtrip_and_rejects_incompatible_geometry():
    flow = bounded()
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    ligand, poi, _ = flow.decode(state)
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    recovered, audit = flow.encode_target(ligand, poi, edges)
    assert torch.allclose(recovered, state, atol=1e-9)
    assert max(audit.values()) < 1e-9
    for bad_ligand, bad_poi in ((ligand + 10, poi + 10), (ligand, poi + 10)):
        try:
            flow.encode_target(bad_ligand, bad_poi, edges)
        except ValueError as error:
            assert 'bound' in str(error)
        else:
            raise AssertionError('unreachable target was accepted')
    invalid = ligand.clone()
    invalid[0] += .2
    try:
        flow.encode_target(invalid, poi, edges)
    except ValueError:
        pass
    else:
        raise AssertionError('rigid offsets cannot repair internal warhead geometry')


def test_bounded_offsets_are_total_not_per_step_and_preserve_chemistry():
    flow = bounded()
    state = flow.strict.ligand.new_zeros(flow.size)
    velocity = torch.ones_like(state) * 20
    for _ in range(100):
        state, (ligand, poi, _) = flow.step(state, velocity, .1)
    reference = flow.strict.ligand
    rotation, translation, _ = proper_rigid_fit(reference[:3], ligand[:3])
    center = reference[:3].mean(0)
    assert (rotation @ center + translation - center).norm() <= .5 + 1e-10
    assert torch.allclose((ligand[1:] - ligand[:-1]).norm(dim=-1),
                          (reference[1:] - reference[:-1]).norm(dim=-1), atol=1e-10)
    assert torch.allclose(torch.cdist(poi, poi), torch.cdist(flow.strict.poi, flow.strict.poi), atol=1e-10)
    _, strict_poi, _ = flow.strict.decode(state[:flow.torsion_count])
    # Undo the E3 offset, then measure the POI deviation at its binding center.
    unshifted = (poi - translation) @ rotation
    rp, tp, _ = proper_rigid_fit(strict_poi, unshifted)
    cp = flow.strict.decode(state[:flow.torsion_count])[0][5:].mean(0)
    assert (rp @ cp + tp - cp).norm() <= .8 + 1e-10
    for matrix, limit in ((rotation, .15), (rp, .2)):
        angle = torch.acos(((torch.trace(matrix) - 1) / 2).clamp(-1, 1))
        assert angle <= limit + 1e-10


def test_bounded_flow_path_and_cartesian_derivatives():
    flow = bounded()
    generator = torch.Generator().manual_seed(7315)
    target, prior = (flow.sample_prior(generator=generator) for _ in range(2))
    state, _, velocity = flow.path(target, prior, 1.)
    for _ in range(20):
        state, _ = flow.step(state, velocity, .05)
    assert torch.allclose(state, target, atol=1e-10)
    state, _, velocity = flow.path(target, prior, .4)
    def coords(q):
        return torch.cat(flow.decode(q)[:2])
    _, tangent = torch.autograd.functional.jvp(coords, state, velocity)
    eps = 1e-6
    finite = (coords(state + eps * velocity) - coords(state - eps * velocity)) / (2 * eps)
    assert torch.allclose(tangent, finite, atol=1e-7)
    latent = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    ball_map(latent, .5).sum().backward()
    assert torch.allclose(latent.grad, torch.ones(3, dtype=torch.float64) * .5)


def test_bounded_flow_frame_equivariance_zero_bounds_and_extreme_latents():
    flow = bounded()
    strict = flow.strict
    rotation = strict.ligand.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = strict.ligand.new_tensor([100., -30., 50.])
    moved = BoundedBinaryTorsionFlow(AnchoredTorsionFlow(
        strict.ligand @ rotation.T + shift, strict.poi @ rotation.T + shift,
        rotation @ strict.body, strict.plan), flow.end1, flow.end2,
        translation_bounds=flow.translation_bounds, rotation_bounds=flow.rotation_bounds)
    state = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
    transformed = torch.cat((state[:flow.torsion_count],
                             (state[flow.torsion_count:].reshape(4, 3) @ rotation.T).flatten()))
    first, second = flow.decode(state), moved.decode(transformed)
    for a, b in zip(first[:2], second[:2]):
        assert torch.allclose(b, a @ rotation.T + shift, atol=1e-10)
    assert torch.allclose(second[2], rotation @ first[2], atol=1e-10)
    exact = BoundedBinaryTorsionFlow(strict, flow.end1, flow.end2,
                                    translation_bounds=(0., 0.), rotation_bounds=(0., 0.))
    for a, b in zip(exact.decode(state), strict.decode(state[:flow.torsion_count])):
        assert torch.allclose(a, b, atol=1e-10)
    extreme = ball_map(strict.ligand.new_tensor([1e200, -1e200, 1e200]), .5)
    assert torch.isfinite(extreme).all() and extreme.norm() <= .5 + 1e-12
