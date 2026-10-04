"""CPU tests for the training-only bounded joint rigid endpoint oracle."""

import math

import torch

from tricoopdiff.experiments.joint_endpoint_experiment import (
    joint_rigid_endpoint_targets,
    stable_so3_log,
)
from tricoopdiff.pose_anchors import apply_rigid, assemble_interface_pair
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.pose_anchors import rotation_exp as _rotation_exp


DTYPE = torch.float64


def _raises(message, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f'expected ValueError containing {message!r}')


def _random_rotation(generator):
    axis = torch.randn(3, dtype=DTYPE, generator=generator)
    axis /= axis.norm()
    angle = (torch.rand((), dtype=DTYPE, generator=generator) * 2 - 1) * 2.8
    return _rotation_exp(axis * angle)


def _fixture(seed=0):
    generator = torch.Generator().manual_seed(seed)
    native_ligand = torch.randn(1, 13, 3, dtype=DTYPE, generator=generator) * torch.tensor(
        [2., 3., 4.], dtype=DTYPE)
    native_poi = torch.randn(1, 19, 3, dtype=DTYPE, generator=generator) * torch.tensor(
        [5., 7., 9.], dtype=DTYPE)
    ra, rb = _random_rotation(generator)[None], _random_rotation(generator)[None]
    ta, tb = (torch.randn(1, 3, dtype=DTYPE, generator=generator) * 10 for _ in range(2))
    ligand = apply_rigid(native_ligand, ra, ta)
    poi, body = assemble_interface_pair(native_poi, ra, ta, rb, tb)
    data = {
        'coords': native_ligand,
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'edge_mask': torch.empty(0, dtype=torch.bool),
        'mask_rotate': torch.empty((0, native_ligand.shape[1]), dtype=torch.bool),
    }
    return ligand, poi, body, native_ligand, native_poi, data


def test_exact_one_step_recovers_joint_endpoint_across_random_poses():
    for seed in range(24):
        ligand, poi, body, native_ligand, native_poi, data = _fixture(seed)
        time = torch.tensor([0.17 + (seed % 7) * 0.1], dtype=DTYPE)
        target, diagnostic = joint_rigid_endpoint_targets(
            ligand, poi, body, native_ligand, native_poi, time)
        result = SDPIFlow.step(ligand, poi, body, target, time, data)
        assert torch.allclose(result[0], native_ligand, atol=2e-9, rtol=0)
        assert torch.allclose(result[1], native_poi, atol=2e-9, rtol=0)
        assert torch.allclose(result[2], torch.eye(3, dtype=DTYPE)[None], atol=2e-9, rtol=0)
        assert diagnostic['ligand_fit_rmsd'].max() < 1e-12
        assert diagnostic['poi_fit_rmsd'].max() < 1e-12


def test_target_is_equivariant_under_global_rigid_motion():
    ligand, poi, body, native_ligand, native_poi, _ = _fixture(31)
    generator = torch.Generator().manual_seed(91)
    q = _random_rotation(generator)[None]
    shift = torch.tensor([[80., -31., 12.]], dtype=DTYPE)
    original, _ = joint_rigid_endpoint_targets(
        ligand, poi, body, native_ligand, native_poi, torch.tensor([.6], dtype=DTYPE))
    moved, _ = joint_rigid_endpoint_targets(
        apply_rigid(ligand, q, shift), apply_rigid(poi, q, shift), q @ body @ q.transpose(-1, -2),
        apply_rigid(native_ligand, q, shift), apply_rigid(native_poi, q, shift),
        torch.tensor([.6], dtype=DTYPE))
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert torch.allclose(moved[branch][component],
                                  original[branch][component] @ q.transpose(-1, -2),
                                  atol=2e-9, rtol=0)


def test_local_step_matches_cartesian_finite_difference_tangent():
    ligand, poi, body, native_ligand, native_poi, data = _fixture(8)
    time = torch.tensor([.7], dtype=DTYPE)
    target, _ = joint_rigid_endpoint_targets(
        ligand, poi, body, native_ligand, native_poi, time)
    h = 1e-6
    plus = SDPIFlow.step(ligand, poi, body, target, torch.tensor([h], dtype=DTYPE), data)
    # Directly evaluate the same smooth one-parameter branch at negative h.
    center = ligand.mean(1)
    transforms = []
    for branch in ('e3_ligand', 'poi_ligand'):
        rotation = _rotation_exp(torch.tensor([h], dtype=DTYPE)[:, None] * target[branch]['rot'])
        destination = center + h * target[branch]['tr']
        translation = destination - (rotation @ center[..., None]).squeeze(-1)
        transforms.append((rotation, translation))
    minus_ligand = apply_rigid(ligand, *transforms[0])
    minus_poi, _ = assemble_interface_pair(poi, *transforms[0], *transforms[1])
    derivative = ((plus[0] - minus_ligand) / (2 * h), (plus[1] - minus_poi) / (2 * h))
    for step in (2e-4, 1e-4):
        actual = SDPIFlow.step(ligand, poi, body, target, torch.tensor([step], dtype=DTYPE), data)
        errors = [((value - start) / step - tangent).abs().max()
                  for value, start, tangent in zip(actual[:2], (ligand, poi), derivative)]
        if step == 2e-4:
            coarse = max(errors)
        else:
            assert max(errors) < coarse * .51


def test_low_time_is_rejected_or_velocity_is_bounded():
    fixture = _fixture(12)
    args = fixture[:5] + (torch.tensor([1e-5], dtype=DTYPE),)
    _raises('minimum', joint_rigid_endpoint_targets, *args, min_time=.1)
    bounded, diagnostic = joint_rigid_endpoint_targets(*args, min_time=.1, low_time='clamp')
    reference, _ = joint_rigid_endpoint_targets(
        *fixture[:5], torch.tensor([.1], dtype=DTYPE))
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert torch.allclose(bounded[branch][component], reference[branch][component])
    assert diagnostic['low_time_clamped'].item()
    assert diagnostic['effective_time'].item() == .1


def test_blend_and_per_component_correction_caps_are_explicit():
    fixture = _fixture(5)
    time = torch.tensor([.4], dtype=DTYPE)
    oracle, _ = joint_rigid_endpoint_targets(*fixture[:5], time)
    teacher = {branch: {component: torch.zeros_like(oracle[branch][component])
                        for component in ('tr', 'rot')}
               for branch in ('e3_ligand', 'poi_ligand')}
    bounded, diagnostic = joint_rigid_endpoint_targets(
        *fixture[:5], time, teacher_targets=teacher, blend=.25,
        correction_cap={'tr': .5, 'rot': .2})
    for branch in ('e3_ligand', 'poi_ligand'):
        assert bounded[branch]['tr'].norm(dim=-1).max() <= .125 + 1e-12
        assert bounded[branch]['rot'].norm(dim=-1).max() <= .05 + 1e-12
        for component in ('tr', 'rot'):
            key = f'correction_norm_{branch}_{component}'
            assert torch.allclose(diagnostic[key], oracle[branch][component].norm(dim=-1))
    assert diagnostic['bounded_target'].item()


def test_pi_branch_is_rejected_by_default_but_stable_log_is_finite():
    native_ligand = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                                   [0., 0., 3.]]], dtype=DTYPE)
    native_poi = native_ligand * 2 + 4
    rotation = _rotation_exp(torch.tensor([[math.pi, 0., 0.]], dtype=DTYPE))
    ligand = apply_rigid(native_ligand, rotation, torch.zeros(1, 3, dtype=DTYPE))
    body = torch.eye(3, dtype=DTYPE)[None]
    _raises('branch cut', joint_rigid_endpoint_targets,
            ligand, native_poi, body, native_ligand, native_poi,
            torch.tensor([.5], dtype=DTYPE))
    log = stable_so3_log(rotation, reject_near_pi=False)
    assert torch.isfinite(log).all()
    assert torch.allclose(log.norm(dim=-1), torch.tensor([math.pi], dtype=DTYPE), atol=1e-9)
    assert torch.allclose(_rotation_exp(log), rotation, atol=1e-9, rtol=0)


def test_shape_or_body_inconsistency_is_rejected_in_exact_mode():
    ligand, poi, body, native_ligand, native_poi, _ = _fixture(4)
    changed = native_ligand.clone()
    changed[:, 0] += torch.tensor([2., -1., 3.], dtype=DTYPE)
    _raises('rigid-equivalent', joint_rigid_endpoint_targets,
            ligand, poi, body, changed, native_poi, torch.tensor([.5], dtype=DTYPE))
    _raises('body frame', joint_rigid_endpoint_targets,
            ligand, poi, torch.eye(3, dtype=DTYPE)[None], native_ligand, native_poi,
            torch.tensor([.5], dtype=DTYPE))


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
