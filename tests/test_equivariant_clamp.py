"""Saturating scalar/vector/tensor channels must not pick a global axis."""

import sys
from pathlib import Path

import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.geometry.equivariant_clamp import clamp_irrep_features, validate_clamp_mode


def test_saturated_irreps_remain_rotation_equivariant():
    torch.manual_seed(317)
    irreps = o3.Irreps('2x0e + 2x1o + 1x1e + 2x2e')
    features = torch.randn(7, irreps.dim, dtype=torch.float64) * 300
    rotation = o3.rand_matrix(dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)
    expected = clamp_irrep_features(features, irreps, 100) @ representation.T
    actual = clamp_irrep_features(features @ representation.T, irreps, 100)
    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-8)
    legacy_error = (features.clamp(-100, 100) @ representation.T
                    - (features @ representation.T).clamp(-100, 100)).abs().max()
    assert legacy_error > 1, 'fixture must actually exercise the old equivariance failure'


def test_norm_bounds_finite_gradients_and_large_inputs():
    irreps = o3.Irreps('1x0e + 2x1o')
    features = torch.tensor([[200., 300., 400., 0., 1., 2., 3.]], requires_grad=True)
    result = clamp_irrep_features(features, irreps, 100)
    assert torch.equal(result[:, -3:], features[:, -3:])
    assert torch.allclose(result[:, 1:4], torch.tensor([[60., 80., 0.]]))
    result.sum().backward()
    assert torch.isfinite(features.grad).all()
    large = clamp_irrep_features(torch.full((2, 3), 1e30), '1x1o', 100)
    assert torch.isfinite(large).all()
    assert torch.allclose(large.norm(dim=-1), torch.full((2,), 100.))


def test_unknown_modes_are_rejected():
    assert validate_clamp_mode('componentwise') == 'componentwise'
    assert validate_clamp_mode('irrep_norm') == 'irrep_norm'
    try:
        validate_clamp_mode('typo')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown mode was silently accepted')


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')
