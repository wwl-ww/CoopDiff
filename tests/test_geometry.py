"""Geometry regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import warnings

import numpy as np
import torch

from e3nn import o3
from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles
from tricoopdiff.diffdock_compat import _knn_graph_pytorch, _radius_graph_pytorch, _radius_pytorch
from tricoopdiff.geometry.equivariant_clamp import clamp_irrep_features, validate_clamp_mode
from tricoopdiff.geometry.rigid import rotation_exp
from tricoopdiff.legacy.trifuse import BondContextEncoder


# Torsion utils

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


def torsion_utils_chemical_context_fixture():
    generator = torch.Generator().manual_seed(73)
    coords = torch.tensor([[0., 1., .2], [0., 0., 0.], [1., 0., 0.],
                           [1., .2, 1.], [2., 1., 1.], [1., -1., .3]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 2, 5],
                          [1, 0, 2, 1, 3, 2, 4, 3, 5, 2]])
    selected = torch.zeros(10, dtype=torch.bool)
    selected[2] = True
    rotate = torch.tensor([[False, False, True, True, True, True]])
    attributes = torch.zeros(10, 4, dtype=torch.float64)
    attributes[:, 0] = 1.
    tokens = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    context = torch.randn(1, 8, generator=generator, dtype=torch.float64)
    encoder = BondContextEncoder(8, chemical_context=True).double().eval()
    return encoder, tokens, coords, edges, selected, rotate, context, attributes


def test_chemical_bond_context_is_se3_invariant_and_atom_permutation_independent():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    expected = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    rotation = rotation_exp(coords.new_tensor([.4, -.7, .2]))
    moved = coords @ rotation.T + coords.new_tensor([13., -7., 2.])
    actual = encoder(tokens, moved, edges, selected, rotate, context, attributes)
    assert expected.shape == (1, 1, 8)
    assert torch.allclose(expected, actual, atol=1e-11, rtol=0.)
    permutation = torch.tensor([3, 5, 0, 4, 1, 2])
    inverse = permutation.argsort()
    actual = encoder(tokens[permutation], coords[permutation], inverse[edges], selected,
                     rotate[:, permutation], context, attributes)
    assert torch.allclose(expected, actual, atol=1e-11, rtol=0.), (
        'torsion context depends on which adjacent atom has the smallest index')


def test_chemical_bond_context_reads_distant_fragment_content_and_bond_type():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    expected = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    changed = tokens.clone()
    changed[4] += torch.arange(8, dtype=tokens.dtype)
    fragment_result = encoder(changed, coords, edges, selected, rotate, context, attributes)
    changed_attributes = attributes.clone()
    changed_attributes[selected] = attributes.new_tensor([0., 1., 0., 0.])
    bond_result = encoder(tokens, coords, edges, selected, rotate, context, changed_attributes)
    assert not torch.allclose(expected, fragment_result, atol=1e-7, rtol=0.)
    assert not torch.allclose(expected, bond_result, atol=1e-7, rtol=0.)


def test_chemical_torsion_context_has_finite_geometry_and_fragment_gradients():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    tokens.requires_grad_()
    coords.requires_grad_()
    output = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    output.square().sum().backward()
    assert torch.isfinite(tokens.grad).all() and tokens.grad[4].norm() > 0
    assert torch.isfinite(coords.grad).all() and coords.grad.norm() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in encoder.parameters())
    # Collinear neighboring atoms have no defined phase, not a NaN dihedral.
    collinear = torch.arange(6, dtype=torch.float64)[:, None].expand(-1, 3).clone().requires_grad_()
    encoder(tokens.detach(), collinear, edges, selected, rotate, context, attributes).sum().backward()
    assert torch.isfinite(collinear.grad).all()


def test_chemical_torsion_phase_retains_signed_handedness():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    descriptors = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: descriptors.append(args[0]))
    try:
        encoder(tokens, coords, edges, selected, rotate, context, attributes)
        encoder(tokens, coords * coords.new_tensor([1., 1., -1.]),
                edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    phase_start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    phase = descriptors[0][:, phase_start:phase_start + 3]
    reflected = descriptors[1][:, phase_start:phase_start + 3]
    assert phase[0, 0].abs() > .01 and 0 < phase[0, 2] <= 1
    assert torch.allclose(reflected[:, 0], -phase[:, 0], atol=1e-12, rtol=0.)
    assert torch.allclose(reflected[:, 1:], phase[:, 1:], atol=1e-12, rtol=0.)


def test_chemical_torsion_masks_and_empty_torsion_keep_explicit_contracts():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    empty = encoder(tokens, coords, edges, torch.zeros_like(selected), None, context)
    assert empty.shape == (1, 0, 8) and torch.isfinite(empty).all()
    broken = rotate.clone()
    broken[0, 4] = False
    for mask, attr in ((rotate.float(), attributes), (broken, attributes),
                       (None, attributes), (rotate, None)):
        try:
            encoder(tokens, coords, edges, selected, mask, context, attr)
        except ValueError:
            pass
        else:
            raise AssertionError('chemical torsion encoder accepted invalid chemistry/fragment data')
    legacy = BondContextEncoder(8).double()
    assert legacy.net[1].in_features == 8 * 4 + 16 + 2
    assert legacy(tokens, coords, edges, selected, rotate, context).shape == (1, 1, 8)


def test_trifuse_graph_preserves_bond_attributes_for_chemical_context():
    from tricoopdiff.inputs.graphs import build_trifuse_graph

    _, tokens, coords, edges, selected, rotate, _, attributes = torsion_utils_chemical_context_fixture()
    graph = build_trifuse_graph(coords[:2], tokens[:2], coords[:3], tokens[:3],
                                coords, tokens, edges, selected, rotate,
                                lig_edge_attr=attributes)
    assert graph['lig_edge_attr'] is attributes


def torsion_utils_symmetric_phase_fixture(neighbor_count, phase_harmonics=3, chemically_symmetric=True):
    angles = torch.arange(neighbor_count, dtype=torch.float64) * (2 * np.pi / neighbor_count)
    ring = torch.stack((torch.zeros_like(angles), angles.cos(), angles.sin()), -1)
    coords = torch.cat((torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64),
                        ring, ring + ring.new_tensor([1., 0., 0.])))
    undirected = [(0, 1)] + [(0, index + 2) for index in range(neighbor_count)]
    undirected += [(1, index + 2 + neighbor_count) for index in range(neighbor_count)]
    edges = torch.tensor([pair for bond in undirected for pair in (bond, bond[::-1])]).T
    selected = torch.zeros(edges.shape[1], dtype=torch.bool)
    selected[0] = True
    rotate = torch.zeros(1, len(coords), dtype=torch.bool)
    rotate[0, 1] = True
    rotate[0, 2 + neighbor_count:] = True
    attributes = torch.zeros(edges.shape[1], 4, dtype=torch.float64)
    attributes[:, 0] = 1.
    generator = torch.Generator().manual_seed(752)
    tokens = torch.randn(len(coords), 8, generator=generator, dtype=torch.float64)
    if chemically_symmetric:
        tokens[2:2 + neighbor_count] = tokens[2].clone()
        tokens[2 + neighbor_count:] = tokens[2 + neighbor_count].clone()
    context = torch.randn(1, 8, generator=generator, dtype=torch.float64)
    encoder = BondContextEncoder(8, chemical_context=True, phase_harmonics=phase_harmonics).double().eval()
    return encoder, tokens, coords, edges, selected, rotate, context, attributes


def torsion_utils_rotate_phase_fragment(coords, rotate, angle):
    result = coords.clone()
    rotation = rotation_exp(coords.new_tensor([angle, 0., 0.]))
    result[rotate[0]] = (coords[rotate[0]] - coords[1]) @ rotation.T + coords[1]
    return result


def test_torsion_fourier_resolves_twofold_symmetry_hidden_by_mean_directions():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(2)
    rotated = torsion_utils_rotate_phase_fragment(coords, rotate, np.pi / 2)
    base = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    actual = encoder(tokens, rotated, edges, selected, rotate, context, attributes)
    assert not torch.allclose(base, actual, atol=1e-7, rtol=0.), (
        'opposing neighbors erased a distinguishable 90-degree torsion change')
    legacy = BondContextEncoder(8, chemical_context=True, phase_harmonics=1).double().eval()
    legacy_base = legacy(tokens, coords, edges, selected, rotate, context, attributes)
    legacy_rotated = legacy(tokens, rotated, edges, selected, rotate, context, attributes)
    assert torch.allclose(legacy_base, legacy_rotated, atol=1e-12, rtol=0.), (
        'legacy v2 inference must retain its original mean-direction computation')
    assert legacy.net[1].in_features == 95 and encoder.net[1].in_features == 99


def test_torsion_fourier_retains_threefold_symmetry_but_resolves_half_period():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(3)
    base = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    periodic = encoder(tokens, torsion_utils_rotate_phase_fragment(coords, rotate, 2 * np.pi / 3),
                       edges, selected, rotate, context, attributes)
    distinct = encoder(tokens, torsion_utils_rotate_phase_fragment(coords, rotate, np.pi / 3),
                       edges, selected, rotate, context, attributes)
    assert torch.allclose(base, periodic, atol=1e-12, rtol=0.)
    assert not torch.allclose(base, distinct, atol=1e-7, rtol=0.), (
        'threefold neighboring directions erased a distinguishable 60-degree torsion change')


def test_torsion_fourier_zero_and_near_collinear_phase_has_no_fake_harmonics_or_exploding_gradient():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(2)
    phase_start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    for transverse_scale in (0., 1e-7):
        positions = coords.clone()
        positions[:, 1:] *= transverse_scale
        positions.requires_grad_()
        descriptors = []
        hook = encoder.net.register_forward_pre_hook(lambda module, args: descriptors.append(args[0]))
        try:
            output = encoder(tokens, positions, edges, selected, rotate, context, attributes)
        finally:
            hook.remove()
        phase = descriptors[0][:, phase_start:phase_start + 7]
        if transverse_scale == 0.:
            assert torch.equal(phase, torch.zeros_like(phase)), 'empty/undefined phase invented cos(2phi)=-1'
        (output.sum() + phase.sum()).backward()
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().max() < 100., 'near-collinear phase normalization amplified gradients'


def test_torsion_fourier_higher_orders_keep_signed_reflection_and_legacy_first_order():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    descriptors = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: descriptors.append(args[0]))
    try:
        encoder(tokens, coords, edges, selected, rotate, context, attributes)
        encoder(tokens, coords * coords.new_tensor([1., 1., -1.]),
                edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    original, reflected = [descriptor[:, start:start + 7] for descriptor in descriptors]
    sine_columns, cosine_columns = [0, 3, 5], [1, 4, 6]
    assert torch.allclose(original[:, sine_columns], -reflected[:, sine_columns], atol=1e-12, rtol=0.)
    assert torch.allclose(original[:, cosine_columns], reflected[:, cosine_columns], atol=1e-12, rtol=0.)
    assert torch.equal(original[:, 2], reflected[:, 2])


def test_torsion_fourier_phase_and_derivative_are_continuous_across_old_collinear_cutoff():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(2)
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    descriptors, derivatives = [], []
    for scale in (.99e-8, 1.e-8, 1.01e-8):
        amount = coords.new_tensor(scale, requires_grad=True)
        positions = torch.cat((coords[:, :1], coords[:, 1:] * amount), -1)
        captured = []
        hook = encoder.net.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            encoder(tokens, positions, edges, selected, rotate, context, attributes)
        finally:
            hook.remove()
        phase = captured[0][:, start:start + 7]
        derivative, = torch.autograd.grad(phase.sum(), amount)
        descriptors.append(phase)
        derivatives.append(derivative)
        assert torch.isfinite(phase).all() and torch.isfinite(derivative)
        assert bool(((phase[:, 2] >= 0) & (phase[:, 2] <= 1)).all())
    assert (descriptors[0] - descriptors[2]).abs().max() < 1e-20
    assert (derivatives[0] - derivatives[2]).abs().max() < 1e-10


def test_torsion_fourier_sine_has_nonzero_angle_gradient_at_symmetric_zero_rotation():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(2)
    angle = coords.new_tensor(0., requires_grad=True)
    rotation = rotation_exp(torch.stack((angle, angle * 0, angle * 0)))
    positions = torch.where(rotate[0, :, None], (coords - coords[1]) @ rotation.T + coords[1], coords)
    captured = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        encoder(tokens, positions, edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    sine_second_harmonic = captured[0][:, start + 3]
    derivative, = torch.autograd.grad(sine_second_harmonic.sum(), angle)
    assert torch.isfinite(derivative) and derivative.abs() > 1., (
        'the Fourier repair must offer a trainable signed direction, not only distinguish endpoints')


def test_torsion_fourier_binds_neighbor_chemistry_to_phase_and_resolves_half_turn():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_symmetric_phase_fixture(
        2, chemically_symmetric=False)
    tokens.requires_grad_()
    captured = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        original = encoder(tokens, coords, edges, selected, rotate, context, attributes)
        rotated = encoder(tokens, torsion_utils_rotate_phase_fragment(coords, rotate, np.pi),
                           edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    assert not torch.allclose(original, rotated, atol=1e-7, rtol=0.), (
        'opposing but chemically different neighbors were treated as a true 180-degree symmetry')
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    first_phase = captured[0][:, start:start + 2]
    opposite_phase = captured[1][:, start:start + 2]
    assert torch.allclose(first_phase, -opposite_phase, atol=1e-12, rtol=0.)
    (first_phase - opposite_phase).square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert bool((tokens.grad[2:].norm(dim=-1) > 1e-8).all()), (
        'chemical attention must propagate phase supervision to both neighboring fragments')


def test_torsion_chemical_phase_has_safe_zero_attention_when_endpoint_has_no_other_neighbor():
    encoder, tokens, coords, edges, selected, _, context, attributes = torsion_utils_chemical_context_fixture()
    selected = torch.zeros_like(selected)
    selected[6] = True
    rotate = torch.zeros(1, len(coords), dtype=torch.bool)
    rotate[0, 4] = True
    tokens.requires_grad_()
    coords.requires_grad_()
    captured = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        output = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    phase = captured[0][:, start:start + 7]
    assert torch.equal(phase, torch.zeros_like(phase))
    output.square().sum().backward()
    assert torch.isfinite(tokens.grad).all() and torch.isfinite(coords.grad).all()


def test_torsion_phase_attention_preserves_valid_neighbors_with_extreme_negative_affinity():
    encoder, tokens, coords, edges, selected, rotate, context, attributes = torsion_utils_chemical_context_fixture()
    # Every valid neighboring score is below -1e4, while padded/axis atoms
    # would otherwise dominate an arbitrarily finite masking sentinel.
    tokens[1] = tokens[2] = 100.
    tokens[[0, 3, 5]] = -100.
    tokens.requires_grad_()
    captured = []
    hook = encoder.net.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        output = encoder(tokens, coords, edges, selected, rotate, context, attributes)
    finally:
        hook.remove()
    start = 8 * tokens.shape[-1] + encoder.rbf_centers.numel() + 6
    phase = captured[0][:, start:start + 7]
    assert phase[0, :2].abs().max() > .01 and phase[0, 2] > .1
    assert torch.isfinite(output).all() and torch.isfinite(phase).all()
    phase.sum().backward()
    assert torch.isfinite(tokens.grad).all()


# Torus score norm

def test_import_preserves_numpy_rng():
    before = np.random.get_state()
    from tricoopdiff.diffdock.utils import torus
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert torus.SCORE_NORM_VERSION == 'wrapped_normal_quadrature_v1'


def test_score_norm_matches_references():
    from tricoopdiff.diffdock.utils import torus
    rng = np.random.default_rng(42)
    sigmas = np.array([.01, .1, .5, 1., 2., 3.14])
    estimate = torus.score_norm(sigmas)
    # Lookup uses the nearest log-sigma row; compare to that actual sigma.
    indices = np.round(np.clip(
        (np.log(sigmas/np.pi)-np.log(torus.SIGMA_MIN)) /
        np.log(torus.SIGMA_MAX/torus.SIGMA_MIN)*torus.SIGMA_N, 0, torus.SIGMA_N)).astype(int)
    effective = torus.sigma[indices]
    assert np.allclose(estimate[:2], 1/effective[:2]**2, rtol=1e-4)
    draws = rng.normal(size=(200000, len(sigmas))) * effective
    mc = np.mean(torus.score(draws, effective)**2, axis=0)
    assert np.allclose(estimate, mc, rtol=.025), (estimate, mc)
    again = torus.deterministic_score_norm(torus.p_, torus.score_, torus.x, chunk_size=17)
    assert np.array_equal(again, torus.score_norm_)


def test_checkpoint_normalization_provenance():
    from tricoopdiff.diffdock.utils import torus
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert torus.validate_score_norm_version({}) == 'legacy_unrecorded'
        assert len(caught) == 1 and 'cannot be recovered' in str(caught[0].message)
    assert torus.validate_score_norm_version({
        'torus_score_norm_version': torus.SCORE_NORM_VERSION,
    }) == torus.SCORE_NORM_VERSION
    try:
        torus.validate_score_norm_version({'torus_score_norm_version': 'wrong'})
    except ValueError:
        pass
    else:
        raise AssertionError('known normalization mismatches must fail')


# Knn compat

def test_knn_messages_flow_from_neighbor_to_query():
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    edges = _knn_graph_pytorch(coords, 1)
    assert torch.equal(edges, torch.tensor([[1, 0, 1], [0, 1, 2]]))
    reverse = _knn_graph_pytorch(coords, 1, flow='target_to_source')
    assert torch.equal(reverse, edges.flip(0))


def test_small_receptors_do_not_acquire_self_loops():
    coords = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    edges = _knn_graph_pytorch(coords, 24)
    assert edges.shape == (2, 90)
    assert not (edges[0] == edges[1]).any()
    assert torch.equal(torch.bincount(edges[1]), torch.full((10,), 9))
    assert _knn_graph_pytorch(coords[:1], 24).shape == (2, 0)


def test_batches_and_explicit_self_loops():
    coords = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [2., 0.]])
    batch = torch.tensor([0, 0, 1, 1])
    edges = _knn_graph_pytorch(coords, 1, batch=batch)
    assert torch.equal(batch[edges[0]], batch[edges[1]])
    loops = _knn_graph_pytorch(coords, 1, batch=batch, loop=True)
    assert torch.equal(loops, torch.arange(4).repeat(2, 1))


def test_large_translation_preserves_nearest_neighbor_order():
    torch.manual_seed(118)
    coords = torch.randn(40, 3) + 10000
    distances = (coords[:, None] - coords[None, :]).square().sum(-1)
    distances.fill_diagonal_(float('inf'))
    nearest = distances.argmin(-1)
    edges = _knn_graph_pytorch(coords, 1)
    assert torch.equal(edges[0], nearest), 'cancellation changed nearest neighbors'
    assert torch.equal(edges[1], torch.arange(len(coords)))


# Radius compat

def test_radius_keeps_cuda_source_order_and_strict_boundary():
    x = torch.tensor([[0.5, 0.], [0.25, 0.], [0.75, 0.], [1., 0.]])
    y = torch.zeros(1, 2)
    edges = _radius_pytorch(x, y, 1, max_num_neighbors=2)
    assert edges.dtype == torch.long
    assert torch.equal(edges, torch.tensor([[0, 0], [0, 1]]))
    edges = _radius_pytorch(x, y, 1, max_num_neighbors=10)
    assert torch.equal(edges[1], torch.tensor([0, 1, 2])), 'radius boundary is strict'


def test_cap_is_shared_across_source_blocks_and_batches():
    x = torch.zeros(2200, 3)
    x[:1020] = 10
    y = torch.zeros(3, 3)
    bx = torch.tensor([0] * 2100 + [2] * 100)
    by = torch.tensor([0, 1, 2])
    edges = _radius_pytorch(x, y, 1, bx, by, max_num_neighbors=8)
    assert torch.equal(edges[0], torch.tensor([0] * 8 + [2] * 8))
    assert torch.equal(edges[1], torch.cat((torch.arange(1020, 1028),
                                           torch.arange(2100, 2108))))


def test_graph_matches_163_self_filter_and_positional_arguments():
    x = torch.zeros(6, 3)
    # v1.6.3 requests cap+1 before removing self, even if self was not selected.
    edges = _radius_graph_pytorch(x, 1, None, False, 2)
    assert not (edges[0] == edges[1]).any()
    assert torch.equal(edges[0, edges[1] == 5], torch.tensor([0, 1, 2]))
    reverse = _radius_graph_pytorch(x, 1, flow='target_to_source', max_num_neighbors=2)
    assert torch.equal(reverse, edges.flip(0))


def test_empty_zero_cap_and_large_coordinates():
    x = torch.tensor([10000., 10000.25, 10000.5])
    edges = _radius_pytorch(x, x[:1], .4)
    assert torch.equal(edges, torch.tensor([[0, 0], [0, 1]]))
    assert _radius_pytorch(x, x, 1, max_num_neighbors=0).shape == (2, 0)
    assert _radius_pytorch(x[:0], x, 1).shape == (2, 0)
    assert _radius_graph_pytorch(x, 1, max_num_neighbors=0).shape == (2, 0)


# Equivariant clamp

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
