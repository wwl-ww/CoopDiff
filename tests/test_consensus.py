"""Shared geometry and protected-binary local compatibility contracts.

Retired velocity-head assertions remain on model-v4, alongside their code.
"""

import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.diffdock.utils.torus import SCORE_NORM_VERSION
from tricoopdiff.geometry.rigid import rotation_exp
from tricoopdiff.geometry.rigid import rotation_exp as so3_exp
from tricoopdiff.models.consensus import (
    BRANCHES, ConsensusFuse, production_model_config,
)
from tricoopdiff.sdpi_flow import SDPIFlow, INDEPENDENT_RIGID_MODE, SYMMETRIC_TORSION_MODE
from tricoopdiff.train import validate_consensus_checkpoint


def fixture():
    torch.manual_seed(71)
    graph = {'node_features': torch.randn(15, 16), 'node_coords': torch.randn(15, 3),
             'node_types': torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
             'lig_edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             'lig_edge_mask': torch.tensor([False, False, True, False, False, False]),
             'lig_mask_rotate': torch.tensor([[False, False, True, True]]),
             'lig_edge_attr': torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 4).float(),
             'expert_node_features': torch.cat((torch.randn(15, 116), torch.ones(15, 2)), -1),
             'expert_node_vectors': torch.zeros(15, 40, 3),
             'protein_local_geometry': torch.zeros(11, 32)}
    proposals = {b: {c: torch.randn(1, 1 if c == 'tor' else 3)
                      for c in ('tr', 'rot', 'tor')} for b in BRANCHES}
    fuse = ConsensusFuse(node_dim=32, input_dim=16, num_transformer_layers=1, dropout=0).eval()
    clock = torch.randn(1, 9)
    clock[:, 5:8] = .05
    clock[:, -1] = 1.
    return fuse, graph, proposals, clock


def test_legacy_checkpoint_rejected():
    try:
        validate_consensus_checkpoint({'checkpoint_version': 17})
    except ValueError as error:
        assert '18' in str(error)
    else:
        raise AssertionError('legacy weights must not masquerade as model-v2')


def _checkpoint_payload(version):
    runtime = {
        'activation_clamp': 100.0, 'activation_clamp_mode': 'irrep_norm',
        'residual_weight': 1e-3, 'ema_decay': 0.999,
        'torus_score_norm_version': SCORE_NORM_VERSION,
    }
    if version in (19, 20):
        runtime.update({
            'flow_time_contract': 'shared_flow_expert_clock_v1',
            'candidate_source_contract': 'candidate_source_conditioning_v1',
            'training_source_mix': 'sdpi_bitseed_1_1',
        })
    if version == 20:
        runtime['training_endpoint_protocol'] = 'online_bitseed_training_c1_m1_v1'
    return {
        'checkpoint_version': version,
        'architecture': 'residual_consensus_v2',
        'generative_process': 'sdpi_flow_v1',
        'expert_lora_config': None, 'expert_lora_state_dict': {},
        'trifuse_state_dict': {'weight': torch.ones(1)},
        'ema_trifuse_state_dict': {'weight': torch.ones(1)},
        'model_config': {'node_dim': 32},
        'expert_base_identity': {'sha256': 'test'},
        'runtime_config': runtime,
    }


def test_checkpoint_contract_accepts_v18_v19_and_explicit_v20():
    validate_consensus_checkpoint(_checkpoint_payload(18))
    validate_consensus_checkpoint(_checkpoint_payload(19))
    validate_consensus_checkpoint(_checkpoint_payload(20))


def test_v19_checkpoint_requires_explicit_source_and_time_contracts():
    payload = _checkpoint_payload(19)
    del payload['runtime_config']['candidate_source_contract']
    try:
        validate_consensus_checkpoint(payload)
    except ValueError as error:
        assert 'source contract' in str(error)
    else:
        raise AssertionError('v19 must record its candidate source contract')


def test_v20_checkpoint_requires_online_c1m1_for_mixed_training():
    payload = _checkpoint_payload(20)
    payload['runtime_config']['training_endpoint_protocol'] = 'full_pool_c4_m8'
    try:
        validate_consensus_checkpoint(payload)
    except ValueError as error:
        assert 'endpoint protocol' in str(error)
    else:
        raise AssertionError('mixed v20 must record online C=1/M=1 generation')


def test_global_node_context_reaches_nodes_outside_local_receptive_field():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    torch.manual_seed(31)
    local = TernaryGeometryEncoder(16, hidden_dim=16, joint_geometry=True)
    global_model = TernaryGeometryEncoder(16, hidden_dim=16, joint_geometry=True,
                                         global_node_context=True)
    missing, unexpected = global_model.load_state_dict(local.state_dict(), strict=False)
    assert missing and all(key.startswith('global_updates.') for key in missing)
    assert not unexpected
    features = torch.randn(18, 16, requires_grad=True)
    types = torch.arange(3).repeat_interleave(6)
    coords = torch.arange(54, dtype=torch.float32).reshape(18, 3) / 10
    coords[-1] = 1000
    local_features, _ = local._joint_contacts(features, coords, types)
    upgraded, _ = global_model._joint_contacts(features, coords, types)
    assert torch.equal(local_features, upgraded), 'zero-initialized global upgrade must be identity'
    local_grad = torch.autograd.grad(local_features[0].square().sum(), features)[0]
    assert torch.count_nonzero(local_grad[-1]) == 0, 'fixture must exclude the remote node locally'
    for update in global_model.global_updates:
        torch.nn.init.normal_(update[-1].weight, std=.03)
    upgraded, _ = global_model._joint_contacts(features, coords, types)
    global_grad = torch.autograd.grad(upgraded[0].square().sum(), features)[0]
    assert torch.isfinite(global_grad).all()
    assert global_grad[-1].abs().sum() > 0, 'remote ligand node must influence E3 node via global context'
    assert global_model.global_node_context


def test_spatial_edge_features_are_global_invariant_and_handed():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, hidden_dim=16, joint_geometry=True,
                                     spatial_geometry=True)
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                           [0., 3., 1.], [1., 3., 1.],
                           [0., 1., 0.], [1., 1., 0.]], requires_grad=True)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    receiver = torch.tensor([4])
    direction = torch.tensor([[0., -1., 0.]])
    geometry = encoder._spatial_edge_geometry(coords, types, receiver, direction)
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    moved = encoder._spatial_edge_geometry(coords @ rotation.T + 20, types,
                                           receiver, direction @ rotation.T)
    assert geometry.shape == (1, 8)
    assert torch.allclose(geometry, moved, atol=2e-6)
    # The E3-ligand edge is identical, but moving POI must change its context.
    displaced = coords.detach().clone()
    displaced[types == 1] += torch.tensor([0., 0., 4.])
    changed = encoder._spatial_edge_geometry(displaced, types, receiver, direction)
    assert not torch.allclose(changed, geometry)
    reflection = torch.diag(torch.tensor([-1., 1., 1.]))
    mirrored = encoder._spatial_edge_geometry(coords @ reflection, types,
                                              receiver, direction @ reflection)
    assert torch.allclose(mirrored[:, :-1], geometry[:, :-1])
    assert geometry[:, -1].abs().sum() > .01
    assert torch.allclose(mirrored[:, -1], -geometry[:, -1])
    geometry.square().sum().backward()
    assert torch.isfinite(coords.grad).all()
    assert coords.grad[types == 1].abs().sum() > 0
    collapsed = torch.zeros_like(coords, requires_grad=True)
    zero_geometry = encoder._spatial_edge_geometry(collapsed, types, receiver,
                                                  torch.zeros_like(direction))
    zero_geometry.sum().backward()
    assert torch.isfinite(zero_geometry).all() and torch.isfinite(collapsed.grad).all()


def test_unpooled_global_attention_reaches_distant_nodes_without_summary_broadcast():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, hidden_dim=16, joint_geometry=True,
                                     global_node_attention=True)
    features = torch.randn(18, 16, requires_grad=True)
    types = torch.arange(3).repeat_interleave(6)
    coords = torch.arange(54, dtype=torch.float32).reshape(18, 3) / 10
    coords[-1] = 1000
    assert not encoder.global_node_context, 'this check must not rely on pooled component summaries'
    for attention in encoder.global_attention:
        torch.nn.init.normal_(attention.out_proj.weight, std=.03)
    output, _ = encoder._joint_contacts(features, coords, types)
    gradient = torch.autograd.grad(output[0].square().sum(), features)[0]
    assert torch.isfinite(gradient).all() and gradient[-1].abs().sum() > 0


def test_chunked_global_histogram_preserves_values_and_coordinate_gradients():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, hidden_dim=16).double()
    encoder.RBF_CHUNK_SIZE = 3  # Exercise several independently recomputed chunks.
    torch.manual_seed(33)
    left = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    distances = torch.cdist(left, right).flatten()
    reference = torch.exp(-.5 * ((distances[:, None] - encoder.rbf_centers)
                                 / encoder.rbf_width) ** 2).mean(0)
    actual = encoder._distance_histogram(distances)
    assert torch.allclose(actual, reference, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (left, right), retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().sum(), (left, right))
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-12)
               for a, b in zip(actual_grad, reference_grad))
    with torch.no_grad():
        assert torch.allclose(encoder._distance_histogram(distances), reference)


def test_independent_decoder_does_not_move_protein_when_only_ligand_moves():
    from tricoopdiff.sdpi_flow import SDPIFlow, INDEPENDENT_RIGID_MODE

    _, graph, _, _ = fixture()
    ligand = graph['node_coords'][graph['node_types'] == 2][None]
    poi = graph['node_coords'][graph['node_types'] == 1][None]
    body = torch.eye(3)[None]
    velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                  'e3_ligand': {'tr': torch.tensor([[1., 2., 3.]]),
                                'rot': torch.tensor([[.5, -.2, .3]])},
                  'poi_ligand': {'tr': torch.zeros(1, 3), 'rot': torch.zeros(1, 3)}}
    moved_ligand, moved_poi, moved_body = SDPIFlow.rigid_step(
        ligand, poi, body, velocities, torch.tensor([.05]))
    assert not torch.allclose(moved_ligand, ligand)
    assert torch.allclose(moved_poi, poi, atol=1e-7)
    assert torch.equal(moved_body, body)
    velocities['poi_ligand']['rot'] = torch.tensor([[.2, .4, -.3]])
    _, rotated_poi, _ = SDPIFlow.rigid_step(ligand, poi, body, velocities, torch.tensor([.05]))
    assert torch.allclose(rotated_poi.mean(1), poi.mean(1), atol=1e-6)
    _, held_poi, held_body = SDPIFlow.rigid_step(
        ligand, poi, body, velocities, torch.tensor([.05]), freeze_poi=True)
    assert torch.equal(held_poi, poi) and torch.equal(held_body, body)


def test_direct_checkpoint_cannot_be_decoded_as_legacy_motion():
    from tricoopdiff.models.consensus import DIRECT_ARCHITECTURE

    payload = _checkpoint_payload(20)
    payload['architecture'] = DIRECT_ARCHITECTURE
    payload['model_config']['direct_ternary_motion'] = True
    validate_consensus_checkpoint(payload)
    payload['architecture'] = 'residual_consensus_v2'
    try:
        validate_consensus_checkpoint(payload)
    except ValueError as error:
        assert 'disagree' in str(error)
    else:
        raise AssertionError('new output semantics must never silently use the old decoder')


def test_chunked_expert_tensor_products_preserve_outputs_and_gradients():
    from e3nn import o3
    from torch_scatter import scatter
    from tricoopdiff.diffdock.models.tensor_layers import tp_scatter_multigroup
    torch.manual_seed(73)
    tp = o3.FullyConnectedTensorProduct('2x0e', '1x0e', '2x0e', shared_weights=False).double()
    layers = torch.nn.ModuleList([torch.nn.Linear(3, tp.weight_numel).double() for _ in range(2)])
    nodes = torch.randn(17, 2, dtype=torch.float64, requires_grad=True)
    attributes = [torch.randn(1400, 3, dtype=torch.float64, requires_grad=True),
                  torch.randn(1300, 3, dtype=torch.float64, requires_grad=True)]
    indices = torch.randint(17, (2, 2700))
    harmonics = torch.randn(2700, 1, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(2700, 1, dtype=torch.float64, requires_grad=True)
    inputs = [nodes, *attributes, harmonics, weights, *layers.parameters()]
    direct = torch.cat([layers[0](attributes[0]), layers[1](attributes[1])]) * weights
    reference = scatter(tp(nodes[indices[1]], harmonics, direct), indices[0], dim=0, dim_size=17, reduce='mean')
    expected = torch.autograd.grad(reference.square().sum(), inputs)
    actual = tp_scatter_multigroup(tp, layers, nodes, indices, attributes, harmonics, edge_weight=weights)
    gradients = torch.autograd.grad(actual.square().sum(), inputs)
    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-8)
    assert all(torch.allclose(a, b, atol=1e-9, rtol=1e-7) for a, b in zip(gradients, expected))


def test_joint_direction_has_bounded_gradient_at_overlapping_nodes():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder
    encoder = TernaryGeometryEncoder(16, 32, joint_geometry=True, spatial_geometry=True)
    coords = torch.zeros(6, 3, requires_grad=True)
    features = torch.randn(6, 32)
    _, joint = encoder._joint_contacts(features, coords, torch.tensor([0, 0, 1, 1, 2, 2]))
    joint['direction'].sum().backward()
    assert torch.isfinite(coords.grad).all() and coords.grad.abs().max() <= 100


def test_geometry_reuses_pair_distances_and_handles_single_node_components():
    from unittest.mock import patch
    from tricoopdiff.models.consensus import TernaryGeometryEncoder
    encoder = TernaryGeometryEncoder(16, 32, joint_geometry=True, spatial_geometry=True)
    coords = torch.zeros(5, 3, requires_grad=True)
    features = torch.randn(5, 16)
    with patch('torch.cdist', wraps=torch.cdist) as distances:
        result = encoder(features, coords, torch.tensor([0, 0, 1, 1, 2]))
    assert distances.call_count == 6, 'each component pair and kNN distance matrix should be built once'
    (result['global'].sum() + result['ligand_tokens'][0].sum()).backward()
    assert torch.isfinite(coords.grad).all(), 'zero-radius components must not create sqrt(0) gradient NaNs'


def test_consensus_rejects_unknown_architecture_arguments():
    try:
        ConsensusFuse(typo_dimension=64)
    except TypeError:
        pass
    else:
        raise AssertionError('unknown architecture options must not be silently ignored')


def test_ligand_category_encoding_never_mutates_integer_inputs():
    from tricoopdiff.models.consensus import LigandFeatureEncoder

    encoder = LigandFeatureEncoder(16)
    features = torch.zeros(3, len(encoder.embeddings), dtype=torch.long)
    features[0, 0], features[1, 0] = 999, -1
    original = features.clone()
    encoded = encoder(features)
    assert torch.isfinite(encoded).all()
    assert torch.equal(features, original), 'category encoding modified the caller graph'
    assert torch.allclose(encoded, encoder(original.float()))


def test_ternary_contacts_and_gradients_survive_large_common_translation():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, 32, joint_geometry=True, spatial_geometry=True).eval()
    line = torch.stack((torch.arange(32).float() / 8, torch.zeros(32), torch.zeros(32)), -1)
    coords = torch.cat((line, line + torch.tensor([0., 2., 0.]),
                        line + torch.tensor([0., 0., 2.]))).requires_grad_()
    translated = (coords.detach() + 1024).requires_grad_()
    types, features = torch.arange(3).repeat_interleave(32), torch.randn(96, 16)
    outputs = [encoder(features, value, types) for value in (coords, translated)]
    for key in ('global', 'e3l', 'pol'):
        assert torch.allclose(outputs[0][key], outputs[1][key], atol=1e-6), key
    first, second = (output['joint_geometry'] for output in outputs)
    assert torch.equal(first['receiver'], second['receiver'])
    assert torch.equal(first['sender'], second['sender']), 'translation changed nearest neighbours'
    for output in outputs:
        (output['global'].square().mean() + output['ligand_tokens'][0].square().mean()).backward()
    assert torch.isfinite(coords.grad).all() and torch.isfinite(translated.grad).all()
    assert torch.allclose(coords.grad, translated.grad, atol=1e-6), 'translation changed geometric gradients'


def test_continuous_geometry_is_se3_and_atom_permutation_invariant():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    generator = torch.Generator().manual_seed(941)
    encoder = TernaryGeometryEncoder(16, 16, joint_geometry=True, spatial_geometry=True,
                                     continuous_geometry=True).double().eval()
    features = torch.randn(9, 16, generator=generator, dtype=torch.float64)
    coords = (torch.randn(9, 3, generator=generator, dtype=torch.float64) * 4).requires_grad_()
    types = torch.arange(3).repeat_interleave(3)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    expected = encoder(features, coords, types, lig_edge_index=edges)
    rotation = so3_exp(coords.new_tensor([.3, -.5, .2]))
    moved = (coords.detach() @ rotation.T + coords.new_tensor([11., -7., 4.])).requires_grad_()
    transformed = encoder(features, moved, types, lig_edge_index=edges)
    for key in ('global', 'e3l', 'pol'):
        assert torch.allclose(expected[key], transformed[key], atol=1e-10, rtol=0.), key
    assert torch.allclose(expected['ligand_tokens'][0], transformed['ligand_tokens'][0],
                          atol=1e-10, rtol=0.)
    permutations = [torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]), torch.tensor([2, 1, 0])]
    permutation = torch.cat([value + 3 * index for index, value in enumerate(permutations)])
    reordered = encoder(features[permutation], coords[permutation], types,
                         lig_edge_index=permutations[2].argsort()[edges])
    assert torch.allclose(expected['global'], reordered['global'], atol=1e-10, rtol=0.)
    assert torch.allclose(expected['ligand_tokens'][0][permutations[2]],
                          reordered['ligand_tokens'][0], atol=1e-10, rtol=0.)
    for output in (expected, transformed):
        (output['global'].square().sum() + output['ligand_tokens'][0].square().sum()).backward()
    assert torch.isfinite(coords.grad).all() and torch.isfinite(moved.grad).all()
    assert torch.allclose(coords.grad @ rotation.T, moved.grad, atol=1e-9, rtol=0.)


def test_continuous_radius_edges_vanish_with_zero_boundary_derivative():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, 8, joint_geometry=True, continuous_geometry=True).double()
    for radius, types in ((12., torch.tensor([0, 0, 1, 2])),
                          (24., torch.tensor([0, 1, 2]))):
        base = torch.zeros(len(types), 3, dtype=torch.float64)
        base[:, 0] = torch.arange(len(types), dtype=torch.float64) * 60
        base[1, 0] = radius
        features = torch.randn(len(types), 8, dtype=torch.float64)
        values, gradients, edge_counts = [], [], []
        for offset in (-1e-3, 0., 1e-3):
            coords = base.clone()
            coords[1, 0] += offset
            coords.requires_grad_()
            output, info = encoder._joint_contacts(features, coords, types)
            (output.square().sum() + coords.sum() * 0).backward()
            values.append(output.detach())
            gradients.append(coords.grad)
            edge_counts.append(len(info['receiver']))
            assert torch.isfinite(output).all() and torch.isfinite(coords.grad).all()
            assert bool(((info['weight'] >= 0) & (info['weight'] <= 1)).all())
        assert edge_counts == [2, 0, 0], 'radius topology did not match the explicit support'
        assert (values[0] - values[1]).abs().max() < 1e-6
        assert (gradients[0] - gradients[1]).abs().max() < 1e-3
        assert torch.equal(values[1], values[2])


def test_continuous_contact_descriptors_do_not_jump_at_thresholds_or_distance_ties():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, 8, continuous_geometry=True).double()
    values = []
    for offset in (-1e-6, 0., 1e-6):
        distances = (torch.tensor([4., 4., 8., 12.], dtype=torch.float64) + offset).requires_grad_()
        descriptor = encoder._pair_descriptor(distances)
        descriptor.sum().backward()
        assert torch.isfinite(distances.grad).all()
        values.append(descriptor)
    assert (values[0] - values[2]).abs().max() < 1e-6


def test_continuous_weighted_rbf_chunking_matches_direct_values_and_gradients():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    encoder = TernaryGeometryEncoder(16, 8, continuous_geometry=True).double()
    generator = torch.Generator().manual_seed(946)
    distances = (torch.rand(90, 100, generator=generator, dtype=torch.float64) * 30).requires_grad_()
    weights = torch.rand(90, 100, generator=generator, dtype=torch.float64).requires_grad_()
    for per_row in (False, True):
        actual = encoder._weighted_rbf(distances, weights, per_row=per_row)
        rbf = torch.exp(-.5 * ((distances[..., None] - encoder.rbf_centers) / encoder.rbf_width).square())
        expected = (rbf * weights[..., None]).sum(1)
        if not per_row:
            expected = expected.sum(0)
        assert torch.allclose(actual, expected, atol=1e-11, rtol=0.)
        actual_grad = torch.autograd.grad(actual.square().sum(), (distances, weights), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), (distances, weights), retain_graph=True)
        for left, right in zip(actual_grad, expected_grad):
            assert torch.isfinite(left).all()
            assert torch.allclose(left, right, atol=1e-9, rtol=1e-11)


def test_continuous_joint_message_chunks_match_unbounded_reference_backward():
    from tricoopdiff.models.consensus import TernaryGeometryEncoder

    generator = torch.Generator().manual_seed(947)
    encoder = TernaryGeometryEncoder(16, 8, joint_geometry=True, continuous_geometry=True).double()
    features = torch.randn(99, 8, generator=generator, dtype=torch.float64).requires_grad_()
    coords = torch.randn(99, 3, generator=generator, dtype=torch.float64).requires_grad_()
    types = torch.arange(3).repeat_interleave(33)
    actual, info = encoder._joint_contacts(features, coords, types)
    receiver, sender, weight = info['receiver'], info['sender'], info['weight']
    assert len(receiver) > 8192, 'the chunk regression must cover more than one message chunk'
    counts = features.new_zeros(99, 1).index_add(0, receiver, weight) + 1.
    expected = features
    for message, update in zip(encoder.messages, encoder.message_updates):
        values = message(torch.cat((expected[receiver], expected[sender], info['geometry']), -1))
        aggregate = torch.zeros_like(expected).index_add(0, receiver, values * weight) / counts
        expected = expected + update(torch.cat((expected, aggregate), -1))
    assert torch.allclose(actual, expected, atol=1e-11, rtol=0.)
    actual_grad = torch.autograd.grad(actual.square().sum(), (features, coords), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (features, coords))
    for left, right in zip(actual_grad, expected_grad):
        assert torch.isfinite(left).all()
        assert torch.allclose(left, right, atol=1e-9, rtol=1e-10), (
            'checkpoint recomputation captured a later layer or chunk')


def test_bounded_vector_retains_large_finite_directions_and_coordinate_gradients():
    from tricoopdiff.models.consensus import bounded_vector

    for dtype, magnitude in ((torch.float32, 1e20), (torch.float64, 1e200)):
        values = torch.tensor([[magnitude, 0., 0.]], dtype=dtype, requires_grad=True)
        compressed = bounded_vector(values)
        assert torch.equal(compressed, torch.tensor([[1., 0., 0.]], dtype=dtype)), (
            'finite expert directions disappeared because their squared norm overflowed')
        compressed.sum().backward()
        assert torch.isfinite(values.grad).all() and values.grad[0, 1] > 0
        assert torch.allclose(values.grad[0, 1] * magnitude, values.new_tensor(1.))
    ordinary = torch.tensor([[0., 0., 0.], [.4, -.8, 2.]], requires_grad=True)
    expected = ordinary / (1 + torch.linalg.vector_norm(ordinary, dim=-1, keepdim=True))
    actual = bounded_vector(ordinary)
    assert torch.equal(actual, expected), 'ordinary old-checkpoint arithmetic changed'
    actual.sum().backward()
    assert torch.isfinite(ordinary.grad).all()
    values = torch.tensor([[1e20, -2e20, 3e20]])
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]]))[0]
    assert torch.allclose(bounded_vector(values @ rotation.T),
                          bounded_vector(values) @ rotation.T, atol=2e-7, rtol=0.)


def test_symmetric_bond_context_matches_reversed_fragments_and_gradients():
    from tricoopdiff.models.consensus import BondContextEncoder

    _, graph, _, _ = fixture()
    encoder = BondContextEncoder(32, chemical_context=True, symmetric=True).double()
    tokens = torch.randn(4, 32, dtype=torch.double, requires_grad=True)
    coords = graph['node_coords'][-4:].double().requires_grad_()
    context = torch.randn(1, 32, dtype=torch.double, requires_grad=True)
    edge, mask, rotate = (graph[key] for key in ('lig_edge_index', 'lig_edge_mask', 'lig_mask_rotate'))
    attributes = graph['lig_edge_attr'].double()
    a = encoder(tokens, coords, edge, mask, rotate, context, attributes)
    b = encoder(tokens, coords, edge.flip(0), mask, ~rotate, context, attributes)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12), 'bond bookkeeping changes symmetric tokens'
    probe = torch.randn_like(a)
    ga = torch.autograd.grad((a * probe).sum(), (tokens, coords, context), retain_graph=True)
    gb = torch.autograd.grad((b * probe).sum(), (tokens, coords, context), retain_graph=True)
    for left, right in zip(ga, gb):
        assert torch.isfinite(left).all() and left.abs().sum() > 0
        assert torch.allclose(left, right, atol=1e-11, rtol=1e-10), 'bond reversal changes gradients'
    direction = torch.randn_like(coords)
    step = 1e-5
    values = [encoder(tokens, coords + sign * step * direction, edge, mask, rotate,
                      context, attributes) for sign in (-1, 1)]
    numeric = ((values[1] - values[0]) * probe).sum() / (2 * step)
    assert torch.allclose(numeric, (ga[1] * direction).sum(), atol=1e-7, rtol=1e-5)



def _velocity_vector(output):
    value = output['normalized_velocity']
    return torch.cat([value[branch][key][0] for branch in BRANCHES for key in ('tr', 'rot')]
                     + [value['torsion'][0]])


def _local_module(fuse):
    from tricoopdiff.models.local_compatibility import LocalCompatibilityCorrection
    return next(module for module in fuse.modules() if isinstance(module, LocalCompatibilityCorrection))


def test_protected_binary_baseline_is_exact_at_zero_correction_and_never_reweighted():
    fuse, graph, proposals, clock = fixture()
    base = fuse(graph, proposals, clock)
    assert torch.equal(base['local_correction'], torch.zeros_like(base['local_correction']))
    assert torch.equal(_velocity_vector(base), base['binary_velocity']), 'zero correction must restore the exact binary field'
    assert torch.equal(_velocity_vector(base), _velocity_vector(fuse(graph, proposals, clock, source=1))), \
        'source provenance must not become a learned motion or expert gate'
    local = _local_module(fuse)
    with torch.no_grad():
        local.pair_head[-1].weight.normal_(std=.1)
        local.triplet_head[-1].weight.normal_(std=.1)
    changed = fuse(graph, proposals, clock)
    assert changed['local_correction'].abs().sum() > 0
    assert torch.equal(changed['binary_velocity'], base['binary_velocity']), 'learned compatibility changed protected binary scores'
    assert torch.allclose(_velocity_vector(changed), changed['binary_velocity'] + changed['local_correction'])
    assert changed['local_correction'][:12].reshape(4, 3).norm(dim=-1).max() < local.action_limit
    assert changed['local_correction'][12:].abs().max() < local.action_limit
    assert not any(name in name_ for name_, _ in fuse.named_parameters()
                   for name in ('film', 'anchor', 'operator_controller', 'distance_head', 'expert_prior_gate'))


def test_local_fusion_equivariance_no_native_inputs_and_empty_torsions():
    fuse, graph, proposals, clock = fixture()
    with torch.no_grad():
        _local_module(fuse).pair_head[-1].weight.normal_(std=.1)
        _local_module(fuse).triplet_head[-1].weight.normal_(std=.1)
    rotation = rotation_exp(torch.tensor([[.3, -.4, .2]]))[0]
    shifted_graph = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 9.}
    shifted_proposals = {name: {key: value @ rotation.T if key != 'tor' else value
                               for key, value in branch.items()} for name, branch in proposals.items()}
    first = fuse(graph, proposals, clock)
    moved = fuse(shifted_graph, shifted_proposals, clock)
    for field in ('binary_velocity', 'local_correction'):
        assert torch.allclose(moved[field][:12].reshape(4, 3), first[field][:12].reshape(4, 3) @ rotation.T,
                              atol=2e-5, rtol=2e-4), field
        assert torch.allclose(moved[field][12:], first[field][12:], atol=2e-5)
    poisoned = fuse({**graph, 'native_coords': torch.full_like(graph['node_coords'], float('nan')),
                     'target_ligand': torch.full((1, 4, 3), float('nan'))}, proposals, clock)
    assert torch.equal(_velocity_vector(first), _velocity_vector(poisoned)), 'native labels entered production inference'
    empty = {name: {**branch, 'tor': torch.empty(1, 0)} for name, branch in proposals.items()}
    output = fuse(graph, empty, clock)
    assert output['normalized_velocity']['torsion'].shape == (1, 0)
    assert output['binary_velocity'].shape == (12,)


def test_scalar_compatibility_trains_from_executed_coordinates_and_runs_in_inference_mode():
    fuse, graph, proposals, clock = fixture()
    output = fuse(graph, proposals, clock)
    ligand = graph['node_coords'][None, graph['node_types'] == 2]
    poi = graph['node_coords'][None, graph['node_types'] == 1]
    ed = graph['node_coords'][None, graph['node_types'] == 0]
    ld = {'coords': ligand, 'edge_index': graph['lig_edge_index'],
          'edge_mask': graph['lig_edge_mask'], 'mask_rotate': graph['lig_mask_rotate']}
    flow = SDPIFlow()
    moved = flow.apply_output(ligand, poi, torch.eye(3)[None],
        flow.velocities(output['normalized_velocity'], ed, poi), torch.tensor([.05]), ld, output)
    loss = moved[0].square().mean() + moved[1].square().mean()
    loss.backward()
    local = _local_module(fuse)
    assert all(head[-1].weight.grad is not None and torch.isfinite(head[-1].weight.grad).all()
               and head[-1].weight.grad.abs().sum() > 0 for head in (local.pair_head, local.triplet_head))
    with torch.no_grad():
        local.pair_head[-1].weight.normal_(std=.1)
    ordinary = fuse(graph, proposals, clock)
    with torch.inference_mode():
        evaluated = fuse(graph, proposals, clock)
    assert torch.allclose(_velocity_vector(ordinary), _velocity_vector(evaluated), atol=1e-6)
    assert not evaluated['local_correction'].requires_grad


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')
