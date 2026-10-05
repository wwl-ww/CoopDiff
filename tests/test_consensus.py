"""Shared geometry plus the model-4.5 constraint motion contract.

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
    BRANCHES, CONSTRAINT_MODEL_PROFILE, ConsensusFuse, production_model_config,
)
from tricoopdiff.models.constraint_motion import (
    binary_rigid_maps, horizontal_torsion_jacobian, point_jacobian, solve_velocity,
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
    return fuse, graph, proposals, torch.randn(1, 9)



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



def test_relation_backbone_keeps_interfaces_se3_and_atom_permutation():
    from tricoopdiff.models.consensus import TernaryRelationEncoder

    _, graph, _, _ = fixture()
    encoder = TernaryRelationEncoder(16, 24).double().eval()
    features = graph['node_features'].double()
    coords = graph['node_coords'].double().requires_grad_()
    types, edges = graph['node_types'], graph['lig_edge_index']
    reference = encoder(features, coords, types, lig_edge_index=edges)
    rotation = so3_exp(coords.new_tensor([.3, -.5, .2]))
    moved = (coords.detach() @ rotation.T + coords.new_tensor([7., -9., 17.])).requires_grad_()
    transformed = encoder(features, moved, types, lig_edge_index=edges)
    for key in ('global', 'e3l', 'pol', 'protein_pair'):
        assert torch.allclose(reference[key], transformed[key], atol=1e-10, rtol=0.), key
    for key in ('pair', 'relation', 'features'):
        assert torch.allclose(reference['joint_geometry'][key], transformed['joint_geometry'][key],
                              atol=1e-10, rtol=0.), key
    permutations = (torch.tensor([2, 4, 0, 1, 3]), torch.tensor([5, 2, 0, 1, 3, 4]),
                    torch.tensor([3, 0, 2, 1]))
    permutation = torch.cat((permutations[0], permutations[1] + 5, permutations[2] + 11))
    reordered = encoder(features[permutation], coords[permutation], types,
                        lig_edge_index=permutations[2].argsort()[edges])
    assert torch.allclose(reference['protein_pair'], reordered['protein_pair'], atol=1e-10, rtol=0.)
    assert torch.allclose(reference['joint_geometry']['features'][permutation],
                          reordered['joint_geometry']['features'], atol=1e-10, rtol=0.)
    # Pair relations are conditioned by the ligand, not an independent PP head.
    changed = features.clone()
    changed[types == 2, 0] = 0
    altered = encoder(changed, coords, types, lig_edge_index=edges)
    assert not torch.allclose(reference['protein_pair'], altered['protein_pair'])
    grad = torch.autograd.grad(reference['protein_pair'].square().sum(), coords)[0]
    transformed_grad = torch.autograd.grad(transformed['protein_pair'].square().sum(), moved)[0]
    assert torch.isfinite(grad).all()
    assert torch.allclose(grad @ rotation.T, transformed_grad, atol=1e-9, rtol=1e-9)



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



def test_geometric_anchors_cover_homogeneous_protein_and_respect_symmetry():
    from tricoopdiff.models.consensus import TernaryRelationEncoder

    torch.manual_seed(42)
    coords = (torch.randn(60, 3, dtype=torch.double) * 10).requires_grad_()
    def anchors(x):
        return TernaryRelationEncoder.anchor_geometry(x).softmax(-1) @ x
    points = anchors(coords)
    radius = (coords - coords.mean(0)).square().sum(-1).mean()
    spread = (points - points.mean(0)).square().sum(-1).mean()
    assert spread > .2 * radius, 'geometric anchors collapsed despite a spatially extended protein'
    rotation = so3_exp(coords.new_tensor([.3, -.2, .7]))
    assert torch.allclose(anchors(coords @ rotation.T + 8), points @ rotation.T + 8, atol=1e-10)
    assert torch.allclose(anchors(coords[torch.randperm(len(coords))]), points, atol=1e-10)
    points.square().sum().backward()
    assert torch.isfinite(coords.grad).all() and coords.grad.abs().sum() > 0
    symmetric = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]], dtype=torch.double)
    assert torch.allclose(anchors(symmetric.flip(0)), anchors(symmetric)), 'ties depend on residue order'



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



def test_stable_anchors_retain_coverage_and_equivariance():
    from tricoopdiff.models.consensus import TernaryRelationEncoder

    def anchors(x):
        return TernaryRelationEncoder.anchor_geometry(x, stable=True).softmax(-1) @ x

    for seed in (42, 71, 73):
        torch.manual_seed(seed)
        coords = torch.randn(60, 3, dtype=torch.double) * 10
        points = anchors(coords)
        radius = (coords - coords.mean(0)).square().sum(-1).mean()
        spread = (points - points.mean(0)).square().sum(-1).mean()
        assert spread > .2 * radius, f'stable anchors lost spatial coverage: seed={seed}, ratio={spread/radius}'
        rotation = so3_exp(coords.new_tensor([.3, -.2, .7]))
        assert torch.allclose(anchors(coords @ rotation.T + 8), points @ rotation.T + 8, atol=1e-9)
        assert torch.allclose(anchors(coords[torch.randperm(len(coords))]), points, atol=1e-9)



def test_stable_anchors_near_collinear_differences_and_perturbations():
    from tricoopdiff.models.consensus import TernaryRelationEncoder

    torch.manual_seed(73)
    base = torch.zeros(45, 3, dtype=torch.double)
    base[:, 0] = torch.linspace(-3, 3, len(base))
    noise = torch.randn_like(base)
    probe = torch.randn(16, 3, dtype=torch.double)
    for epsilon in (0., .001, .03):
        coords = (base + epsilon * noise).requires_grad_()

        def objective(x):
            points = TernaryRelationEncoder.anchor_geometry(x, stable=True).softmax(-1) @ x
            return (points.sin() * probe).sum()

        gradient = torch.autograd.grad(objective(coords), coords)[0]
        assert torch.isfinite(gradient).all()
        for _ in range(3):
            direction = torch.randn_like(coords) * .1
            step = 1e-5
            numeric = (objective(coords + step * direction) - objective(coords - step * direction)) / (2 * step)
            assert torch.allclose(numeric, (gradient * direction).sum(), atol=1e-6, rtol=1e-4), (
                'stable anchor local derivatives disagree with finite differences', epsilon)
            perturbed = (coords.detach() + 1e-4 * direction).requires_grad_()
            changed = torch.autograd.grad(objective(perturbed), perturbed)[0]
            assert torch.isfinite(changed).all()
            assert torch.linalg.vector_norm(changed - gradient) < .1 * (1 + torch.linalg.vector_norm(gradient)), (
                'stable anchor gradients amplify tiny coordinate perturbations', epsilon)



def test_expert_coordinate_stop_preserves_film_and_forward():
    from torch_geometric.data import HeteroData
    from tricoopdiff.models.consensus import ConsensusCoopDiff

    class Expert(torch.nn.Module):
        def forward_features(self, graph, film=None, return_vectors=False):
            return graph['ligand'].pos.square().sum() + graph['receptor'].pos.square().sum() + film.square().sum()

    model = ConsensusCoopDiff.__new__(ConsensusCoopDiff)
    torch.nn.Module.__init__(model)
    model.expert = Expert()
    model.trifuse = torch.nn.Identity()
    model.trifuse.endpoint_v2 = model.trifuse.refined_conditions = True
    model.expert_input_grad = True
    graph = HeteroData()
    graph['ligand'].pos = torch.randn(4, 3, requires_grad=True)
    graph['receptor'].pos = torch.randn(6, 3, requires_grad=True)
    film = torch.ones(3, requires_grad=True)
    model.expert_coordinate_grad = True
    original = model._read_expert(graph, film)
    model.expert_coordinate_grad = False
    stopped = model._read_expert(graph, film)
    assert torch.equal(original, stopped), 'coordinate stop changed expert forward'
    stopped.backward()
    assert graph['ligand'].pos.grad is None and graph['receptor'].pos.grad is None
    assert torch.equal(film.grad, 2 * film.detach()), 'FiLM learning path was severed'
    assert graph['ligand'].pos.requires_grad, 'shared trajectory was detached in place'



def test_joint_solver_rank_deficiency_reliability_and_gradients():
    # Contradictory opinions with a known weighted optimum, including a null DOF.
    a = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.double)
    b = torch.tensor([1., -1.], dtype=torch.double)
    weights = torch.tensor([9., 1.], dtype=torch.double)
    solution, _ = solve_velocity(a, b, weights, ridge=.01)
    assert torch.allclose(solution, torch.tensor([8 / 10.01, 0.], dtype=torch.double))
    zero, _ = solve_velocity(a * 0, b, weights)
    assert torch.equal(zero, torch.zeros_like(zero)), 'unobserved motion must be damped to zero'
    torch.manual_seed(3)
    values = (torch.randn(7, 4, dtype=torch.double, requires_grad=True),
              torch.randn(7, dtype=torch.double, requires_grad=True),
              torch.rand(7, dtype=torch.double, requires_grad=True))
    assert torch.autograd.gradcheck(lambda *args: solve_velocity(*args)[0], values, fast_mode=True)



def test_binary_maps_match_independent_interface_contract():
    centers = torch.randn(3, 3, dtype=torch.double)
    scales = torch.tensor([3., 7.], dtype=torch.double)
    velocity = torch.randn(15, dtype=torch.double)
    a_map, b_map = binary_rigid_maps(centers, scales, len(velocity))
    a, b = a_map @ velocity, b_map @ velocity
    target = {'e3_ligand': {'tr': a[None, :3] * scales[0], 'rot': a[None, 3:] * math.pi},
              'poi_ligand': {'tr': b[None, :3] * scales[1], 'rot': b[None, 3:] * math.pi},
              'torsion': velocity[None, 12:] * math.pi}
    independent = SDPIFlow.independent_targets(target, centers[2][None, None], centers[1][None, None])
    assert torch.allclose(independent['poi_ligand']['tr'][0], velocity[6:9] * scales[1])
    assert torch.allclose(independent['poi_ligand']['rot'][0], velocity[9:12] * math.pi)



def test_joint_point_jacobian_matches_actual_symmetric_update():
    ligand = torch.tensor([[0., 1., .3], [0., 0., 0.], [1., 0., 0.], [2., 1., .5]], dtype=torch.double)
    poi = torch.randn(5, 3, dtype=torch.double) + 5
    centers = torch.stack((torch.zeros(3, dtype=torch.double), poi.mean(0), ligand.mean(0)))
    scales = torch.tensor([3., 5.], dtype=torch.double)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    mask = torch.tensor([False, False, True, False, False, False])
    fragments = torch.tensor([[False, False, True, True]])
    ld = {'edge_index': edges, 'edge_mask': mask, 'mask_rotate': fragments}
    tangent = horizontal_torsion_jacobian(ligand, edges, mask, fragments)
    reversed_tangent = horizontal_torsion_jacobian(ligand, edges.flip(0), mask, ~fragments)
    assert torch.allclose(tangent, reversed_tangent, atol=1e-12)
    j_ligand = point_jacobian(ligand, 2, centers, scales, 13)
    j_ligand[:, :, 12:] = tangent
    j_poi = point_jacobian(poi, 1, centers, scales, 13)
    velocity = torch.randn(13, dtype=torch.double) * .2
    physical = {'rigid_output_mode': INDEPENDENT_RIGID_MODE, 'torsion_update_mode': SYMMETRIC_TORSION_MODE,
                'torsion': velocity[None, 12:] * math.pi,
                'e3_ligand': {'tr': velocity[None, :3] * scales[0], 'rot': velocity[None, 3:6] * math.pi},
                'poi_ligand': {'tr': velocity[None, 6:9] * scales[1], 'rot': velocity[None, 9:12] * math.pi}}
    eps = 1e-7
    moved_ligand, moved_poi, _ = SDPIFlow.differentiable_step(
        ligand[None], poi[None], torch.eye(3, dtype=torch.double)[None], physical,
        torch.tensor([eps], dtype=torch.double), ld)
    for original, moved, jacobian in ((ligand, moved_ligand[0], j_ligand), (poi, moved_poi[0], j_poi)):
        assert torch.allclose((original - moved) / eps, jacobian @ velocity, atol=1e-6), \
            'constraint tangent disagrees with reverse-time rigid/torsion decoder'



def test_constraint_model_equivariance_backward_and_no_torsion():
    fuse, graph, proposals, state = fixture()
    for head in (fuse.constraint_head.contact_head[-1], fuse.constraint_head.torsion_precision[-1],
                 fuse.constraint_head.expert_precision):
        torch.nn.init.normal_(head.weight, std=.03)
    graph['node_coords'].requires_grad_()
    base = fuse(graph, proposals, state)
    rotation = rotation_exp(torch.tensor([[.3, -.5, .2]]))[0]
    transformed = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 8.}
    rotated_proposals = {name: {key: value @ rotation.T if key != 'tor' else value
                               for key, value in branch.items()} for name, branch in proposals.items()}
    moved = fuse(transformed, rotated_proposals, state)
    for name in ('e3_ligand', 'poi_ligand'):
        for key in ('tr', 'rot'):
            assert torch.allclose(moved['normalized_velocity'][name][key],
                                  base['normalized_velocity'][name][key] @ rotation.T, atol=2e-5)
    assert torch.allclose(moved['normalized_velocity']['torsion'], base['normalized_velocity']['torsion'], atol=2e-5)
    values = base['normalized_velocity']
    loss = sum(values[name][key].square().sum() for name in ('e3_ligand', 'poi_ligand') for key in ('tr', 'rot'))
    (loss + values['torsion'].square().sum()).backward()
    assert torch.isfinite(graph['node_coords'].grad).all()
    for module in (fuse.constraint_head.contact_head[-1], fuse.constraint_head.torsion_precision[-1],
                   fuse.constraint_head.expert_precision, fuse.pooling.pair_updates[0][-1]):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0, 'joint solver must train both constraints and their encoder'
    for branch in proposals.values():
        branch['tor'] = torch.empty(1, 0)
    empty = fuse(graph, proposals, state)
    assert empty['normalized_velocity']['torsion'].shape == (1, 0)
    assert empty['consensus_diagnostics']['solved_dofs'] == 12



def test_constraint_checkpoint_is_explicit_and_cannot_load_as_v9():
    from test_training_contract import _endpoint_trainer_fixture
    from tricoopdiff.train import ConsensusTrainer, validate_consensus_checkpoint
    trainer, _, _ = _endpoint_trainer_fixture(profile='ternary_endpoint_v9')
    fuse, _, _, _ = fixture()
    trainer.model.trifuse = fuse
    trainer = ConsensusTrainer(trainer.model, trainer.diffusion, device='cpu')
    # Validator-only payload: use the real profile and solver identity. Normal
    # training/inference save/load with native experts is checked by the smoke.
    payload = trainer.checkpoint_payload()
    assert payload['runtime_config']['constraint_solver_contract'] == 'supervised_terminal_distance_v2'
    validate_consensus_checkpoint(payload)
    for mutation in ('version', 'contract'):
        invalid = copy.deepcopy(payload)
        if mutation == 'version':
            invalid['checkpoint_version'] = 37
        else:
            invalid['runtime_config']['constraint_solver_contract'] = 'unknown'
        try:
            validate_consensus_checkpoint(invalid)
        except ValueError as error:
            assert 'constraint' in str(error)
        else:
            raise AssertionError('mismatched constraint solver contract was accepted')
    try:
        ConsensusFuse(**production_model_config(32, 16, 1, model_profile='ternary_endpoint_v9'))
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('constraint weights silently loaded as the old motion field')



def test_only_constraint_motion_is_constructed_and_source_arms_share_weights():
    configurations = [production_model_config(32, 16, 1, source_conditioning=enabled) for enabled in (False, True)]
    models = []
    for config in configurations:
        torch.manual_seed(91)
        models.append(ConsensusFuse(**config))
    for model in models:
        assert model.config['model_profile'] == CONSTRAINT_MODEL_PROFILE
        assert not any(hasattr(model, name) for name in ('motion_head', 'direct_torsion_head',
            'expert_prior_gate', 'expert_vector_mix', '_direct_motion', '_legacy_motion', '_combine_conditional_prior'))
    shared = models[1].state_dict()
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, shared[name]), 'source conditioning changed common initialization'


def test_constraint_motion_has_no_train_eval_drift():
    model, graph, proposals, state = fixture()
    model.train()
    a = model(graph, proposals, state)['normalized_velocity']
    model.eval()
    b = model(graph, proposals, state)['normalized_velocity']
    for branch in BRANCHES:
        for component in ('tr', 'rot'):
            assert torch.equal(a[branch][component], b[branch][component])
    assert torch.equal(a['torsion'], b['torsion'])


def test_terminal_distance_controller_corrects_both_signs_and_stops_at_goal():
    from tricoopdiff.models.constraint_motion import terminal_distance_rate
    current = torch.tensor([.5, 1., 2.])
    goal = torch.ones(3)
    for time in (0., .05, .5, 1.):
        rate = terminal_distance_rate(current, goal, torch.tensor(time))
        after = current - .05 * rate
        assert rate[0] < 0 and rate[1] == 0 and rate[2] > 0
        assert torch.all((after - goal).abs() <= (current - goal).abs())
        assert rate.abs().max() <= 2. and torch.isfinite(rate).all()


def test_terminal_geometry_supervision_respects_symmetry_and_stops_target_gradients():
    from tricoopdiff.models.constraint_motion import terminal_geometry_loss
    e3 = torch.tensor([[0., 0., 0.], [0., 4., 0.]], requires_grad=True)
    poi = torch.tensor([[4., 0., 0.], [4., 4., 0.]], requires_grad=True)
    ligand = torch.tensor([[1., 1., 1.], [2., 2., 2.]], requires_grad=True)
    targets = (e3, poi, ligand)
    predictions = []
    for left, right in ((2, 0), (2, 1), (1, 0)):
        delta = targets[left][:, None] - targets[right][None]
        distance = (delta.square().sum(-1) + .01).sqrt().detach() / 10.
        predictions.append({'distance': distance.requires_grad_(), 'receiver_type': left,
                            'anchor_type': right, 'anchor_weights': torch.eye(2, requires_grad=True)})
    candidates = torch.stack((ligand.flip(0), ligand))
    assert terminal_geometry_loss(predictions, e3, poi, candidates) == 0
    perturbed = [{**p, 'distance': p['distance'] + .3} for p in predictions]
    loss = terminal_geometry_loss(perturbed, e3, poi, candidates)
    assert loss > 0
    loss.backward()
    assert all(p['distance'].grad is not None and p['distance'].grad.abs().sum() > 0 for p in predictions)
    assert all(p['anchor_weights'].grad is None for p in predictions)
    assert all(p.grad is None for p in targets), 'teacher structure must never receive model gradients'
    rotation = rotation_exp(torch.tensor([[.2, -.4, .1]]))[0]
    moved_loss = terminal_geometry_loss(perturbed, e3 @ rotation.T + 7.,
        poi @ rotation.T + 7., candidates @ rotation.T + 7.)
    assert torch.allclose(loss, moved_loss, atol=1e-7)


def test_target_geometry_does_not_consume_native_labels_and_has_no_free_torsion_head():
    from tricoopdiff.models.constraint_motion import terminal_geometry_loss
    fuse, graph, proposals, state = fixture()
    assert not hasattr(fuse.constraint_head, 'torsion_head')
    baseline = fuse(graph, proposals, state)
    poisoned = fuse({**graph, 'native_coords': torch.full_like(graph['node_coords'], float('nan'))},
                    proposals, state)
    for left, right in zip(baseline['target_geometry'], poisoned['target_geometry']):
        assert torch.equal(left['distance'], right['distance'])
        assert not left['anchor_weights'].requires_grad
    native = [graph['node_coords'][graph['node_types'] == kind] for kind in range(3)]
    loss = terminal_geometry_loss(baseline['target_geometry'], native[0], native[1], native[2][None])
    loss.backward()
    assert fuse.constraint_head.contact_head[-1].weight.grad.abs().sum() > 0
    assert fuse.constraint_head.expert_precision.weight.grad is None, \
        'reliability must not be able to reduce the terminal geometry supervision'


def test_uniform_reliability_changes_do_not_control_speed():
    fuse, graph, proposals, state = fixture()
    before = fuse(graph, proposals, state)['normalized_velocity']
    # All reliability logits are initialized identically. Change their common
    # scale without changing the terminal distances or binary suggestions.
    with torch.no_grad():
        fuse.constraint_head.contact_head[-1].bias[1].fill_(2.)
        fuse.constraint_head.expert_precision.bias.fill_(2.)
        fuse.constraint_head.torsion_precision[-1].bias.fill_(2.)
    after = fuse(graph, proposals, state)['normalized_velocity']
    for branch in BRANCHES:
        for component in ('tr', 'rot'):
            assert torch.allclose(before[branch][component], after[branch][component], atol=1e-6)
    assert torch.allclose(before['torsion'], after['torsion'], atol=1e-6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')
