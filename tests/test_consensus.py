"""Nonzero-residual equivariance, proposal isolation, gradients and EMA checks."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.models.consensus import ConsensusFuse, BRANCHES
from tricoopdiff.train import validate_consensus_checkpoint
from tricoopdiff.geometry.rigid import rotation_exp as so3_exp
from tricoopdiff.diffdock.utils.torus import SCORE_NORM_VERSION


def fixture():
    torch.manual_seed(71)
    graph = {'node_features': torch.randn(15, 16), 'node_coords': torch.randn(15, 3),
             'node_types': torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
             'lig_edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             'lig_edge_mask': torch.tensor([False, False, True, False, False, False]),
             'lig_mask_rotate': torch.tensor([[False, False, True, True]])}
    proposals = {b: {c: torch.randn(1, 1 if c == 'tor' else 3)
                      for c in ('tr', 'rot', 'tor')} for b in BRANCHES}
    fuse = ConsensusFuse(node_dim=32, input_dim=16, num_transformer_layers=1, dropout=0).eval()
    return fuse, graph, proposals, torch.randn(1, 9)


def test_zero_residual_preserves_each_interface_prior():
    fuse, graph, proposals, state = fixture()
    original = copy.deepcopy(proposals)
    result = fuse(graph, proposals, state)
    for branch in BRANCHES:
        for component in ('tr', 'rot'):
            assert torch.allclose(result['normalized_velocity'][branch][component],
                                  torch.sigmoid(torch.tensor(2.)) * proposals[branch][component])
        for component in ('tr', 'rot', 'tor'):
            assert torch.equal(proposals[branch][component], original[branch][component])
    assert result['residual_penalty'] == 0
    assert result['rank_features'].shape == (1, 32)


def test_nonzero_residual_is_se3_equivariant():
    fuse, graph, proposals, state = fixture()
    torch.nn.init.normal_(fuse.residual_head.weight, std=0.1)
    torch.nn.init.normal_(fuse.gate_head.weight, std=0.1)
    base = fuse(graph, proposals, state)
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    moved_graph = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 10}
    moved_proposals = {b: {c: v @ rotation.T if c != 'tor' else v
                           for c, v in scores.items()} for b, scores in proposals.items()}
    moved = fuse(moved_graph, moved_proposals, state)
    assert torch.allclose(base['rank_features'], moved['rank_features'], atol=2e-5)
    for b in BRANCHES:
        for c in ('tr', 'rot'):
            assert torch.allclose(moved['normalized_velocity'][b][c],
                                  base['normalized_velocity'][b][c] @ rotation.T,
                                  atol=2e-5), (b, c)
    assert torch.allclose(base['normalized_velocity']['torsion'], moved['normalized_velocity']['torsion'], atol=2e-5)


def test_residual_heads_and_gates_receive_supervision():
    fuse, graph, proposals, state = fixture()
    result = fuse(graph, proposals, state)
    loss = sum((result['normalized_velocity'][b][c] - 1).square().mean()
               for b in BRANCHES for c in ('tr', 'rot'))
    loss = loss + (result['normalized_velocity']['torsion'] - 1).square().mean()
    loss.backward()
    for module in (fuse.gate_head, fuse.residual_head, fuse.torsion_coordinator.net[-1]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
        assert torch.isfinite(module.weight.grad).all()


def test_source_conditioning_distinguishes_paths_and_receives_gradient():
    _, graph, proposals, state = fixture()
    fuse = ConsensusFuse(
        node_dim=32, input_dim=16, num_transformer_layers=1,
        dropout=0, source_conditioning=True,
    ).eval()
    sdpi = fuse(graph, proposals, state, source='sdpi')
    bitseed = fuse(graph, proposals, state, source='bitseed')
    assert not torch.allclose(sdpi['rank_features'], bitseed['rank_features'])
    loss = sum(value.square().mean() for branch in BRANCHES
               for value in bitseed['normalized_velocity'][branch].values())
    loss = loss + bitseed['normalized_velocity']['torsion'].square().mean()
    loss.backward()
    gradient = fuse.source_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[1].abs().sum() > 0
    assert gradient[0].abs().sum() == 0


def test_unconditioned_arm_accepts_both_source_labels_without_changing_output():
    fuse, graph, proposals, state = fixture()
    sdpi = fuse(graph, proposals, state, source=0)
    bitseed = fuse(graph, proposals, state, source=1)
    assert torch.equal(sdpi['rank_features'], bitseed['rank_features'])


def test_b_and_c_share_identical_common_parameter_initialization():
    torch.manual_seed(991)
    arm_b = ConsensusFuse(
        node_dim=32, input_dim=16, num_transformer_layers=1, dropout=0,
        source_conditioning=False,
    )
    torch.manual_seed(991)
    arm_c = ConsensusFuse(
        node_dim=32, input_dim=16, num_transformer_layers=1, dropout=0,
        source_conditioning=True,
    )
    state_b, state_c = arm_b.state_dict(), arm_c.state_dict()
    common = [name for name in state_b if not name.startswith('source_')]
    assert common
    assert all(torch.equal(state_b[name], state_c[name]) for name in common)


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


def test_no_torsion_has_finite_empty_output():
    fuse, graph, proposals, state = fixture()
    graph['lig_edge_mask'] = torch.zeros(6, dtype=torch.bool)
    graph['lig_mask_rotate'] = torch.zeros(0, 4, dtype=torch.bool)
    for branch in BRANCHES:
        proposals[branch]['tor'] = torch.empty(1, 0)
    output = fuse(graph, proposals, state)
    assert output['normalized_velocity']['torsion'].shape == (1, 0)
    assert torch.isfinite(output['residual_penalty'])


def test_protein_pair_head_warmstart_is_transparent_and_trainable():
    legacy, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**legacy.config, protein_pair_motion=True).eval()
    missing, unexpected = fuse.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith('protein_pair_head.') for k in missing)
    base, output = legacy(graph, proposals, state), fuse(graph, proposals, state)
    for branch in BRANCHES:
        for component in ('tr', 'rot'):
            assert torch.equal(base['normalized_velocity'][branch][component],
                               output['normalized_velocity'][branch][component])
    assert 'protein_pair_motion' not in legacy.config
    assert fuse.config['protein_pair_motion'] is True
    loss = sum((output['normalized_velocity'][b][c] - 1).square().mean()
               for b in BRANCHES for c in ('tr', 'rot'))
    loss.backward()
    grad = fuse.protein_pair_head[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_protein_pair_motion_is_equivariant_and_independent_of_expert_vectors():
    legacy, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**legacy.config, protein_pair_motion=True).eval()
    for branch in BRANCHES:
        for component in ('tr', 'rot', 'tor'):
            proposals[branch][component].zero_()
    torch.nn.init.normal_(fuse.protein_pair_head[-1].weight, std=.1)
    graph['node_coords'].requires_grad_(True)
    base = fuse(graph, proposals, state)
    assert base['normalized_velocity']['poi_ligand']['tr'].norm() > 1e-6
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    moved = fuse({**graph, 'node_coords': graph['node_coords'] @ rotation.T + 10}, proposals, state)
    for branch in BRANCHES:
        for component in ('tr', 'rot'):
            assert torch.allclose(moved['normalized_velocity'][branch][component],
                                  base['normalized_velocity'][branch][component] @ rotation.T,
                                  atol=2e-5), (branch, component)
    base['normalized_velocity']['poi_ligand']['tr'].square().sum().backward()
    grad = graph['node_coords'].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad[graph['node_types'] == 1].abs().sum() > 0


def test_frozen_expert_context_preserves_input_derivatives_and_outer_no_grad():
    from tricoopdiff.models.consensus import ConsensusCoopDiff

    fuse, _, _, _ = fixture()
    expert = torch.nn.Linear(3, 3)
    expert.num_conv_layers = 1
    model = ConsensusCoopDiff(expert, fuse)
    x = torch.randn(2, 3, requires_grad=True)
    with model._expert_context():
        y = model.expert(x)
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in expert.parameters())
    with torch.no_grad(), model._expert_context():
        assert not model.expert(x).requires_grad
    model.expert_input_grad = False
    with model._expert_context():
        assert not model.expert(x).requires_grad


def test_checkpointed_expert_matches_direct_coordinate_gradient():
    from torch_geometric.data import HeteroData
    from tricoopdiff.models.consensus import ConsensusCoopDiff

    class Expert(torch.nn.Module):
        num_conv_layers = 1

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, graph):
            x = graph['ligand'].pos.mean(0) - graph['receptor'].pos.mean(0)
            y = self.linear(x).sin().unsqueeze(0)
            graph['ligand'].hidden = y
            return y, y.square(), y[:, :1]

    fuse, _, _, _ = fixture()
    model = ConsensusCoopDiff(Expert(), fuse)
    graph = HeteroData()
    graph['ligand'].pos = torch.randn(4, 3, requires_grad=True)
    graph['receptor'].pos = torch.randn(5, 3, requires_grad=True)
    checkpointed = model._read_expert(graph)
    inputs = (graph['ligand'].pos, graph['receptor'].pos)
    grad_checkpointed = torch.autograd.grad(sum(x.square().sum() for x in checkpointed), inputs)
    direct = model.expert(graph.clone())
    grad_direct = torch.autograd.grad(sum(x.square().sum() for x in direct), inputs)
    assert all(torch.equal(a, b) for a, b in zip(checkpointed, direct))
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(grad_checkpointed, grad_direct))
    assert all(p.grad is None for p in model.expert.parameters())
    assert 'hidden' not in graph['ligand'], 'recomputation must not mutate the caller graph'


def test_direct_motion_is_identity_without_an_additive_expert_prior():
    from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE

    legacy, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**legacy.config, direct_ternary_motion=True).eval()
    for branch in BRANCHES:
        for component in ('tr', 'rot', 'tor'):
            proposals[branch][component] *= 1000
    output = fuse(graph, proposals, state)
    predicted = output['normalized_velocity']
    assert predicted['rigid_output_mode'] == INDEPENDENT_RIGID_MODE
    assert all(torch.count_nonzero(predicted[b][c]) == 0
               for b in BRANCHES for c in ('tr', 'rot'))
    assert torch.count_nonzero(predicted['torsion']) == 0
    assert output['residual_penalty'] == 0
    assert not hasattr(fuse, 'residual_head'), 'direct motion must not instantiate an unused residual head'


def test_direct_motion_uses_joint_geometry_and_is_equivariant():
    legacy, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**legacy.config, direct_ternary_motion=True,
                         global_node_context=True, spatial_geometry=True,
                         motion_attention=True, global_node_attention=True).eval()
    for update in fuse.pooling.global_updates:
        torch.nn.init.normal_(update[-1].weight, std=.03)
    for attention in fuse.pooling.global_attention:
        torch.nn.init.normal_(attention.out_proj.weight, std=.03)
    torch.nn.init.normal_(fuse.motion_head[-1].weight, std=.1)
    torch.nn.init.normal_(fuse.direct_torsion_head[-1].weight, std=.1)
    # Nonzero motion is possible even when all frozen expert proposals vanish.
    proposals = {b: {c: torch.zeros_like(v) for c, v in p.items()}
                 for b, p in proposals.items()}
    graph['node_coords'] = graph['node_coords'].requires_grad_()
    output = fuse(graph, proposals, state)['normalized_velocity']
    restored = ConsensusFuse(**fuse.config).eval()
    restored.load_state_dict(fuse.state_dict(), strict=True)
    restored_output = restored(graph, proposals, state)['normalized_velocity']
    assert restored.pooling.global_node_context
    for b in BRANCHES:
        for c in ('tr', 'rot'):
            assert torch.equal(restored_output[b][c], output[b][c])
    assert sum(output[b][c].abs().sum() for b in BRANCHES for c in ('tr', 'rot')) > 0
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    moved = fuse({**graph, 'node_coords': graph['node_coords'] @ rotation.T + 10},
                 proposals, state)['normalized_velocity']
    for b in BRANCHES:
        for c in ('tr', 'rot'):
            assert torch.allclose(moved[b][c], output[b][c] @ rotation.T, atol=2e-5), (b, c)
    assert torch.allclose(moved['torsion'], output['torsion'], atol=2e-5)
    loss = sum(output[b][c].square().sum() for b in BRANCHES for c in ('tr', 'rot'))
    loss = loss + output['torsion'].square().sum()
    loss.backward()
    for module in (fuse.pooling.messages[0], fuse.pooling.messages[1], fuse.motion_head,
                   *fuse.pooling.global_updates, *fuse.pooling.global_attention,
                   *fuse.pooling.global_attention_norms):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert torch.isfinite(graph['node_coords'].grad).all()


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


def test_interface_attention_is_not_biased_by_partner_node_count():
    from tricoopdiff.models.consensus import attentive_interface_motion

    weights = torch.tensor([[2., 3., 0., 0.], [4., 5., 0., 0.]], requires_grad=True)
    direction = torch.eye(3)[:2]
    torque = direction.roll(1, dims=1)
    types = torch.tensor([0, 2])
    reference = attentive_interface_motion(weights, direction, torque, types)
    indices = torch.tensor([0] + [1] * 10)
    repeated = attentive_interface_motion(weights[indices], direction[indices], torque[indices], types[indices])
    assert all(torch.allclose(a, b) for a, b in zip(reference, repeated))
    sum(value.square().sum() for value in reference).backward()
    assert torch.isfinite(weights.grad).all()


def test_checkpointed_ternary_preserves_dropout_outputs_and_gradients():
    from torch.utils.checkpoint import checkpoint

    legacy, graph, proposals, state = fixture()
    config = {**legacy.config, 'dropout': .2, 'direct_ternary_motion': True,
              'global_node_context': True, 'spatial_geometry': True, 'motion_attention': True,
              'global_node_attention': True}
    plain = ConsensusFuse(**config).train()
    for head in (plain.motion_head, plain.direct_torsion_head):
        torch.nn.init.normal_(head[-1].weight, std=.03)
    recomputed = copy.deepcopy(plain)
    graph['node_coords'].requires_grad_()
    copied_graph = {**graph, 'node_coords': graph['node_coords'].detach().clone().requires_grad_()}
    def loss(output):
        velocities = output['normalized_velocity']
        return sum(velocities[b][c].square().sum() for b in BRANCHES for c in ('tr', 'rot')) + velocities['torsion'].square().sum()
    torch.manual_seed(91)
    expected = plain(graph, proposals, state)
    loss(expected).backward()
    torch.manual_seed(91)
    actual = checkpoint(recomputed, copied_graph, proposals, state, use_reentrant=False)
    loss(actual).backward()
    assert torch.equal(loss(actual), loss(expected))
    assert torch.allclose(graph['node_coords'].grad, copied_graph['node_coords'].grad, atol=1e-6)
    for left, right in zip(plain.parameters(), recomputed.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5)


def test_production_profile_has_same_train_and_eval_motion():
    from tricoopdiff.models.consensus import production_model_config

    _, graph, proposals, state = fixture()
    fuse = ConsensusFuse(**production_model_config(32, 16, 1))
    for head in (fuse.motion_head, fuse.direct_torsion_head):
        torch.nn.init.normal_(head[-1].weight, std=.03)
    training = fuse.train()(graph, proposals, state)['normalized_velocity']
    with torch.no_grad():
        evaluation = fuse.eval()(graph, proposals, state)['normalized_velocity']
    for b in BRANCHES:
        for c in ('tr', 'rot'):
            assert torch.allclose(training[b][c], evaluation[b][c], atol=2e-6, rtol=1e-5)
    assert torch.allclose(training['torsion'], evaluation['torsion'], atol=2e-6, rtol=1e-5)
    assert fuse.config['dropout'] == 0 and fuse.rotation_scale_normalization
    restored = ConsensusFuse(**fuse.config)
    restored.load_state_dict(fuse.state_dict(), strict=True)
    try:
        ConsensusFuse(**{**fuse.config, 'dropout': .1})
    except ValueError as error:
        assert 'production model profile' in str(error)
    else:
        raise AssertionError('a named production profile must not silently enable dropout')


def test_rotation_levers_are_scale_normalized_and_have_finite_degenerate_gradients():
    from tricoopdiff.models.consensus import rotation_levers

    coords = torch.tensor([[2., 0., 0.], [-2., 0., 0.], [0., 2., 1.]])
    normalized = rotation_levers(coords, True)
    assert torch.allclose(normalized, rotation_levers(coords * 10 + 20, True), atol=1e-6)
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    assert torch.allclose(rotation_levers(coords @ rotation.T + 10, True),
                          normalized @ rotation.T, atol=1e-6)
    degenerate = torch.zeros(3, 3, requires_grad=True)
    rotation_levers(degenerate, True).square().sum().backward()
    assert torch.isfinite(degenerate.grad).all()


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


def test_direct_model_contains_only_active_heads_and_migrates_old_weights():
    from tricoopdiff.models.consensus import production_model_config
    model = ConsensusFuse(**production_model_config(32, 16, 1, True))
    obsolete = ('gate_head', 'residual_head', 'source_gate_head',
                'torsion_coordinator', 'judge_head', 'state_head')
    assert not any(name.startswith(tuple(prefix + '.' for prefix in obsolete))
                   for name in model.state_dict())
    state = copy.deepcopy(model.state_dict())
    historical = dict(state)
    for name in obsolete:
        historical[name + '.unused.weight'] = torch.randn(2, 3)
    model.load_state_dict(historical, strict=True)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    parent = torch.nn.Module()
    parent.add_module('trifuse', model)
    parent.load_state_dict({'trifuse.' + name: value for name, value in historical.items()}, strict=True)
    assert len(model.active_state_dict(historical)) == len(state)
    historical['unknown.weight'] = torch.randn(2, 3)
    try:
        model.load_state_dict(historical, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('migration must not hide genuinely unknown weights')


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


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')
