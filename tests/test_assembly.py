"""Ternary constraints, joint geometric solving and equivariant gradients."""

import copy
import sys
from pathlib import Path

import torch


def test_rigid_body_output_heads_are_parameter_independent():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    torch.manual_seed(9)
    decoder = LegalPoseDecoder(8, 4).double()
    h = torch.randn(6, 8, dtype=torch.double)
    coords, vectors = torch.randn(6, 3, dtype=torch.double), torch.randn(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    state = torch.zeros(1, 9, dtype=torch.double)
    state[:, -1] = 1.
    output = decoder(h, vectors, coords, types, torch.empty(2, 0, dtype=torch.long),
                     torch.zeros(0, 2, dtype=torch.bool), state, [torch.empty(0), torch.empty(0)])
    for own, other in (BRANCHES, BRANCHES[::-1]):
        own_parameters = list(decoder.rigid[own].parameters())
        other_parameters = list(decoder.rigid[other].parameters())
        loss = sum(output[own][field].square().sum() for field in ('tr', 'rot'))
        gradients = torch.autograd.grad(loss, own_parameters + other_parameters,
                                        retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients[:len(own_parameters)])
        assert all(g is None for g in gradients[len(own_parameters):]), (
            'one-step rigid heads must not share parameters across physical bodies')


def test_shared_rigid_migration_duplicates_values_not_parameter_storage():
    model = TernaryAssembly(node_dim=32, input_dim=16)
    legacy = dict(model.state_dict())
    for layer in ('0', '2'):
        for suffix in ('weight', 'bias'):
            prefix = 'controller.assembly.rigid'
            value = legacy.pop(f'{prefix}.{BRANCHES[0]}.{layer}.{suffix}')
            legacy.pop(f'{prefix}.{BRANCHES[1]}.{layer}.{suffix}')
            legacy[f'{prefix}.{layer}.{suffix}'] = value
    migrated = model.migrate_shared_rigid_weights(legacy)
    model.load_state_dict(migrated, strict=True)
    for left, right in zip(model.controller.assembly.rigid[BRANCHES[0]].parameters(),
                           model.controller.assembly.rigid[BRANCHES[1]].parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr(), 'migration must not tie the new body heads'


def test_matcha_evidence_preserves_complete_frame_and_equivariance():
    from tricoopdiff.models.expert import FrozenMatcha

    torch.manual_seed(17)
    receptor = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    ligand = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    frame = torch.eye(3, dtype=torch.double, requires_grad=True)
    vectors = FrozenMatcha.node_vectors(receptor, ligand, frame)
    assert vectors.shape == (11, FrozenMatcha.node_vector_dim, 3)
    assert torch.equal(vectors[:, 2:], frame.T[None].expand(11, -1, -1)), (
        'canonical-frame tokens need all three basis axes, not just the first')
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]], dtype=torch.double))[0]
    moved = FrozenMatcha.node_vectors(receptor @ rotation.T + 17.,
                                     ligand @ rotation.T + 17., rotation @ frame)
    assert torch.allclose(moved, vectors @ rotation.T, atol=1e-12)
    weights = torch.randn_like(vectors)
    gradients = torch.autograd.grad((vectors * weights).sum(), (receptor, ligand, frame))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert (gradients[2].abs().sum(0) > 0).all(), 'each local frame axis must remain differentiable'


def test_logit_only_weight_migration_preserves_logits_and_zeros_new_values():
    model = TernaryAssembly(node_dim=32, input_dim=16)
    current = model.state_dict()
    legacy = {key: value.clone() for key, value in current.items()}
    heads = [('controller.structure.geometry.2', 1),
             ('controller.assembly.distance.2', 1)]
    heads.extend((f'controller.layers.{i}.geometry.2', layer.heads)
                 for i, layer in enumerate(model.controller.layers))
    for prefix, count in heads:
        for suffix in ('weight', 'bias'):
            key = f'{prefix}.{suffix}'
            legacy[key] = legacy[key][-count:].clone()
    migrated = model.migrate_logit_only_geometry(legacy)
    model.load_state_dict(migrated, strict=True)
    for prefix, count in heads:
        for suffix in ('weight', 'bias'):
            key = f'{prefix}.{suffix}'
            assert torch.equal(migrated[key][-count:], legacy[key])
            assert torch.count_nonzero(migrated[key][:-count]) == 0
    malformed = dict(legacy)
    malformed['controller.structure.geometry.2.weight'] = torch.empty(2, 32)
    try:
        model.migrate_logit_only_geometry(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('migration must reject unrelated shape mismatches')


def test_rigid_decoder_reads_distance_with_identical_node_identities():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    torch.manual_seed(3)
    decoder = LegalPoseDecoder(8, 4).double()
    h = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    vectors = torch.randn(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    state = torch.zeros(1, 9, dtype=torch.double)
    state[:, -1] = 1.
    output = decoder(h, vectors, coords, types, torch.empty(2, 0, dtype=torch.long),
                     torch.zeros(0, 2, dtype=torch.bool), state,
                     [torch.empty(0), torch.empty(0)])
    loss = sum(output[branch][field].square().sum()
               for branch in BRANCHES for field in ('tr', 'rot'))
    gradient = torch.autograd.grad(loss, decoder.distance[-1].weight)[0]
    assert gradient.norm() > 1e-8, (
        'rigid control must read geometric values even when atom identities coincide')


def test_interaction_geometry_reaches_scalar_values_with_repeated_identities():
    from tricoopdiff.models.assembly import EquivariantInteraction

    torch.manual_seed(2)
    layer = EquivariantInteraction(8, 4).double()
    h = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    vectors = torch.zeros(6, 4, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    scalar, _ = layer._messages(layer.query(h), vectors, coords, layer.key(h),
                               layer.value(h), vectors, coords, types,
                               torch.zeros(6, 6, 4, dtype=torch.double))
    gradient = torch.autograd.grad(scalar.square().sum(), layer.geometry[-1].weight)[0]
    assert gradient.norm() > 1e-6, (
        'scalar geometry must not vanish merely because source identity values coincide')


def test_protein_structure_reads_distances_with_identical_node_identities():
    from tricoopdiff.models.assembly import ComponentStructureEncoder

    torch.manual_seed(1)
    encoder = ComponentStructureEncoder(8).double()
    identity = torch.randn(1, 8, dtype=torch.double).expand(6, -1)
    coords = torch.randn(6, 3, dtype=torch.double)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    edges = torch.tensor([[4, 5], [5, 4]])
    attributes = torch.zeros(2, 4, dtype=torch.double)
    original = encoder(identity, coords, types, edges, attributes)
    changed_coords = coords.clone()
    changed_coords[0] *= 4
    changed = encoder(identity, changed_coords, types, edges, attributes)
    assert not torch.allclose(original[:2], changed[:2], atol=1e-8), (
        'protein geometry must survive even when attention values have identical identities')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.models.assembly import TernaryAssembly, BRANCHES
from tricoopdiff.train import validate_assembly_checkpoint
from tricoopdiff.geometry.rigid import rotation_exp as so3_exp
from tricoopdiff.diffdock.utils.torus import SCORE_NORM_VERSION


def fixture():
    torch.manual_seed(71)
    graph = {'node_features': torch.randn(15, 16), 'node_coords': torch.randn(15, 3),
             'node_types': torch.tensor([0] * 5 + [1] * 6 + [2] * 4),
             'lig_edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
             'lig_edge_mask': torch.tensor([False, False, True, False, False, False]),
             'lig_mask_rotate': torch.tensor([[False, False, True, True]]),
             'lig_edge_attr': torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 4).float(),
             'expert_node_features': torch.cat((torch.randn(15, 116), torch.ones(15, 2)), -1),
             'expert_node_vectors': torch.randn(15, 40, 3)}
    proposals = {b: {c: torch.randn(1, 1 if c == 'tor' else 3)
                      for c in ('tr', 'rot', 'tor')} for b in BRANCHES}
    fuse = TernaryAssembly(node_dim=32, input_dim=16).eval()
    state = torch.zeros(1, 9)
    state[:, 5:8], state[:, -1] = .05, 1.
    return fuse, graph, proposals, state


def test_v8_unified_controller_equivariance_and_gradient_paths():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = fixture()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).train()
    assert not hasattr(fuse, 'pooling'), 'new controller must not retain summary control modules'
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['node_coords'].requires_grad_()
    base = fuse(graph, proposals, state)
    rotation = so3_exp(torch.tensor([[.2, -.5, .7]]))[0]
    moved_graph = {**graph,
                   'node_coords': graph['node_coords'] @ rotation.T + 17.,
                   'expert_node_vectors': graph['expert_node_vectors'] @ rotation.T}
    moved_proposals = {branch: {key: (value @ rotation.T if key != 'tor' else value)
                                for key, value in proposal.items()}
                       for branch, proposal in proposals.items()}
    moved = fuse(moved_graph, moved_proposals, state)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(moved['normalized_velocity'][branch][field],
                                  base['normalized_velocity'][branch][field] @ rotation.T,
                                  atol=4e-5), (branch, field)
    assert torch.allclose(base['normalized_velocity']['torsion'],
                          moved['normalized_velocity']['torsion'], atol=4e-5)
    assert torch.allclose(base['rank_features'], moved['rank_features'], atol=4e-5)
    output = base['normalized_velocity']
    loss = sum(output[b][c].square().sum() for b in BRANCHES for c in ('tr', 'rot'))
    loss = loss + output['torsion'].square().sum()
    loss.backward()
    for key in ('expert_node_vectors', 'node_coords'):
        gradient = graph[key].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0, key
    for name, parameter in fuse.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_conditional_node_controller_does_not_consume_binary_rigid_advice():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['expert_node_features'].requires_grad_()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2)).eval()
    base = fuse(graph, proposals, state)['normalized_velocity']
    changed = {branch: {**proposal, 'tr': proposal['tr'] * 100 + 17,
                       'rot': proposal['rot'] * -100 - 23}
               for branch, proposal in proposals.items()}
    alternate = fuse(graph, changed, state)['normalized_velocity']
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.equal(base[branch][field], alternate[branch][field]), (
                'binary rigid advice must not bypass conditional node evidence')
    assert torch.equal(base['torsion'], alternate['torsion'])
    loss = sum(base[branch][field].square().sum()
               for branch in BRANCHES for field in ('tr', 'rot'))
    gradients = torch.autograd.grad(loss, (graph['expert_node_features'], graph['expert_node_vectors']))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients), (
        'removing rigid advice must preserve differentiable conditional binary evidence')


def test_v8_distant_component_and_atom_identity_reach_actions():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = fixture()
    graph['node_coords'][:5] += 200.
    graph['expert_node_vectors'] = torch.randn(15, 4, 3, requires_grad=True)
    graph['expert_node_features'].requires_grad_()
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).eval()
    output = fuse(graph, proposals, state)['normalized_velocity']
    for action in (output['poi_ligand']['tr'], output['torsion']):
        gradients = torch.autograd.grad(action.square().sum(),
                                         (graph['expert_node_vectors'], graph['expert_node_features']),
                                         retain_graph=True)
        for gradient in gradients:
            assert gradient[:5].abs().sum() > 0, 'distant E3 information must reach POI and torsion'
    swapped = graph['expert_node_features'].detach().clone()
    swapped[[-1, -2]] = swapped[[-2, -1]]
    changed = fuse({**graph, 'expert_node_features': swapped}, proposals, state)['normalized_velocity']
    assert not torch.allclose(output['torsion'], changed['torsion'], atol=1e-9), (
        'bond control must retain the association between atom identity and its geometric lever')


def test_v8_node_permutation_and_empty_torsions():
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE, production_model_config

    _, graph, proposals, state = fixture()
    graph['expert_node_vectors'] = torch.randn(15, 4, 3)
    fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE, expert_vector_dim=2,
    )).eval()
    base = fuse(graph, proposals, state)['normalized_velocity']
    permutation = torch.tensor([4, 2, 0, 3, 1, 10, 8, 6, 9, 7, 5, 14, 12, 11, 13])
    changed = {**graph}
    for key in ('node_features', 'node_coords', 'node_types',
                'expert_node_features', 'expert_node_vectors'):
        changed[key] = graph[key][permutation]
    ligand_permutation = permutation[11:] - 11
    inverse = torch.argsort(ligand_permutation)
    changed['lig_edge_index'] = inverse[graph['lig_edge_index']]
    changed['lig_mask_rotate'] = graph['lig_mask_rotate'][:, ligand_permutation]
    result = fuse(changed, proposals, state)['normalized_velocity']
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(base[branch][field], result[branch][field], atol=1e-6)
    assert torch.allclose(base['torsion'], result['torsion'], atol=1e-6)
    empty_proposals = {b: {**p, 'tor': torch.zeros(1, 0)} for b, p in proposals.items()}
    assert fuse(graph, empty_proposals, state)['normalized_velocity']['torsion'].shape == (1, 0), (
        'no_torsion experts must disable torsions even on a flexible molecular graph')
    graph['lig_edge_mask'] = torch.zeros_like(graph['lig_edge_mask'])
    graph['lig_mask_rotate'] = torch.zeros(0, 4, dtype=torch.bool)
    proposals = {b: {**p, 'tor': torch.zeros(1, 0)} for b, p in proposals.items()}
    assert fuse(graph, proposals, state)['normalized_velocity']['torsion'].shape == (1, 0)


def test_v8_chunked_attention_matches_direct_backward():
    from tricoopdiff.models.assembly import EquivariantInteraction

    torch.manual_seed(128)
    layer = EquivariantInteraction(16, 4)
    h = torch.randn(69, 16, requires_grad=True)
    vectors = torch.randn(69, 4, 3, requires_grad=True)
    coords = torch.randn(69, 3, requires_grad=True)
    types = torch.arange(69) % 3
    edges = torch.zeros(2, 0, dtype=torch.long)
    attributes = torch.zeros(0, 4)
    layer.train()
    chunked = layer(h, vectors, coords, types, edges, attributes)
    chunked_grad = torch.autograd.grad(sum(x.square().sum() for x in chunked),
                                       (h, vectors, coords), retain_graph=True)
    full = layer._messages(layer.query(h), vectors, coords, layer.key(h), layer.value(h),
                           vectors, coords, types, h.new_zeros(69, 69, 4))
    expected_vectors = vectors + layer.vector_mix(full[1].transpose(1, 2)).transpose(1, 2)
    expected_vectors = expected_vectors / (1. + expected_vectors.square().mean((1, 2), keepdim=True)).sqrt()
    norms = expected_vectors.square().sum(-1).clamp_min(1e-12).sqrt()
    expected_h = layer.norm(h + layer.scalar_mix(torch.cat((h, full[0], norms), -1)))
    expected_grad = torch.autograd.grad(expected_h.square().sum() + expected_vectors.square().sum(),
                                        (h, vectors, coords))
    for actual, expected in zip(chunked, (expected_h, expected_vectors)):
        assert torch.allclose(actual, expected, atol=2e-6)
    for actual, expected in zip(chunked_grad, expected_grad):
        assert torch.allclose(actual, expected, atol=2e-5), 'chunking changed action gradients'


def test_v8_expert_node_identity_frame_and_scatter_gradients():
    from types import SimpleNamespace
    from tricoopdiff.models.assembly import TernaryAssemblyModel, production_model_config

    class FeatureExpert(torch.nn.Module):
        node_scalar_dim = 2
        node_vector_dim = 2
        num_conv_layers = 1

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(dynamic_max_cross=True)

        def forward_features(self, graph, *, include_vectors=False):
            coords = torch.cat((graph['ligand'].pos, graph['receptor'].pos))
            identities = torch.cat((graph['ligand'].x[:, 0], graph['receptor'].x[:, 0]))
            scalars = torch.stack((coords.square().sum(-1), identities), -1)
            zero = coords.new_zeros(1, 3)
            result = (zero, zero, coords.new_zeros(1), scalars)
            if include_vectors:
                result += (coords[:, None, :].expand(-1, self.node_vector_dim, -1),)
            return result

    positions = torch.zeros(12, 3)
    positions[:, 0] = torch.tensor(list(range(9)) + [100, 101, 102])
    features = torch.zeros(1, 12, 16)
    features[0, :, 0] = torch.arange(12)
    edges = torch.stack((torch.arange(11), torch.arange(1, 12)))
    edges = torch.cat((edges, edges.flip(0)), -1)
    e3 = {'coords': positions[None].clone().requires_grad_(),
          'features': features, 'edge_index': edges}
    poi = {'coords': (positions[None] + torch.tensor([0., 2., 0.])).requires_grad_(),
           'features': features + 100, 'edge_index': edges}
    _, graph, _, state = fixture()
    state[:, -1] = .5
    ligand = {'coords': torch.tensor([[[0., 0., 0.], [1., 1., 0.],
                                       [2., 0., 1.], [3., 1., 1.]]], requires_grad=True),
              'features': torch.zeros(1, 4, 16), 'edge_index': graph['lig_edge_index'],
              'edge_attr': graph['lig_edge_attr'], 'edge_mask': graph['lig_edge_mask'],
              'mask_rotate': graph['lig_mask_rotate']}
    from tricoopdiff.models.assembly import VECTOR_EXPERT_MODEL_PROFILE
    vector_fuse = TernaryAssembly(**production_model_config(
        32, 16, model_profile=VECTOR_EXPERT_MODEL_PROFILE,
        expert_scalar_dim=2, expert_vector_dim=2)).eval()
    vector_expert = FeatureExpert()
    vector_model = TernaryAssemblyModel(vector_expert, vector_fuse)
    assert vector_expert.model.smooth_cross_edges, 'v8 requires smooth DiffDock cross-edge support'
    assert not getattr(vector_expert.model, 'smooth_edges', False), 'v8 must preserve DiffDock internal edge weights'
    angle = torch.tensor([[.3, -.2, .7]])
    frame = so3_exp(angle)
    vector_capture = []
    hook = vector_fuse.register_forward_pre_hook(lambda module, args: vector_capture.append(args[0]))
    try:
        vector_model(e3, poi, ligand, torch.tensor([.5]), r_e3_to_poi=frame,
                     receptor_crop_cutoff=torch.tensor([4.]), diffusion_state=state)
    finally:
        hook.remove()
    vector_reads = vector_capture[0]['expert_node_vectors']
    e3_vectors, poi_vectors = vector_reads[:12], vector_reads[12:24]
    expert_scalar_reads = vector_capture[0]['expert_node_features']
    e3_visible = expert_scalar_reads[:12, 4].bool()
    poi_visible = expert_scalar_reads[12:24, 5].bool()
    e3_expected = (e3['coords'][0] - e3['coords'][0].mean(0))[:, None].expand(-1, 2, -1)
    poi_expected = (poi['coords'][0] - poi['coords'][0].mean(0))[:, None].expand(-1, 2, -1)
    assert e3_visible.all() and poi_visible.all(), 'v8 keeps a fixed full-receptor expert node set'
    assert torch.allclose(e3_vectors[:, :2], e3_expected)
    assert e3_vectors[:, 2:].count_nonzero() == 0
    assert poi_vectors[:, :2].count_nonzero() == 0
    assert torch.allclose(poi_vectors[:, 2:], poi_expected, atol=1e-5)


    observed = vector_capture[0]['expert_node_features']
    for read, marker, visibility, offset in ((observed[:12], 1, 4, 0),
                                             (observed[12:24], 3, 5, 100)):
        assert read[:, visibility].bool().all()
        assert torch.equal(read[:, marker], torch.arange(12).float() + offset)
    (observed[:, :4].sum() + vector_reads.square().sum()).backward()
    for coords in (e3['coords'], poi['coords'], ligand['coords']):
        assert coords.grad is not None and torch.isfinite(coords.grad).all()
        assert coords.grad.abs().sum() > 0, 'binary reads must retain coordinate Jacobians'


def test_bounded_vector_retains_large_finite_directions_and_coordinate_gradients():
    from tricoopdiff.models.assembly import bounded_vector

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


def test_v8_has_no_legacy_controller_parameters_or_profile_switches():
    from tricoopdiff.models.assembly import production_model_config, VECTOR_EXPERT_MODEL_PROFILE

    config = production_model_config(32, 16, expert_vector_dim=2)
    assert config['model_profile'] == VECTOR_EXPERT_MODEL_PROFILE
    model = TernaryAssembly(**config)
    assert all(name.startswith('controller.') for name in model.state_dict())
    for obsolete in ('pooling', 'evidence_encoder', 'motion_head', 'direct_torsion_head',
                     'gate_head', 'residual_head', 'bond_context_encoder'):
        assert not hasattr(model, obsolete), obsolete
    for profile in ('ternary_endpoint_v1', 'ternary_endpoint_v4', 'ternary_endpoint_v5',
                    'ternary_endpoint_v6', 'ternary_endpoint_v7'):
        try:
            TernaryAssembly(32, 16, model_profile=profile)
        except ValueError as error:
            assert 'retired architecture' in str(error)
        else:
            raise AssertionError('retired motion architecture remained constructible')
    try:
        TernaryAssembly(32, 16, global_node_context=True)
    except TypeError:
        pass
    else:
        raise AssertionError('old architecture switches must not remain active')




def test_v8_source_conditioning_preserves_common_initialization():
    from tricoopdiff.models.assembly import production_model_config

    torch.manual_seed(129)
    plain = TernaryAssembly(**production_model_config(32, 16, expert_vector_dim=2))
    torch.manual_seed(129)
    conditioned = TernaryAssembly(**production_model_config(32, 16, True, expert_vector_dim=2))
    for name, value in plain.state_dict().items():
        assert torch.equal(value, conditioned.state_dict()[name]), name




def test_legal_pose_identity_execution_and_bond_expert_gradients():
    from tricoopdiff.models.assembly import LegalPoseDecoder
    from tricoopdiff.sdpi_flow import SDPIFlow

    _, graph, _, state = fixture()
    decoder = LegalPoseDecoder(32, 4)
    coords, types = graph['node_coords'], graph['node_types']
    h, vectors = torch.randn(15, 32), torch.randn(15, 4, 3)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    masks = graph['lig_mask_rotate']
    opinions = [torch.randn(1, requires_grad=True) for _ in BRANCHES]
    motion = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    gradients = torch.autograd.grad(motion['torsion'].sum(), opinions)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), (
        'both experts must influence the corresponding torsion, not only its count')
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': masks}
    def execute(output):
        velocity = dict(output)
        for branch, column in zip(BRANCHES, (0, 3)):
            velocity[branch] = {'tr': output[branch]['tr'] * state[:, column:column + 1].exp(),
                                'rot': output[branch]['rot'] * torch.pi}
        velocity['torsion'] = output['torsion'] * torch.pi
        return SDPIFlow.differentiable_step(coords[types == 2][None], coords[types == 1][None],
                                            torch.eye(3)[None], velocity, torch.ones(1), data)
    ligand, poi, _ = execute(motion)
    assert torch.allclose(torch.cdist(poi, poi), torch.cdist(coords[types == 1][None],
                                                          coords[types == 1][None]), atol=1e-6)
    edges = graph['lig_edge_index']
    old_lengths = (coords[types == 2][edges[0]] - coords[types == 2][edges[1]]).norm(dim=-1)
    new_lengths = (ligand[0, edges[0]] - ligand[0, edges[1]]).norm(dim=-1)
    assert torch.allclose(old_lengths, new_lengths, atol=1e-6), 'legal torsions must preserve bond lengths'
    for head in (*decoder.rigid.values(), decoder.bond):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    identity = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    ligand, poi, _ = execute(identity)
    assert torch.allclose(ligand[0], coords[types == 2], atol=1e-6)
    assert torch.allclose(poi[0], coords[types == 1], atol=1e-6)


def test_joint_dof_readout_couples_bonds_and_rigid_without_bond_order_dependence():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    _, graph, _, state = fixture()
    decoder = LegalPoseDecoder(32, 4).eval()
    h, vectors = torch.randn(15, 32), torch.randn(15, 4, 3)
    coords, types = graph['node_coords'], graph['node_types']
    bonds = torch.tensor([[0, 1], [1, 2]])
    masks = torch.tensor([[False, True, True, True], [False, False, True, True]])
    opinions = [torch.randn(2, requires_grad=True) for _ in BRANCHES]
    output = decoder(h, vectors, coords, types, bonds, masks, state, opinions)
    for value in (output['torsion'][0, 0], output[BRANCHES[1]]['tr'].sum()):
        gradients = torch.autograd.grad(value, opinions, retain_graph=True)
        assert all(torch.isfinite(g).all() and g[1].abs() > 0 for g in gradients), (
            'other bond evidence must reach this bond and the rigid decision')
    order = torch.tensor([1, 0])
    moved = decoder(h, vectors, coords, types, bonds[:, order], masks[order], state,
                    [opinion[order] for opinion in opinions])
    assert torch.allclose(moved['torsion'], output['torsion'][:, order], atol=1e-6)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.allclose(moved[branch][field], output[branch][field], atol=1e-6), (
                'physical bond ordering must not alter rigid decisions')


def test_intrinsic_memory_is_independent_of_component_placement_and_ligand_torsion():
    from tricoopdiff.models.assembly import ComponentStructureEncoder
    from tricoopdiff.sdpi_flow import SDPIFlow

    _, graph, _, _ = fixture()
    encoder = ComponentStructureEncoder(32).double()
    identity = torch.randn(15, 32, dtype=torch.float64, requires_grad=True)
    coords = graph['node_coords'].double().requires_grad_()
    types = graph['node_types']
    ids = (types == 2).nonzero().flatten()
    edges = ids[graph['lig_edge_index']]
    attrs = graph['lig_edge_attr'].double()
    original = encoder(identity, coords, types, edges, attrs)
    transformed = coords.clone()
    for component in range(3):
        rotation = so3_exp(torch.randn(1, 3, dtype=torch.float64))[0]
        selected = types == component
        transformed[selected] = coords[selected] @ rotation.T + torch.randn(3).double() * 40.
    moved = encoder(identity, transformed, types, edges, attrs)
    assert torch.allclose(original, moved, atol=1e-10), 'intrinsic memory must ignore relative poses'
    velocity = {b: {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)} for b in BRANCHES}
    velocity.update(torsion=coords.new_tensor([[1.3]]), rigid_output_mode='independent_centroids_v1',
                    torsion_update_mode='continuous_horizontal_v1')
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': graph['lig_mask_rotate']}
    ligand, _, _ = SDPIFlow.differentiable_step(coords[types == 2][None], coords[types == 1][None],
                                               torch.eye(3).double()[None], velocity,
                                               torch.ones(1).double(), data)
    folded = coords.clone()
    folded[types == 2] = ligand[0]
    assert not torch.allclose(folded[types == 2], coords[types == 2])
    assert torch.allclose(original, encoder(identity, folded, types, edges, attrs), atol=1e-10), (
        'legal torsions must not rewrite ligand chemical identity memory')
    gradient = torch.autograd.grad(original[5, 0], coords, retain_graph=True)[0]
    assert torch.equal(gradient[types != 1], torch.zeros_like(gradient[types != 1]))
    original[:, 0].sum().backward()
    assert torch.isfinite(identity.grad).all() and identity.grad.norm() > 0


def test_chemical_multihop_and_rotor_geometry_match_physical_action():
    from tricoopdiff.models.assembly import ComponentStructureEncoder, LegalPoseDecoder

    _, graph, _, _ = fixture()
    coords, types = graph['node_coords'].double(), graph['node_types']
    ligand = types == 2
    ids = ligand.nonzero().flatten()
    identity = torch.randn(15, 32, dtype=torch.float64, requires_grad=True)
    encoder = ComponentStructureEncoder(32).double()
    encoded = encoder(identity, coords, types, ids[graph['lig_edge_index']], graph['lig_edge_attr'].double())
    gradient = torch.autograd.grad(encoded[ids[0], 0], identity)[0]
    assert gradient[ids[-1]].abs().sum() > 0, 'chemical memory must propagate beyond one bond'
    vectors = torch.randn(15, 4, 3, dtype=torch.float64)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    masks = graph['lig_mask_rotate']
    geometry = LegalPoseDecoder.action_geometry(coords, vectors, ligand, bonds, masks)
    xyz = coords[ligand]
    step = 1e-6
    from tricoopdiff.sdpi_flow import SDPIFlow
    ligand_data = {'edge_index': bonds, 'edge_mask': torch.ones(bonds.shape[1], dtype=torch.bool),
                   'mask_rotate': masks}
    velocities = {branch: {'tr': torch.zeros(1, 3, dtype=xyz.dtype),
                          'rot': torch.zeros(1, 3, dtype=xyz.dtype)} for branch in BRANCHES}
    velocities.update(torsion=torch.tensor([[-1.]], dtype=xyz.dtype),
                      rigid_output_mode='independent_centroids_v1',
                      torsion_update_mode='continuous_horizontal_v1')
    moved, _, _ = SDPIFlow.differentiable_step(
        xyz[None], coords[types == 1][None], torch.eye(3, dtype=xyz.dtype)[None],
        velocities, torch.tensor([step], dtype=xyz.dtype), ligand_data)
    tangent = (moved[0] - xyz) / (10. * step)
    projection = (vectors[ligand] * tangent[:, None]).sum(-1)
    assert torch.allclose(geometry[0, ligand, -4:], projection, atol=1e-6), (
        'action features must encode the executed horizontal rotor derivative')
    rotation = so3_exp(torch.tensor([[.7, -.4, .9]], dtype=torch.float64))[0]
    transformed = LegalPoseDecoder.action_geometry(coords @ rotation.T + 30.,
                                                   vectors @ rotation.T, ligand, bonds, masks)
    assert torch.allclose(geometry, transformed, atol=1e-10)


def test_global_chemical_read_and_explicit_dof_overlap():
    from tricoopdiff.models.assembly import ComponentStructureEncoder, LegalPoseDecoder

    count = 12
    coords = torch.randn(count + 2, 3, dtype=torch.float64)
    types = torch.tensor([0, 1] + [2] * count)
    left = torch.arange(2, count + 1)
    right = left + 1
    edges = torch.stack((torch.cat((left, right)), torch.cat((right, left))))
    attributes = torch.zeros(len(edges[0]), 4, dtype=torch.float64)
    attributes[:, 0] = 1.
    identity = torch.randn(count + 2, 32, dtype=torch.float64, requires_grad=True)
    encoder = ComponentStructureEncoder(32).double()
    encoded = encoder(identity, coords, types, edges, attributes)
    gradient = torch.autograd.grad(encoded[2, 0], identity)[0]
    assert gradient[-1].abs().sum() > 0, 'chemical structure read must reach beyond four hops'
    chemistry, connected = encoder.chemical_relations(coords[2:], edges - 2, attributes)
    assert connected.all() and torch.allclose(chemistry[0, -1, 0], torch.tensor(12.).double().log())
    positions = torch.zeros(3, 3)
    affected = torch.tensor([[1., 1., 1., 1.], [0., 0., 1., 1.], [1., 1., 0., 0.]])
    relation = LegalPoseDecoder.joint_relations(positions, affected)
    assert relation[1, 2, 1] == 0 and relation[0, 1, 1] == .5
    assert relation[0, 1, 2] == .5 and relation[1, 0, 2] == 1., (
        'subset/superset action overlap must retain its direction')
    xyz = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    rotors = torch.tensor([[0, 1], [1, 0]])
    masks = torch.tensor([[False, True, True, False, False]]).expand(2, -1)
    axes, tangents = LegalPoseDecoder.rotor_tangents(xyz, rotors, masks)
    assert torch.allclose(axes[0], -axes[1])
    assert torch.allclose(tangents[0], -tangents[1]), (
        'action coupling must distinguish reinforcing from cancelling rotor directions')
    assert torch.allclose(tangents.mean(1), torch.zeros(2, 3, dtype=xyz.dtype), atol=1e-12)
    rotation = so3_exp(torch.tensor([[.2, -.4, .7]], dtype=xyz.dtype))[0]
    moved_axes, moved_tangents = LegalPoseDecoder.rotor_tangents(xyz @ rotation.T + 30., rotors, masks)
    assert torch.allclose(moved_axes, axes @ rotation.T, atol=1e-12)
    assert torch.allclose(moved_tangents, tangents @ rotation.T, atol=1e-12)
    gradient = torch.autograd.grad(tangents.square().sum(), xyz)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_physical_velocity_has_no_inverse_remaining_time_gain():
    from tricoopdiff.models.assembly import LegalPoseDecoder

    _, graph, _, state = fixture()
    decoder = LegalPoseDecoder(32, 4).eval()
    h = torch.randn(15, 32, requires_grad=True)
    vectors = torch.randn(15, 4, 3)
    bonds = graph['lig_edge_index'][:, graph['lig_edge_mask']]
    opinions = [torch.randn(1) for _ in BRANCHES]
    args = (h, vectors, graph['node_coords'], graph['node_types'], bonds, graph['lig_mask_rotate'])
    full = decoder(*args, state, opinions)
    short = state.clone()
    short[:, -1] = .01
    short[:, 5:8] = .01
    scaled = decoder(*args, short, opinions)
    for branch in BRANCHES:
        for field in ('tr', 'rot'):
            assert torch.equal(scaled[branch][field], full[branch][field]), (
                'decoder must not multiply the same evidence by inverse remaining time')
    assert torch.equal(scaled['torsion'], full['torsion'])
    objective = lambda output: sum(output[b][f].square().sum()
                                   for b in BRANCHES for f in ('tr', 'rot')) + output['torsion'].square().sum()
    gradients = [torch.autograd.grad(objective(output), h, retain_graph=True)[0]
                 for output in (full, scaled)]
    assert torch.equal(*gradients), 'late-time decoder must not amplify identical input derivatives'
    with torch.no_grad():
        decoder.bond[-1].bias.fill_(2.)
        for head in decoder.rigid.values():
            head[-1].weight.zero_()
            head[-1].bias.fill_(10.)
    large = decoder(*args, state, opinions)
    assert (large['torsion'] * torch.pi).abs().max() > torch.pi
    assert any((large[branch]['rot'] * torch.pi).norm() > torch.pi for branch in BRANCHES), (
        'rigid angular velocity must not impose a pi total-path budget')
    assert torch.allclose(.01 * scaled['torsion'], .2 * (.05 * full['torsion'])), (
        'executed increment must shrink with dt, not stay constant as dt decreases')
    short[:, -1], short[:, 5:8] = 0., 0.
    terminal = decoder(*args, short, opinions)
    assert all(torch.isfinite(terminal[b][f]).all() for b in BRANCHES for f in ('tr', 'rot', 'tor'))


def test_matcha_mutual_condition_preserves_prior_and_live_geometry():
    torch.manual_seed(19)
    model = TernaryAssembly(node_dim=32, input_dim=16, expert_backend='matcha',
                            expert_scalar_dim=320, expert_vector_dim=3).double()
    receptor = {'features': torch.randn(1, 5, 16, dtype=torch.float64),
                'coords': torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)}
    opposite = {'features': torch.randn(1, 7, 16, dtype=torch.float64, requires_grad=True),
                'coords': torch.randn(1, 7, 3, dtype=torch.float64, requires_grad=True)}
    _, graph, _, _ = fixture()
    ligand = {'features': graph['node_features'][graph['node_types'] == 2][None],
              'edge_index': graph['lig_edge_index'], 'edge_attr': graph['lig_edge_attr'].double(),
              'coords': torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)}
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    assert gamma.shape == beta.shape == (1, 5, 320)
    assert gamma.count_nonzero() == beta.count_nonzero() == 0, (
        'zero-initialized FiLM must preserve the frozen pretrained prior')
    with torch.no_grad():
        model.binary_film[0].weight.normal_(std=.03)
        model.condition_geometry.weight.normal_(std=.03)
    gamma, beta = model.binary_condition(receptor, opposite, ligand)
    gradients = torch.autograd.grad((gamma + beta).square().sum(),
                                    (receptor['coords'], opposite['coords'], opposite['features'], ligand['coords']))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), (
        'mutual conditioning must retain both live coordinate paths and opposite-protein identity')
    displaced = model.binary_condition(receptor, opposite,
                                      {**ligand, 'coords': ligand['coords'] + 50.})
    assert any(not torch.allclose(a, b) for a, b in zip((gamma, beta), displaced)), (
        'opposite protein-ligand assembly changes must reach the conditional expert')
    altered = model.binary_condition(receptor, opposite,
                                    {**ligand, 'edge_attr': torch.zeros_like(ligand['edge_attr'])})
    assert any(not torch.allclose(a, b) for a, b in zip((gamma, beta), altered)), (
        'conditional interface evidence must retain ligand bond chemistry')
    rotation = so3_exp(torch.tensor([[.2, -.4, .7]], dtype=torch.float64))[0]
    moved = lambda data: {**data, 'coords': data['coords'] @ rotation.T + 30.}
    transformed = model.binary_condition(moved(receptor), moved(opposite), moved(ligand))
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip((gamma, beta), transformed)), (
        'scalar FiLM conditions must be invariant to joint rigid transforms')
    identical = {**opposite, 'features': torch.zeros_like(opposite['features'])}
    distant = {**identical, 'coords': identical['coords'] + 100.}
    near, far = model.binary_condition(receptor, identical, ligand), model.binary_condition(receptor, distant, ligand)
    assert any(not torch.allclose(a, b) for a, b in zip(near, far)), (
        'identical opposite-protein identities must not erase absolute relative geometry')
    many = {key: value.repeat(1, 5, 1) for key, value in identical.items()}
    centered = model.binary_condition(receptor, many, ligand)
    translated = model.binary_condition({**receptor, 'coords': receptor['coords'] + 1e6},
                                        {**many, 'coords': many['coords'] + 1e6},
                                        {**ligand, 'coords': ligand['coords'] + 1e6})
    assert all(torch.allclose(a, b, atol=1e-9, rtol=0) for a, b in zip(centered, translated)), (
        'large joint translations must not corrupt pair distances through squared-norm cancellation')
    for stage in (1, 2):
        other = model.binary_condition(receptor, opposite, ligand, stage_index=stage)
        assert all(x.count_nonzero() == 0 for x in other), (
            'training one stage condition must not alter another pretrained stage')
    from tricoopdiff.models.expert import matcha_stage_clock
    for time, expected in ((1., 0), (.8, 0), (.5, 1), (.2, 2), (0., 2)):
        stage, local = matcha_stage_clock(torch.tensor([time], dtype=torch.float64))
        assert stage == expected and bool(((local >= 0) & (local <= 1)).all())


def test_matcha_coordinate_cut_keeps_film_trainable_and_rejects_clock_mismatch():
    from types import SimpleNamespace
    from tricoopdiff.models.assembly import TernaryAssemblyModel

    class PairExpert(torch.nn.Module):
        backend, node_scalar_dim, node_vector_dim = 'matcha', 2, 2
        model = SimpleNamespace(no_torsion=False)

        def read_pair(self, receptor, ligand, time, film):
            xyz = torch.cat((ligand['coords'][0], receptor['coords'][0]))
            condition = sum(x.sum() for x in film)
            tr = xyz.mean(0, keepdim=True) + condition
            return (tr, tr, tr.new_zeros(1, 1), xyz[:, :2] + condition,
                    xyz[:, None].expand(-1, 2, -1))

    _, graph, _, state = fixture()
    inputs = [{'coords': graph['node_coords'][graph['node_types'] == i][None].clone().requires_grad_(),
               'features': torch.zeros(1, int((graph['node_types'] == i).sum()), 16)} for i in range(3)]
    inputs[2].update(edge_index=graph['lig_edge_index'], edge_attr=graph['lig_edge_attr'],
                     edge_mask=graph['lig_edge_mask'], mask_rotate=graph['lig_mask_rotate'])
    for coordinate_grad in (False, True):
        fuse = TernaryAssembly(node_dim=32, input_dim=16, expert_scalar_dim=2,
                               expert_vector_dim=2, expert_backend='matcha')
        model = TernaryAssemblyModel(PairExpert(), fuse, expert_input_grad=coordinate_grad,
                                     checkpoint_ternary=True).train()
        output = model(*inputs, torch.ones(1), diffusion_state=state)
        objective = sum(output['binary_proposals'][branch]['tr'].sum() for branch in BRANCHES)
        gradients = torch.autograd.grad(objective,
            (fuse.binary_film[0].weight, *(data['coords'] for data in inputs)), allow_unused=True)
        assert gradients[0] is not None and gradients[0].abs().sum() > 0, (
            'turning off expert coordinate VJPs must not turn off trainable mutual FiLM')
        assert all((g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)
                   if coordinate_grad else g is None for g in gradients[1:])
        for bad_time, bad_state in ((torch.tensor([.5]), state),
                                   (torch.ones(1), state.clone().index_fill(1, torch.tensor([5]), .1))):
            try:
                model(*inputs, bad_time, diffusion_state=bad_state)
            except ValueError:
                pass
            else:
                raise AssertionError('inconsistent expert clock or per-DOF dt must fail before reading experts')
        doubled = [{**data, 'coords': data['coords'].repeat(2, 1, 1),
                    'features': data['features'].repeat(2, 1, 1)} for data in inputs]
        try:
            model(*doubled, torch.ones(1), diffusion_state=state)
        except ValueError as error:
            assert 'one node-aligned example' in str(error)
        else:
            raise AssertionError('B>1 must not silently discard every example after the first')
        malformed = (torch.full((1, 3), float('inf')), torch.zeros(1, 3), torch.zeros(1, 1),
                     torch.zeros(9, 2), torch.zeros(9, 2, 3))
        try:
            model._checked_read(malformed, inputs[2], inputs[0], BRANCHES[0])
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Matcha non-finite proposals must not be hidden by bounded vector features')


def test_matcha_reads_exact_executor_rotor_masks_including_empty_set():
    from rdkit import Chem
    from tricoopdiff.models.expert import FrozenMatcha

    mol = Chem.MolFromSmiles('CCCC')
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    selected = torch.tensor([False, False, True, False, False, False])
    mask = torch.tensor([[False, False, True, True]])
    data = dict(edge_index=edges, edge_mask=selected, mask_rotate=mask)
    bonds, masks = FrozenMatcha.canonical_torsions(mol, data, 'cpu')
    assert torch.equal(bonds, torch.tensor([[1, 2]])) and torch.equal(masks, mask)
    reverse = {**data, 'edge_mask': selected.roll(1), 'mask_rotate': ~mask}
    reversed_bonds, reversed_masks = FrozenMatcha.canonical_torsions(mol, reverse, 'cpu')
    assert torch.equal(reversed_bonds, bonds.flip(1)) and torch.equal(reversed_masks, ~masks), (
        'axis and rotating fragment must be read together, not converted by axis sign alone')
    empty = {**data, 'edge_mask': torch.zeros(6, dtype=torch.bool),
             'mask_rotate': torch.zeros(0, 4, dtype=torch.bool)}
    bonds, masks = FrozenMatcha.canonical_torsions(mol, empty, 'cpu')
    assert bonds.shape == (0, 2) and masks.shape == (0, 4)


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_') and callable(function):
            function()
            print(f'{name} passed')
