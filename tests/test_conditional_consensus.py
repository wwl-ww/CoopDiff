"""Conditioned consensus migration, gradients, equivariance and frozen expert boundaries."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_consensus import fixture
from tricoopdiff.condition_features import SideCondition, build_condition_features
from tricoopdiff.conditional_consensus import (ConditionedConsensusFuse, ConditionedConsensusCoopDiff,
                                              SideAwareConditionFusion, load_prototype_checkpoint)
from tricoopdiff.consensus import BRANCHES, ConsensusCoopDiff, ConsensusFuse
from tricoopdiff.consensus_train import validate_consensus_checkpoint
from tricoopdiff.diffusion import so3_exp


def conditions():
    molecule = Chem.MolFromSmiles('CCON')
    side = SideCondition(provenance='template', evidence_id='fixture:known',
                         fragment=Chem.MolFromSmiles('CO'), atom_map=(1, 2),
                         pocket_probabilities=[.8, .3, .1, 0., 0.])
    return build_condition_features(molecule, (5, 6), (side, None))


def velocities(result):
    return [result['normalized_velocity'][b][c] for b in BRANCHES for c in ('tr', 'rot')] + [
        result['normalized_velocity']['torsion']]


def test_side_aware_fusion_preserves_identity_missingness_and_linker_gradients():
    hidden = 8
    module = SideAwareConditionFusion(hidden)
    encoded = {
        'protein1': torch.randn(5, hidden), 'protein2': torch.randn(6, hidden),
        'ligand': torch.randn(4, hidden), 'sides': torch.randn(2, hidden),
    }
    features = {name: value.clone() for name, value in conditions().items()}
    features['sides'][1] = features['sides'][0]
    features['ligand'][3, 1] = 1.
    features['ligand'][0, 2] = 1.
    side, linker, joint = module(encoded, features)
    assert side.shape == (2, hidden) and linker.shape == joint.shape == (hidden,)
    assert all(torch.count_nonzero(value) for value in (side[0], side[1], linker, joint))
    joint.square().sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    absent = build_condition_features(Chem.MolFromSmiles('CCON'), (5, 6))
    absent_side, absent_linker, absent_joint = module(encoded, absent)
    assert torch.count_nonzero(absent_side) == 0
    assert torch.count_nonzero(absent_linker) == 0
    assert torch.count_nonzero(absent_joint) == 0


def test_migration_preserves_base_weights_rng_and_absent_condition_outputs():
    base, graph, proposals, state = fixture()
    rng = torch.random.get_rng_state()
    model = ConditionedConsensusFuse.from_base(base)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in base.state_dict().items())
    expected = base(graph, proposals, state)
    absent = build_condition_features(Chem.MolFromSmiles('CCON'), (5, 6))
    for result in (model(graph, proposals, state),
                   model(graph, proposals, state, conditions=absent),
                   model(graph, proposals, state, conditions=conditions())):
        assert all(torch.equal(a, b) for a, b in zip(velocities(expected), velocities(result)))
        assert torch.equal(expected['rank_features'], result['rank_features'])


def test_cpu_condition_initialization_does_not_reseed_cuda():
    base, _, _, _ = fixture()
    with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA reseeded')):
        ConditionedConsensusFuse.from_base(base)


def test_condition_opens_from_exact_zero_then_reaches_encoders():
    base, graph, proposals, state = fixture()
    model = ConditionedConsensusFuse.from_base(base)
    original = copy.deepcopy((graph, proposals))
    before = model(graph, proposals, state)
    conditioned = model(graph, proposals, state, conditions=conditions())
    assert all(torch.equal(a, b) for a, b in zip(velocities(before), velocities(conditioned)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sum((v - .5).square().mean() for v in velocities(conditioned)).backward()
    assert model.condition_scale.grad.abs() > 0
    assert model.condition_gate_head.weight.grad.abs().sum() == 0
    assert sum(p.grad.abs().sum() for p in model.condition_encoder.parameters()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sum((v - .5).square().mean() for v in velocities(model(graph, proposals, state, conditions=conditions()))).backward()
    assert model.condition_gate_head.weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.condition_encoder.parameters()) > 0
    assert model.condition_node_projection.weight.grad.abs().sum() > 0
    assert all(torch.equal(graph[k], original[0][k]) for k in graph)
    assert all(torch.equal(proposals[b][c], original[1][b][c]) for b in BRANCHES for c in ('tr', 'rot', 'tor'))
    prepared, _, _ = model.prepare_conditions(graph, conditions())
    assert prepared['node_features'] is graph['node_features'], 'raw atom category IDs must never be perturbed'


def test_conditional_velocity_is_equivariant_with_nonzero_residuals():
    base, graph, proposals, state = fixture()
    model = ConditionedConsensusFuse.from_base(base)
    torch.nn.init.normal_(model.residual_head.weight, std=.1)
    torch.nn.init.normal_(model.gate_head.weight, std=.1)
    original = model(graph, proposals, state, conditions=conditions())
    rotation = so3_exp(torch.tensor([[.4, -.2, .8]]))[0]
    moved_graph = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 10}
    moved_proposals = {b: {c: v @ rotation.T if c != 'tor' else v for c, v in row.items()}
                       for b, row in proposals.items()}
    changed = model(moved_graph, moved_proposals, state, conditions=conditions())
    assert torch.allclose(original['rank_features'], changed['rank_features'], atol=2e-5)
    for b in BRANCHES:
        for c in ('tr', 'rot'):
            assert torch.allclose(changed['normalized_velocity'][b][c],
                                  original['normalized_velocity'][b][c] @ rotation.T, atol=2e-5)
    assert torch.allclose(changed['normalized_velocity']['torsion'], original['normalized_velocity']['torsion'], atol=2e-5)


def test_prototype_roundtrip_and_legacy_rejection():
    base, graph, proposals, state = fixture()
    model = ConditionedConsensusFuse.from_base(base)
    checkpoint = model.prototype_checkpoint('a' * 64)
    restored = load_prototype_checkpoint(checkpoint).eval()
    assert all(torch.equal(a, b) for a, b in zip(
        velocities(model(graph, proposals, state, conditions=conditions())),
        velocities(restored(graph, proposals, state, conditions=conditions()))))
    try:
        validate_consensus_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError('prototype must not masquerade as a production checkpoint')
    try:
        base(graph, proposals, state, conditions=conditions())
    except ValueError:
        pass
    else:
        raise AssertionError('legacy fuse must reject unknown conditions')


def test_condition_node_alignment_rejected_and_source_stays_independent():
    _, graph, proposals, state = fixture()
    base = ConsensusFuse(node_dim=32, input_dim=16, num_transformer_layers=1, dropout=0, source_conditioning=True).eval()
    model = ConditionedConsensusFuse.from_base(base)
    features = conditions()
    first = model(graph, proposals, state, source=0, conditions=features)
    second = model(graph, proposals, state, source=1, conditions=features)
    assert not torch.equal(first['rank_features'], second['rank_features'])
    try:
        model(graph, proposals, state, conditions={**features, 'protein1': torch.zeros(4, 3)})
    except ValueError:
        pass
    else:
        raise AssertionError('unaligned receptor conditions accepted')


def test_wrapper_never_exposes_condition_channels_or_gradients_to_experts():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.num_conv_layers = 1
            self.model = SimpleNamespace(dynamic_max_cross=False)
            self.seen = []

        def forward(self, graph):
            self.seen.append((graph['ligand'].x.clone(), graph['receptor'].x.clone()))
            vector = self.weight * torch.ones(1, 3)
            return vector, vector * .5, self.weight.reshape(1)

    base, graph, _, state = fixture()
    fuse = ConditionedConsensusFuse.from_base(base)
    expert = Expert()
    model = ConditionedConsensusCoopDiff(expert, fuse).eval()
    types = graph['node_types']
    proteins = []
    for index in (0, 1):
        count = int((types == index).sum())
        edges = torch.stack((torch.arange(count), torch.arange(count).roll(1)))
        proteins.append({'coords': graph['node_coords'][types == index][None],
                         'features': graph['node_features'][types == index][None], 'edge_index': edges})
    ligand = {'coords': graph['node_coords'][types == 2][None], 'features': graph['node_features'][types == 2][None],
              'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
              'mask_rotate': graph['lig_mask_rotate'], 'edge_attr': torch.ones(6, 1)}
    before = model(*proteins, ligand, torch.tensor([.5]), diffusion_state=state)
    after = model(*proteins, ligand, torch.tensor([.5]), diffusion_state=state, conditions=conditions())
    assert all(torch.equal(a, b) for pair_a, pair_b in zip(expert.seen[:2], expert.seen[2:]) for a, b in zip(pair_a, pair_b))
    assert all(torch.equal(before['binary_proposals'][b][c], after['binary_proposals'][b][c])
               for b in BRANCHES for c in ('tr', 'rot', 'tor'))
    sum(v.square().sum() for v in velocities(after)).backward()
    assert expert.weight.grad is None and not expert.weight.requires_grad
    try:
        ConsensusCoopDiff(Expert(), fuse)
    except ValueError:
        pass
    else:
        raise AssertionError('conditional weights cannot use the old architecture wrapper')
