"""Nonzero-residual equivariance, proposal isolation, gradients and EMA checks."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.consensus import ConsensusFuse, BRANCHES
from tricoopdiff.consensus_train import validate_consensus_checkpoint
from tricoopdiff.diffusion import so3_exp
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


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(name, 'passed')
