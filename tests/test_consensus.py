"""Nonzero-residual equivariance, proposal isolation, gradients and EMA checks."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.consensus import ConsensusFuse, BRANCHES
from tricoopdiff.consensus_train import validate_consensus_checkpoint
from tricoopdiff.diffusion import so3_exp


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


def test_legacy_checkpoint_rejected():
    try:
        validate_consensus_checkpoint({'checkpoint_version': 17})
    except ValueError as error:
        assert '18' in str(error)
    else:
        raise AssertionError('legacy weights must not masquerade as model-v2')


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
