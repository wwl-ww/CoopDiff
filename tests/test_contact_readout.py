"""Contact interaction preserves invariance, replay and old initialization."""
import copy
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from test_structural_ranker import fixture
from tricoopdiff.ranking_interface import InterfaceMessagePassing
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss
from tricoopdiff.ranking_backward import backward_list


def test_contact_symmetry_and_shared_initialization():
    g = fixture(); features = torch.randn(len(g['node_types']), 16)
    for mode in ('contacts', 'contacts-independent'):
        module = InterfaceMessagePassing(16, readout=mode).eval()
        out = module(features, g['node_coords'], g['node_types'], g['lig_edge_index'])
        rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = module(features, g['node_coords'] @ rot + 7, g['node_types'], g['lig_edge_index'])
        assert torch.allclose(out, moved, atol=3e-6)
        perm = torch.arange(len(features))
        perm[:5] = torch.tensor([4, 2, 0, 1, 3])
        shuffled = module(features[perm], g['node_coords'][perm], g['node_types'][perm], g['lig_edge_index'])
        assert torch.allclose(out, shuffled, atol=3e-6)
    torch.manual_seed(42); a = StructuralQualityRanker(16, 16, interface_layers=2, interface_readout='contacts')
    torch.manual_seed(42); b = StructuralQualityRanker(16, 16, interface_layers=2, interface_readout='contacts-independent')
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_contact_gradient_replay():
    g = fixture(); g.pop('stage'); g.pop('source')
    for mode in ('contacts', 'contacts-independent'):
        model = StructuralQualityRanker(16, 16, interface_layers=2, interface_readout=mode)
        replay = copy.deepcopy(model)
        case = {'graphs': [g, g], 'stages': ['initial', 'final'], 'sources': ['sdpi', 'bitseed']}
        loss = lambda out: structural_ranking_loss(out, torch.tensor([.1, .6]))
        loss(model.forward_list(case['graphs'], stages=case['stages'], sources=case['sources'])).backward()
        backward_list(replay, case, 'cpu', loss, chunk_size=1)
        for (name, p), (_, q) in zip(model.named_parameters(), replay.named_parameters()):
            if p.grad is not None:
                assert torch.allclose(p.grad, q.grad, atol=3e-6, rtol=3e-4), name
        assert model.interface.contact_project[1].weight.grad.abs().sum() > 0
        if mode == 'contacts':
            assert model.interface.contact_interaction.linear1.weight.grad.abs().sum() > 0
        else:
            assert model.interface.contact_interaction.linear1.weight.grad is None
