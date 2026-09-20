"""Sparse interface topology, symmetry and structural ranking gradients."""

from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch

from test_structural_ranker import fixture
from tricoopdiff.ranking_interface import interface_edges, InterfaceMessagePassing
from tricoopdiff.structural_ranker import StructuralQualityRanker, structural_ranking_loss


def test_sparse_edge_types_deduplication_and_cutoff():
    g = fixture()
    edges, kinds = interface_edges(g['node_coords'], g['node_types'], g['lig_edge_index'])
    rows = torch.cat((edges.T, kinds[:, None]), 1)
    assert len(rows.unique(dim=0)) == len(rows)
    assert set(kinds.tolist()) == {0, 1, 2, 3}
    assert all((b, a, k) in set(map(tuple, rows.tolist())) for a, b, k in rows.tolist())
    coords = g['node_coords'].clone()
    coords[g['node_types'] == 0] += 100
    coords[g['node_types'] == 1] -= 100
    _, distant_kinds = interface_edges(coords, g['node_types'], g['lig_edge_index'])
    assert set(distant_kinds.tolist()) == {0}, 'far proteins gained artificial contacts'


def test_message_passing_is_rigid_and_permutation_invariant():
    g = fixture()
    module = InterfaceMessagePassing(16, layers=2)
    features = torch.randn(15, 16)
    out = module(features, g['node_coords'], g['node_types'], g['lig_edge_index'])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    moved = module(features, g['node_coords'] @ rotation + 7, g['node_types'], g['lig_edge_index'])
    assert torch.allclose(out, moved, atol=2e-6)
    ligand_perm = torch.tensor([2, 0, 3, 1])
    permutation = torch.cat((torch.tensor([4, 2, 0, 1, 3]), torch.tensor([10, 8, 6, 9, 7, 5]), ligand_perm + 11))
    permuted = module(features[permutation], g['node_coords'][permutation], g['node_types'][permutation],
                      torch.argsort(ligand_perm)[g['lig_edge_index']])
    assert torch.allclose(out, permuted, atol=2e-6)


def test_bond_connectivity_changes_propagated_context():
    torch.manual_seed(17)
    module = InterfaceMessagePassing(16, layers=2)
    features = torch.randn(7, 16)
    types = torch.tensor([0, 1, 2, 2, 2, 2, 2])
    coords = torch.tensor([[100., 0, 0], [-100., 0, 0], [0., 0, 0], [1., 0, 0],
                           [2., 0, 0], [3., 0, 0], [4., 0, 0]])
    path = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    star = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
    assert not torch.allclose(module(features, coords, types, path), module(features, coords, types, star))


def test_optional_branch_keeps_base_initialization_and_gets_gradients():
    g = fixture(); g.pop('stage'); g.pop('source')
    torch.manual_seed(25); base = StructuralQualityRanker(16, 16)
    torch.manual_seed(25); model = StructuralQualityRanker(16, 16, interface_layers=2)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items())
    shifted = {**g, 'node_coords': g['node_coords'].clone()}
    shifted['node_coords'][5:11] += 2
    out = model.forward_list([g, shifted], stages=['initial', 'final'], sources=['sdpi', 'bitseed'])
    structural_ranking_loss(out, torch.tensor([.6, .1])).backward()
    for parameter in (model.interface.messages[0][1].weight, model.interface.updates[0][1].weight,
                      model.interface_gate, model.encoder.protein_feature_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_checkpoint_matches_direct_outputs_and_gradients():
    import copy
    g = fixture(); g.pop('stage'); g.pop('source')
    model = StructuralQualityRanker(16, 16, interface_layers=2)
    direct = copy.deepcopy(model).eval()
    output = model(**g, stage='initial', source='sdpi')['score']
    expected = direct(**g, stage='initial', source='sdpi')['score']
    assert torch.allclose(output, expected, atol=1e-7)
    output.sum().backward(); expected.sum().backward()
    for (name, parameter), (_, reference) in zip(model.named_parameters(), direct.named_parameters()):
        if reference.grad is not None:
            assert parameter.grad is not None and torch.allclose(parameter.grad, reference.grad, atol=1e-6), name


if __name__ == '__main__':
    test_sparse_edge_types_deduplication_and_cutoff()
    test_message_passing_is_rigid_and_permutation_invariant()
    test_bond_connectivity_changes_propagated_context()
    test_optional_branch_keeps_base_initialization_and_gets_gradients()
    test_checkpoint_matches_direct_outputs_and_gradients()
    print('Interface message passing: 5 tests passed')
