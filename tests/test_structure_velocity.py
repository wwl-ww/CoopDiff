"""Sampler finite differences and complete-map selection contracts."""

import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from test_consensus import fixture
from tricoopdiff.binary_geometry import select_binary_mapping
from tricoopdiff.structure_velocity import cartesian_velocities, structural_velocity_loss
from tricoopdiff.sdpi_flow import SDPIFlow


def test_complete_mapping_rejects_pairwise_hybrid():
    alternatives = torch.zeros(2, 2, 2, 3)
    alternatives[..., 2] = 1
    first = torch.tensor([[1., 5.], [5., 1.]])
    alternatives[0, ..., 0] = first
    alternatives[1, ..., 0] = first.flip(0)
    alternatives[..., 1] = alternatives[..., 0]
    # Each pair fits some alternative, but the whole pose fits neither.
    hybrid = torch.ones(2, 2)
    chosen, _ = select_binary_mapping(hybrid, alternatives)
    assert (hybrid - chosen[..., 0]).square().sum() > 0
    chosen, index = select_binary_mapping(first.flip(0), alternatives)
    assert int(index) == 1 and torch.equal(chosen[..., 0], first.flip(0))
    reordered, _ = select_binary_mapping(first.flip(0), alternatives.flip(0))
    assert torch.equal(chosen, reordered)


def test_cartesian_tangent_matches_actual_reverse_sampler():
    _, graph, proposals, _ = fixture()
    coords = graph['node_coords'].double()
    ligand, poi = coords[-4:][None], coords[5:11][None]
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': graph['lig_mask_rotate']}
    velocities = {b: {k: v.double() for k, v in row.items()} for b, row in proposals.items()}
    velocities['torsion'] = torch.tensor([[.7]], dtype=torch.float64)
    expected_l, expected_p = cartesian_velocities(ligand[0], poi[0], velocities, data)
    dt = torch.tensor([1e-5], dtype=torch.float64)
    moved_l, moved_p, _ = SDPIFlow.step(ligand, poi, torch.eye(3).double()[None],
                                       velocities, dt, data)
    assert torch.allclose((ligand - moved_l)[0] / dt, expected_l, atol=1e-4)
    assert torch.allclose((poi - moved_p)[0] / dt, expected_p, atol=1e-4)


def test_structure_loss_has_torsion_and_both_rigid_gradients():
    _, graph, proposals, _ = fixture()
    coords = graph['node_coords']
    data = {'edge_index': graph['lig_edge_index'], 'edge_mask': graph['lig_edge_mask'],
            'mask_rotate': graph['lig_mask_rotate']}
    predicted = {b: {k: v.clone().requires_grad_() for k, v in row.items()}
                 for b, row in proposals.items()}
    predicted['torsion'] = torch.tensor([[.7]], requires_grad=True)
    target = {b: {k: torch.zeros_like(v) for k, v in row.items()}
              for b, row in proposals.items()}
    target['torsion'] = torch.zeros(1, 1)
    loss = structural_velocity_loss(coords[:5], coords[-4:], coords[5:11],
                                     predicted, target, data, None, 10.)
    loss.backward()
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert predicted[branch][component].grad.abs().sum() > 0
    assert predicted['torsion'].grad.abs().sum() > 0
    zero = structural_velocity_loss(coords[:5], coords[-4:], coords[5:11],
                                     predicted, predicted, data, None, 10.)
    assert zero == 0


if __name__ == '__main__':
    test_complete_mapping_rejects_pairwise_hybrid()
    test_cartesian_tangent_matches_actual_reverse_sampler()
    test_structure_loss_has_torsion_and_both_rigid_gradients()
    print('Structure velocity and mapping checks passed')
