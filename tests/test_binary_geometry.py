"""Binary geometry reaches current-state velocities and linker torsion heads."""

import sys
from pathlib import Path

import torch
from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from test_consensus import fixture
from test_conditional_consensus import velocities
from tricoopdiff.binary_geometry import BinaryGeometryEncoder
from tricoopdiff.condition_features import build_supervised_contact_features
from tricoopdiff.condition_augmentation import ConditionFeatureDropout, dropout_condition_features
from tricoopdiff.conditional_consensus import ConditionedConsensusFuse
from tricoopdiff.diffusion import so3_exp


def inputs():
    base, graph, proposals, state = fixture()
    coords, types = graph['node_coords'], graph['node_types']
    features = build_supervised_contact_features(
        Chem.MolFromSmiles('CCON'), (coords[types == 0], coords[types == 1]), coords[types == 2])
    moved = coords.clone()
    moved[types == 2] += torch.tensor([2., -1., .5])
    return base, {**graph, 'node_coords': moved}, proposals, state, features


def test_dynamic_geometry_equivariance_and_absence():
    _, graph, _, _, features = inputs()
    encoder = BinaryGeometryEncoder(16)
    nodes, basis, _ = encoder(graph, features)
    assert basis.abs().sum() > 0
    rotation = so3_exp(torch.tensor([[.3, -.5, .8]]))[0]
    moved = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 12.}
    other_nodes, other_basis, _ = encoder(moved, features)
    assert torch.allclose(nodes, other_nodes, atol=1e-5)
    assert torch.allclose(basis @ rotation.T, other_basis, atol=1e-5)
    absent, _ = dropout_condition_features(
        features, seed=0, config=ConditionFeatureDropout(1., 0., 0., 0., 0.))
    assert all(value.count_nonzero() == 0 for value in encoder(graph, absent))


def test_geometry_reaches_velocity_and_torsion_gradients():
    base, graph, proposals, state, features = inputs()
    model = ConditionedConsensusFuse.from_base(base)
    with torch.no_grad():
        model.condition_scale.fill_(.2)
    result = model(graph, proposals, state, conditions=features)
    loss = sum((value - .4).square().mean() for value in velocities(result))
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(('condition_geometry', 'condition_vector_head', 'condition_torsion_head')):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    # Torsion training itself must reach the geometry encoder through bond context.
    # The inherited torsion residual head starts at zero for parent preservation.
    torch.optim.SGD(model.parameters(), lr=.01).step()
    model.zero_grad(set_to_none=True)
    model(graph, proposals, state, conditions=features)['normalized_velocity']['torsion'].sum().backward()
    assert sum(p.grad.abs().sum() for p in model.condition_geometry.parameters()
               if p.grad is not None) > 0
    result = model(graph, proposals, state, conditions=features)
    rotation = so3_exp(torch.tensor([[.3, -.5, .8]]))[0]
    moved = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 12.}
    rotated = {b: {c: v if c == 'tor' else v @ rotation.T for c, v in row.items()}
               for b, row in proposals.items()}
    other = model(moved, rotated, state, conditions=features)
    for before, after in zip(velocities(result)[:-1], velocities(other)[:-1]):
        assert torch.allclose(before @ rotation.T, after, atol=2e-5)
    assert torch.allclose(velocities(result)[-1], velocities(other)[-1], atol=2e-5)


def test_receptor_identity_and_zero_torsion_contract():
    base, graph, proposals, state, features = inputs()
    encoder = BinaryGeometryEncoder(16)
    nodes, _, _ = encoder(graph, features)
    changed = {key: value.clone() for key, value in features.items()}
    changed['binary1'] = changed['binary1'].flip(2)
    other, _, _ = encoder(graph, changed)
    assert not torch.allclose(nodes, other), 'receptor-specific correspondence was lost'
    graph = {**graph, 'lig_edge_mask': torch.zeros_like(graph['lig_edge_mask']),
             'lig_mask_rotate': torch.empty(0, 4, dtype=torch.bool)}
    proposals = {b: {**row, 'tor': torch.empty(1, 0)} for b, row in proposals.items()}
    model = ConditionedConsensusFuse.from_base(base)
    result = model(graph, proposals, state, conditions=features)
    assert result['normalized_velocity']['torsion'].shape == (1, 0)


if __name__ == '__main__':
    test_dynamic_geometry_equivariance_and_absence()
    test_geometry_reaches_velocity_and_torsion_gradients()
    test_receptor_identity_and_zero_torsion_contract()
    print('Binary geometry integration checks passed')
