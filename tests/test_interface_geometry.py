"""Current-interface directions preserve geometry and transmit gradients."""

import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from test_binary_geometry import inputs
from test_conditional_consensus import velocities
from tricoopdiff.conditional_consensus import ConditionedConsensusFuse
from tricoopdiff.diffusion import so3_exp
from tricoopdiff.interface_geometry import InterfaceGeometryEncoder


def test_interface_equivariance_and_chemical_sensitivity():
    _, graph, _, _, _ = inputs()
    encoder = InterfaceGeometryEncoder(16)
    chemical = graph['node_features']
    nodes, basis, forces = encoder(graph, chemical)
    rotation = so3_exp(torch.tensor([[.4, -.3, .1]]))[0]
    moved = {**graph, 'node_coords': graph['node_coords'] @ rotation.T + 7.}
    new_nodes, new_basis, new_forces = encoder(moved, chemical)
    assert torch.allclose(nodes, new_nodes, atol=1e-5)
    assert torch.allclose(basis @ rotation.T, new_basis, atol=1e-5)
    assert torch.allclose(forces @ rotation.T, new_forces, atol=1e-5)
    changed = chemical.clone()
    changed[0] += 2.
    assert not torch.allclose(nodes, encoder(graph, changed)[0])


def test_interface_geometry_receives_training_gradients():
    base, graph, proposals, state, conditions = inputs()
    model = ConditionedConsensusFuse.from_base(base)
    with torch.no_grad():
        model.condition_scale.fill_(.2)
    result = model(graph, proposals, state, conditions=conditions)
    sum((v - .4).square().mean() for v in velocities(result)).backward()
    for name, parameter in model.condition_interface.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


if __name__ == '__main__':
    test_interface_equivariance_and_chemical_sensitivity()
    test_interface_geometry_receives_training_gradients()
    print('Interface geometry checks passed')
