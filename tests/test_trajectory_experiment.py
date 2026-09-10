"""CPU contracts for differentiable short-horizon trajectory experiments."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.trajectory_experiment import (
    TRAJECTORY_CONTRACT_MODE,
    TrajectoryConsistencyTrainer,
    differentiable_sdpi_step,
)


def example(dtype=torch.float64):
    ligand = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
          [2., 1., .5], [2.5, 1.5, 1.2]]], dtype=dtype)
    poi = torch.tensor(
        [[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]],
        dtype=dtype)
    body = torch.eye(3, dtype=dtype).reshape(1, 3, 3)
    ligand_data = {
        'edge_index': torch.tensor([[0, 1], [1, 0]]),
        'edge_mask': torch.tensor([True, False]),
        'mask_rotate': np.array([[False, True, True, True, True]]),
    }
    velocities = {
        'e3_ligand': {
            'tr': torch.tensor([[.4, -.2, .3]], dtype=dtype),
            'rot': torch.tensor([[.2, .1, -.3]], dtype=dtype),
        },
        'poi_ligand': {
            'tr': torch.tensor([[-.1, .5, .2]], dtype=dtype),
            'rot': torch.tensor([[.3, -.2, .1]], dtype=dtype),
        },
        'torsion': torch.tensor([[.7]], dtype=dtype),
    }
    return ligand, poi, body, ligand_data, velocities


def test_differentiable_step_matches_production_solver():
    ligand, poi, body, ligand_data, velocities = example()
    dt = torch.tensor([.05], dtype=ligand.dtype)
    expected = SDPIFlow.step(
        ligand, poi, body, velocities, dt, ligand_data)
    actual = differentiable_sdpi_step(
        ligand, poi, body, velocities, dt, ligand_data)
    for value, reference in zip(actual, expected):
        assert torch.allclose(value, reference, atol=2e-7, rtol=2e-7), (
            value - reference).abs().max()


def test_gradients_cross_two_differentiable_steps():
    ligand, poi, body, ligand_data, velocities = example()
    leaves = []
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            value = velocities[branch][component].clone().requires_grad_()
            velocities[branch][component] = value
            leaves.append(value)
    velocities['torsion'] = velocities['torsion'].clone().requires_grad_()
    leaves.append(velocities['torsion'])
    dt = torch.tensor([.05], dtype=ligand.dtype)
    for _ in range(2):
        ligand, poi, body = differentiable_sdpi_step(
            ligand, poi, body, velocities, dt, ligand_data)
    loss = ligand.square().mean() + poi.square().mean() + body.square().mean()
    loss.backward()
    for value in leaves:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert float(value.grad.abs().sum()) > 0


def test_differentiable_step_supports_no_torsion_graphs():
    ligand, poi, body, ligand_data, velocities = example()
    ligand_data = {
        **ligand_data,
        'edge_mask': torch.tensor([False, False]),
        'mask_rotate': np.zeros((0, ligand.shape[1]), dtype=bool),
    }
    velocities = {**velocities, 'torsion': ligand.new_empty((1, 0))}
    dt = torch.tensor([.05], dtype=ligand.dtype)
    expected = SDPIFlow.step(ligand, poi, body, velocities, dt, ligand_data)
    actual = differentiable_sdpi_step(
        ligand, poi, body, velocities, dt, ligand_data)
    for value, reference in zip(actual, expected):
        assert torch.allclose(value, reference, atol=1e-12, rtol=1e-12)


def test_trajectory_contract_has_checkpoint_identity():
    trainer = TrajectoryConsistencyTrainer.__new__(TrajectoryConsistencyTrainer)
    trainer.trajectory_contract_mode = TRAJECTORY_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = .1
    trainer.trajectory_min_time = .25
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == TRAJECTORY_CONTRACT_MODE
    assert contract['trajectory_steps'] == 3
    assert contract['trajectory_weight'] == .1
    assert contract['trajectory_min_time'] == .25


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
