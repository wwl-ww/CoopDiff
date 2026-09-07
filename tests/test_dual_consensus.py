"""Geometric, gradient and isolation gates for the production dual flow."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_initialization import ligand_input
from tricoopdiff.dual_consensus import (
    DualConsensusFlow, ConsensusController, base_conformer, consensus_schedule,
    torsion_step, torsion_features,
)


def fixture(smiles='CCCOCC'):
    np.random.seed(12)
    torch.manual_seed(12)
    data = ligand_input(smiles)
    data['coords'] = data['coords'].double()
    a = torch.randn(1, 15, 3, dtype=torch.float64) * 8
    b = torch.randn(1, 12, 3, dtype=torch.float64) * 6 + 5
    flow = DualConsensusFlow()
    coupling = flow.sample_coupling(a, b, data)
    return flow, data, a, b, coupling


def test_clean_endpoint_and_reachability():
    flow, data, a, b, coupling = fixture()
    state, _ = flow.path(data, b, coupling, torch.tensor([0.], dtype=torch.float64))
    assert torch.allclose(state['a'], coupling['clean'], atol=1e-10)
    assert torch.allclose(state['a'], state['b'], atol=1e-10)
    assert torch.allclose(state['poi'], b, atol=1e-10)
    base = torch.from_numpy(base_conformer(data))[None]
    edges = data['edge_index']
    lengths = lambda x: (x[:, edges[0]] - x[:, edges[1]]).norm(dim=-1)
    assert torch.allclose(lengths(base), lengths(state['a']), atol=1e-8), 'projection changed bond lengths'


def test_full_cartesian_tangent():
    for smiles in ('CCCOCC', 'c1ccccc1', 'CC1CCCCC1'):
        flow, data, a, b, coupling = fixture(smiles)
        for value in (0.1, 0.47, 0.9):
            t = torch.tensor([value], dtype=torch.float64)
            state, velocity = flow.path(data, b, coupling, t)
            errors = []
            for h in (1e-3, 5e-4):
                dt = torch.full_like(t, h)
                actual = flow.step(state, velocity, dt, data)
                expected, _ = flow.path(data, b, coupling, t - dt)
                errors.append(max(float((actual[k] - expected[k]).abs().max()) for k in ('a', 'b', 'poi')))
            assert errors[-1] < 5e-4, (smiles, value, errors)
            assert errors[-1] < errors[0] * .4 + 1e-10, (smiles, value, errors)


def test_no_cross_write_and_poi_carries_only_b():
    flow, data, a, b, coupling = fixture()
    state, velocity = flow.path(data, b, coupling, torch.tensor([.5], dtype=torch.float64))
    saved = {k: v.clone() for k, v in state.items()}
    zero = {n: {k: torch.zeros_like(v) for k, v in comp.items()} for n, comp in velocity.items()}
    for branch, unchanged in (('e3_ligand', ('b', 'poi')), ('poi_ligand', ('a', 'poi')), ('poi', ('a',))):
        v = {**zero, branch: velocity[branch]}
        updated = flow.step(state, v, torch.tensor([.05], dtype=torch.float64), data)
        for key in unchanged:
            assert torch.allclose(updated[key], state[key], atol=1e-10), (branch, key)
        assert all(torch.equal(saved[k], state[k]) for k in saved), 'step mutated input state'


def test_initialization_independent_and_native_free():
    flow, data, a, b, coupling = fixture()
    first, _ = flow.initialize(a, b, data, seed=7)
    changed, _ = flow.initialize(a, b, {**data, 'coords': data['coords'] * 1000}, seed=7)
    assert all(torch.equal(first[k], changed[k]) for k in first)
    assert first['a'].data_ptr() != first['b'].data_ptr()
    assert not torch.allclose(first['a'], first['b'])
    assert not torch.allclose(torsion_features(first['a'], data), torsion_features(first['b'], data))
    assert torch.allclose(torch.cdist(first['poi'], first['poi']), torch.cdist(b, b), atol=1e-8)


def test_schedules():
    early, middle, late = [consensus_schedule(torch.tensor([t])) for t in (1., .5, 0.)]
    assert all(v == 0 for v in early.values())
    assert middle['pose'] == 0 and middle['endpoint'] > 0
    assert all(late[k] >= middle[k] >= early[k] for k in early)


def test_reconstruction_gradients_and_geometry():
    flow, data, a, b, coupling = fixture()
    x = coupling['clean'].clone()
    angles = torch.full((1, len(coupling['bonds'])), .13, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: torsion_step(x, z, data), (angles,), atol=1e-4)
    y = torsion_step(x, angles, data)
    edges = data['edge_index']
    assert torch.allclose((x[:, edges[0]] - x[:, edges[1]]).norm(dim=-1),
                          (y[:, edges[0]] - y[:, edges[1]]).norm(dim=-1), atol=1e-8)


def test_controller_equivariance_and_gradient():
    flow, data, a, b, coupling = fixture()
    state, _ = flow.path(data, b, coupling, torch.tensor([.5], dtype=torch.float64))
    head = ConsensusController(8).double()
    contexts = (torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64))
    n = len(coupling['bonds'])
    bonds = tuple(torch.randn(1, n, 8, dtype=torch.float64) for _ in range(2))
    scores = tuple(torch.randn(1, n, dtype=torch.float64) for _ in range(2))
    t = torch.tensor([.4], dtype=torch.float64)
    out = head(contexts, bonds, state['a'], state['b'], a, state['poi'], scores, t)
    assert all(torch.count_nonzero(v) == 0 for c in out.values() for v in c.values())
    for param in head.parameters():
        torch.nn.init.normal_(param, std=.1)
    out = head(contexts, bonds, state['a'], state['b'], a, state['poi'], scores, t)
    from tricoopdiff.sdpi_flow import _rotation_exp
    r = _rotation_exp(torch.tensor([.3, -.4, .1], dtype=torch.float64))
    transform = lambda x: x @ r.T + 17
    rotated = head(contexts, bonds, transform(state['a']), transform(state['b']), transform(a), transform(state['poi']), scores, t)
    for name in out:
        for key, val in out[name].items():
            expected = val if key == 'tor' else val @ r.T
            assert torch.allclose(rotated[name][key], expected, atol=1e-8), (name, key)
    sum(v.square().sum() for c in out.values() for v in c.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'[OK] {name}', flush=True)
