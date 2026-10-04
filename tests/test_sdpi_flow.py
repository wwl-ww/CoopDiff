"""Endpoint, coordinate-tangent, integration and train/inference flow gates."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tricoopdiff.diffdock_compat  # noqa: F401
from test_initialization import ligand_input
from tricoopdiff.sdpi_flow import SDPIFlow


def fixture(smiles='CCCOCC', seed=12):
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = ligand_input(smiles)
    data['coords'] = data['coords'].double()
    a = torch.randn(1, 20, 3, dtype=torch.float64) * 10
    b = torch.randn(1, 17, 3, dtype=torch.float64) * 8 + 7
    flow = SDPIFlow()
    coupling = flow.sample_coupling(a, b, data)
    return flow, data, a, b, coupling


def test_flow_clean_and_prior_endpoints():
    flow, data, a, b, coupling = fixture()
    ligand, poi, body, _ = flow.path(data, b, coupling, torch.tensor([0.], dtype=torch.float64))
    assert torch.allclose(ligand, data['coords'], atol=1e-10)
    assert torch.allclose(poi, b, atol=1e-10)
    assert torch.allclose(body[0], torch.eye(3).double(), atol=1e-10)
    ligand, poi, body, _ = flow.path(data, b, coupling, torch.tensor([1.], dtype=torch.float64))
    assert np.allclose(ligand.mean(1)[0].numpy(), coupling['centers'][0])
    # Recover the POI template frame and check its independently drawn center.
    local = (ligand - poi.mean(1, keepdim=True)) @ body + b.mean(1, keepdim=True)
    assert np.allclose(local.mean(1)[0].numpy(), coupling['centers'][1])
    assert torch.allclose(torch.cdist(poi, poi), torch.cdist(b, b), atol=1e-10)


def test_truncated_clock_is_refinement_only():
    flow = SDPIFlow()
    try:
        flow.get_inference_schedule(5, t_start=0.25)
    except ValueError as error:
        assert "prior endpoint" in str(error)
    else:
        raise AssertionError("ordinary SDPI inference accepted a truncated clock")
    schedule = flow.get_refinement_schedule(5, t_start=0.25)
    assert np.allclose(schedule["tr_schedule"], np.linspace(0.25, 0.0, 6))
    assert np.allclose(schedule["rot_schedule"], schedule["tr_schedule"])
    assert np.allclose(schedule["tor_schedule"], schedule["tr_schedule"])


def test_flow_complete_cartesian_tangent_and_step_convergence():
    # Both interfaces, mixed torsions, and nontrivial body frames must agree.
    for smiles in ('CCCOCC', 'CC1CCCCC1', 'c1ccccc1'):
        flow, data, a, b, coupling = fixture(smiles)
        for value in (0.1, 0.47, 0.9):
            time = torch.tensor([value], dtype=torch.float64)
            ligand, poi, body, velocities = flow.path(data, b, coupling, time)
            errors = []
            for h in (1e-3, 5e-4):
                dt = torch.tensor([h], dtype=torch.float64)
                actual = flow.step(ligand, poi, body, velocities, dt, data)
                expected = flow.path(data, b, coupling, time - dt)
                error = max(float((actual[i] - expected[i]).abs().max()) for i in (0, 1))
                errors.append(error)
            assert errors[-1] < 3e-4, (smiles, value, errors)
            assert errors[-1] < errors[0] * 0.4 + 1e-10, (smiles, value, errors)


def test_flow_oracle_roundtrip_refines_with_steps():
    flow, data, a, b, coupling = fixture()
    errors = []
    for steps in (50, 100):
        ligand, poi, body, _ = flow.path(data, b, coupling, torch.tensor([1.], dtype=torch.float64))
        dt = torch.tensor([1 / steps], dtype=torch.float64)
        for i in range(steps):
            t = torch.tensor([1 - i / steps], dtype=torch.float64)
            _, _, _, velocity = flow.path(data, b, coupling, t)
            ligand, poi, body = flow.step(ligand, poi, body, velocity, dt, data)
        errors.append(float((ligand - data['coords']).square().mean().sqrt()))
        assert torch.allclose(torch.cdist(poi, poi), torch.cdist(b, b), atol=1e-8)
    assert errors[1] < errors[0] * 0.65, errors


def test_independent_targets_match_actual_path_and_both_decoders():
    from tricoopdiff.trajectory_experiment import differentiable_sdpi_step

    for smiles in ('CCCOCC', 'c1ccccc1'):
        flow, data, _, b, coupling = fixture(smiles)
        for value in (.1, .47, .9):
            time = torch.tensor([value], dtype=torch.float64)
            ligand, poi, body, teacher = flow.path(data, b, coupling, time)
            independent = flow.independent_targets(teacher, ligand, poi)
            errors = []
            for h in (1e-3, 5e-4):
                dt = time.new_tensor([h])
                actual = flow.step(ligand, poi, body, independent, dt, data)
                expected = flow.path(data, b, coupling, time - dt)
                errors.append(max(float((actual[i] - expected[i]).abs().max()) for i in (0, 1)))
                # Decoder parity includes nonzero torsions and body-frame updates.
                differentiable = differentiable_sdpi_step(ligand, poi, body, independent, dt, data)
                assert all(torch.allclose(a, d, atol=1e-8) for a, d in zip(actual, differentiable))
            assert errors[-1] < 3e-4, (smiles, value, errors)
            assert errors[-1] < .4 * errors[0] + 1e-10, (smiles, value, errors)


def test_flow_normalization_and_conditioning_agree():
    flow, data, a, b, _ = fixture()
    t = torch.tensor([0.6], dtype=torch.float64)
    branches = {name: {'tr': torch.randn(1, 3, dtype=torch.float64),
                       'rot': torch.randn(1, 3, dtype=torch.float64),
                       'tor': torch.randn(1, 2, dtype=torch.float64)}
                for name in ('e3_ligand', 'poi_ligand')}
    output = {'interface_scores': branches, 'torsion_score': torch.randn(1, 2, dtype=torch.float64)}
    normalized = flow.normalized_predictions(output, t)
    velocities = flow.velocities(normalized, a, b)
    scales = flow.scales(a, b)
    assert torch.allclose(velocities['e3_ligand']['tr'] / scales[0], normalized['e3_ligand']['tr'])
    assert torch.allclose(velocities['poi_ligand']['rot'] / np.pi, normalized['poi_ligand']['rot'])
    assert torch.allclose(velocities['torsion'] / np.pi, normalized['torsion'])
    state = flow.conditioning(t, t.new_tensor([.05]), a, b)
    assert state.shape == (1, 9) and torch.isfinite(state).all()
    assert torch.allclose(state, flow.conditioning(t, t.new_tensor([.05]), a + 13, b - 8))


def test_flow_loss_backward_and_shared_clock_guard():
    from tricoopdiff.train import TriCoopDiffTrainer

    flow, data, a, b, _ = fixture()
    n_tor = int(data['edge_mask'].sum())

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.expert = SimpleNamespace(model=SimpleNamespace(no_torsion=False, dynamic_max_cross=False))

        def forward(self, *args, **kwargs):
            self.last_conditions = kwargs.get('conditions')
            scores = {name: {'tr': self.weight.expand(1, 3), 'rot': self.weight.expand(1, 3),
                              'tor': self.weight.expand(1, n_tor)}
                      for name in ('e3_ligand', 'poi_ligand')}
            return {'interface_scores': scores, 'torsion_score': self.weight.expand(1, n_tor)}

    model = Predictor()
    # Keep real bound state/forward hooks without constructing checkpoints or
    # an optimizer. A SimpleNamespace no longer implements compute_loss's API.
    trainer = TriCoopDiffTrainer.__new__(TriCoopDiffTrainer)
    trainer.device = 'cpu'
    trainer.model = model
    trainer.diffusion = flow
    trainer.torsion_expert_aux_weight = .25
    coupling = flow.sample_coupling(a, b, data)
    fixed = dict(t_override=torch.tensor([.5]), coupling_override=coupling)
    plain = TriCoopDiffTrainer.compute_loss(trainer, {'coords': a}, {'coords': b}, data, **fixed)
    torch.manual_seed(999)
    diagnostic = TriCoopDiffTrainer.compute_loss(
        trainer, {'coords': a}, {'coords': b}, data, diagnostics=True, **fixed)
    assert torch.equal(plain['loss_total'], diagnostic['loss_total'])
    assert len(diagnostic['rms']) == 14
    assert all(np.isfinite(value) for value in diagnostic['rms'].values())
    assert all(value.requires_grad for value in diagnostic['components'].values())
    for value in (0., .5, 1.):
        losses = TriCoopDiffTrainer.compute_loss(trainer, {'coords': a}, {'coords': b}, data,
                                                t_override=torch.tensor([value]))
        assert torch.isfinite(losses['loss_total'])
        losses['loss_total'].backward()
        assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
        model.zero_grad()
    try:
        TriCoopDiffTrainer.compute_loss(trainer, {'coords': a}, {'coords': b}, data,
                                       t_override=torch.tensor([.5]), t_pol_override=torch.tensor([.7]))
    except ValueError as error:
        assert 'shared interface clock' in str(error)
    else:
        raise AssertionError('mismatched flow clocks must fail')
    marker = {'ligand': torch.ones(1)}
    TriCoopDiffTrainer.compute_loss(trainer, {'coords': a}, {'coords': b}, data,
                                    conditions=marker, **fixed)
    assert model.last_conditions is marker


def test_flow_is_equivariant_under_global_rigid_transform():
    from scipy.spatial.transform import Rotation
    flow, data, a, b, coupling = fixture()
    q = torch.tensor(Rotation.from_rotvec([.3, -.7, .4]).as_matrix())
    offset = torch.tensor([7., -2., 13.], dtype=torch.float64)
    transformed_data = {**data, 'coords': data['coords'] @ q.T + offset}
    transformed_coupling = {**coupling,
        'centers': coupling['centers'] @ q.numpy().T + offset.numpy(),
        'omegas': coupling['omegas'] @ q.numpy().T}
    t = torch.tensor([.63], dtype=torch.float64)
    original = flow.path(data, b, coupling, t)
    transformed = flow.path(transformed_data, b @ q.T + offset, transformed_coupling, t)
    for i in (0, 1):
        assert torch.allclose(transformed[i], original[i] @ q.T + offset, atol=1e-9)
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert torch.allclose(transformed[3][branch][component], original[3][branch][component] @ q.T, atol=1e-9)


def test_flow_rejects_invalid_batch_time_and_velocity():
    flow, data, a, b, coupling = fixture()
    for invalid in (torch.tensor([float('nan')]), torch.tensor([-.1]), torch.tensor([1.1])):
        try:
            flow.path(data, b, coupling, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid flow time accepted')
    try:
        flow.sample_coupling(a.expand(2, -1, -1), b, data)
    except ValueError:
        pass
    else:
        raise AssertionError('B>1 silently truncated')
    ligand, poi, body, velocity = flow.path(data, b, coupling, torch.tensor([.5]))
    velocity['torsion'][0, 0] = float('nan')
    try:
        flow.step(ligand, poi, body, velocity, torch.tensor([.05]), data)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite torsion velocity accepted')


def test_flow_checkpoint_rejects_old_or_wrong_process():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from evaluation_common import validate_checkpoint_payload
    for payload in ({'checkpoint_version': 16},
                    {'checkpoint_version': 17, 'generative_process': 'native_dsm'},
                    {'checkpoint_version': 17}):
        try:
            validate_checkpoint_payload(payload)
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible checkpoint accepted')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
