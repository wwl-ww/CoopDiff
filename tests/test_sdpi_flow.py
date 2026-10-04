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


def test_flow_preconditioning_uses_native_expert_units_and_table():
    from tricoopdiff.diffdock.utils.diffusion_utils import t_to_sigma
    from tricoopdiff.diffdock.utils import so3
    from tricoopdiff.models.expert import so3_score_norm

    args = SimpleNamespace(tr_sigma_min=.1, tr_sigma_max=19.,
                           rot_sigma_min=.03, rot_sigma_max=1.55,
                           tor_sigma_min=.0314, tor_sigma_max=3.14)
    flow = SDPIFlow()
    for dtype in (torch.float32, torch.float64):
        time = torch.linspace(0., 1., 21, dtype=dtype, requires_grad=True)
        actual = flow.t_to_sigma(time, time, time)
        expected = t_to_sigma(time, time, time, args)
        for value, reference in zip(actual, expected):
            assert torch.allclose(value, reference, rtol=2e-6, atol=1e-8)
        sum(value.sum() for value in actual).backward()
        assert torch.isfinite(time.grad).all() and bool((time.grad > 0).all())
        rotation_sigma = actual[1].detach()
        assert torch.equal(so3_score_norm(rotation_sigma), so3.score_norm(rotation_sigma.cpu()))
        assert so3_score_norm(rotation_sigma[0]).shape == torch.Size([])


def test_shared_rotation_exponential_preserves_precision_and_zero_gradient():
    from scipy.spatial.transform import Rotation
    from tricoopdiff.geometry.rigid import rotation_exp

    vectors = torch.tensor([[0., 0., 0.], [1e-9, -1e-9, 2e-9],
                            [.7, -1.2, .3]], dtype=torch.float64, requires_grad=True)
    matrices = rotation_exp(vectors)
    assert matrices.dtype == vectors.dtype
    assert np.allclose(matrices.detach().numpy(), Rotation.from_rotvec(vectors.detach().numpy()).as_matrix(),
                       atol=1e-12, rtol=0)
    (matrices * torch.arange(9, dtype=vectors.dtype).reshape(3, 3)).sum().backward()
    assert torch.isfinite(vectors.grad).all() and vectors.grad[0].abs().sum() > 0


def test_reverse_clock_integrates_to_zero_with_identical_branch_times():
    flow = SDPIFlow()
    for dtype in (torch.float32, torch.float64):
        reference = torch.empty(1, 3, 3, dtype=dtype)
        for steps in (1, 3, 20):
            for start in (1., .25):
                clock = list(flow.reverse_times(reference, steps, start))
                assert len(clock) == steps
                assert all(t.dtype == dtype and dt.dtype == dtype for t, dt in clock)
                assert all(bool((dt > 0).all()) for _, dt in clock)
                assert torch.allclose(sum(dt for _, dt in clock), reference.new_tensor([start]))
                assert torch.equal(clock[-1][0] - clock[-1][1], reference.new_zeros(1))
                for (t, dt), (following, _) in zip(clock, clock[1:]):
                    assert torch.allclose(t - dt, following)
    for invalid in (0, -1, True, 2.5):
        try:
            list(flow.reverse_times(reference, invalid))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid reverse step count accepted')


def test_shared_model_read_preserves_geometry_gradient_and_conditioning():
    flow, ligand_data, e3, poi, _ = fixture()
    ligand = ligand_data['coords'].detach().clone().requires_grad_()
    moved_poi = (poi + 3).detach().requires_grad_()
    body = torch.eye(3, dtype=e3.dtype).unsqueeze(0)
    t, dt = next(flow.reverse_times(ligand))
    conditions = {'fixture': True}

    class Reader:
        expert = SimpleNamespace(model=SimpleNamespace(dynamic_max_cross=True))

        def __call__(self, ed, pd, ld, **kwargs):
            assert pd['coords'] is moved_poi and ld['coords'] is ligand
            assert kwargs['t_e3l'] is kwargs['t_pol'] is t
            assert kwargs['r_e3_to_poi'] is body
            assert kwargs['conditions'] is conditions and kwargs['source'] == 1
            assert torch.equal(kwargs['diffusion_state'], flow.conditioning(t, dt, e3, poi))
            assert torch.equal(kwargs['receptor_crop_cutoff'], flow.t_to_sigma(t, t, t)[0] * 3 + 20)
            return ld['coords'].sum() + pd['coords'].sum()

    output = flow.forward_state(Reader(), {'coords': e3}, {'coords': poi}, ligand_data,
                                ligand, moved_poi, body, t, dt, source=1, conditions=conditions)
    output.backward()
    assert torch.equal(ligand.grad, torch.ones_like(ligand))
    assert torch.equal(moved_poi.grad, torch.ones_like(moved_poi))
    assert not ligand_data['coords'].requires_grad


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
    from tricoopdiff.experiments.trajectory_experiment import differentiable_sdpi_step

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


def test_rigid_fit_gradient_is_regular_at_repeated_singular_values():
    from tricoopdiff.geometry.rigid import proper_rigid_fit, _ProperPolarRotation
    covariance = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_ProperPolarRotation.apply, (covariance,), atol=2e-5)
    source = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                           [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]], requires_grad=True)
    target = source.detach().clone().requires_grad_(True)
    rotation, translation, _ = proper_rigid_fit(source, target)
    (rotation * torch.randn(3, 3)).sum().add(translation.sum()).backward()
    assert torch.isfinite(source.grad).all() and torch.isfinite(target.grad).all()


def test_exact_rigid_fit_rmsd_has_finite_zero_gradient():
    from tricoopdiff.geometry.rigid import proper_rigid_fit

    source = torch.cat((torch.eye(3), -torch.eye(3))).double().requires_grad_()
    _, _, rmsd = proper_rigid_fit(source, source.detach())
    assert rmsd == 0
    rmsd.backward()
    assert torch.isfinite(source.grad).all() and torch.equal(source.grad, torch.zeros_like(source))


def test_candidates_and_decoder_reject_nonrotation_frames():
    from tricoopdiff.initialization.candidates import FixedCandidateInitializer, _valid_candidate

    flow, data, _, native_poi, coupling = fixture()
    ligand, poi, body, velocities = flow.path(data, native_poi, coupling, torch.tensor([.5]))
    candidate = {'ligand': ligand, 'poi': poi, 'rotation': body}
    assert _valid_candidate(candidate)
    invalid = (torch.eye(3) * 2, torch.diag(torch.tensor([-1., 1., 1.])),
               torch.tensor([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]]))
    for matrix in invalid:
        frame = matrix.double().unsqueeze(0)
        assert not _valid_candidate({**candidate, 'rotation': frame})
        for execute in (
                lambda: FixedCandidateInitializer({**candidate, 'rotation': frame}, no_torsion=False),
                lambda: flow.step(ligand, poi, frame, velocities, torch.tensor([.05]), data)):
            try:
                execute()
            except ValueError:
                pass
            else:
                raise AssertionError('a scale, shear or reflection was accepted as a rotation')


def test_sampler_restores_training_mode_and_rng_on_success_and_failure():
    from tricoopdiff.sampler import sample_single
    from tricoopdiff.initialization import TernaryInitializer

    flow, data, e3, poi, _ = fixture()

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.expert = SimpleNamespace(model=SimpleNamespace(no_torsion=False, dynamic_max_cross=False))
            self.fail = False
            self.fixed_eval_child = torch.nn.Dropout()

        def forward(self, *args, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            if self.fail:
                raise RuntimeError('intentional model failure')
            zero = data['coords'].new_zeros
            return {'normalized_velocity': {
                'e3_ligand': {'tr': zero(1, 3), 'rot': zero(1, 3)},
                'poi_ligand': {'tr': zero(1, 3), 'rot': zero(1, 3)},
                'torsion': zero(1, int(data['edge_mask'].sum()))}}

    model = Predictor()
    for training in (True, False):
        model.train(training)
        model.fixed_eval_child.eval()
        for fail in (False, True):
            model.fail = fail
            rng = torch.get_rng_state().clone()
            try:
                sample_single(model, flow, TernaryInitializer(), {'coords': e3}, {'coords': poi},
                              data, steps=1, seed=3, device='cpu')
            except RuntimeError as error:
                assert fail and 'intentional' in str(error)
            else:
                assert not fail
            assert model.training == training, 'sampling leaked eval mode into its caller'
            assert not model.fixed_eval_child.training, 'sampling changed a fixed-eval child module'
            assert torch.equal(torch.get_rng_state(), rng), 'sampling changed training RNG'


def _folded_torsion_decoder_fixture(dtype=torch.float64):
    ligand = torch.tensor([[[-1., 1., 0.], [0., 0., 0.], [1., 0., 0.], [2., 1., 0.]]], dtype=dtype)
    poi = torch.tensor([[[0., 0., 2.], [1., 1., 2.], [2., 0., 3.]]], dtype=dtype)
    data = {'edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            'edge_mask': torch.tensor([False, False, True, False, False, False]),
            'mask_rotate': torch.tensor([[False, False, True, True]])}
    return ligand, poi, torch.eye(3, dtype=dtype)[None], data


def test_continuous_horizontal_torsion_preserves_tags_and_rejects_unknown_decoder():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    ligand, poi, body, data = _folded_torsion_decoder_fixture()
    normalized = {'rigid_output_mode': INDEPENDENT_RIGID_MODE, 'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                  'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                  'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                  'torsion': ligand.new_zeros(1, 1)}
    velocities = SDPIFlow().velocities(normalized, ligand, poi)
    assert velocities['torsion_update_mode'] == CONTINUOUS_TORSION_MODE
    velocities['torsion_update_mode'] = 'unknown'
    try:
        SDPIFlow.step(ligand, poi, body, velocities, ligand.new_ones(1), data)
    except ValueError as error:
        assert 'torsion update mode' in str(error)
    else:
        raise AssertionError('unknown finite-step torsion semantics were silently accepted')


def test_continuous_horizontal_torsion_crosses_folded_rank_deficient_fit_without_pose_jump():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    for dtype in (torch.float32, torch.float64):
        ligand, poi, body, data = _folded_torsion_decoder_fixture(dtype)
        assert torch.linalg.matrix_rank(ligand[0] - ligand.mean(1)[0]) == 2
        results = []
        for offset in (-1e-3, 0., 1e-3):
            angle = ligand.new_tensor([[np.pi + offset]], requires_grad=True)
            velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                          'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                          'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                          'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                          'torsion': angle}
            result = SDPIFlow.step(ligand, poi, body, velocities, ligand.new_ones(1), data)
            result[0].square().sum().backward()
            assert torch.isfinite(angle.grad).all()
            assert torch.allclose(result[0].mean(1), ligand.mean(1), atol=2e-6, rtol=0.)
            u, v = data['edge_index']
            expected = torch.linalg.vector_norm(ligand[:, u] - ligand[:, v], dim=-1)
            actual = torch.linalg.vector_norm(result[0][:, u] - result[0][:, v], dim=-1)
            assert torch.allclose(actual, expected, atol=2e-6, rtol=0.)
            assert torch.equal(result[1], poi) and torch.equal(result[2], body)
            results.append(result[0])
        assert (results[0] - results[2]).abs().max() < .01, 'continuous torsion decoder jumped at pi'
        velocities.pop('torsion_update_mode')
        velocities['torsion'] = ligand.new_tensor([[np.pi]])
        try:
            SDPIFlow.step(ligand, poi, body, velocities, ligand.new_ones(1), data)
        except ValueError as error:
            assert 'underdetermined' in str(error)
        else:
            raise AssertionError('old untagged decoder no longer preserves its historical chart contract')


def test_continuous_horizontal_torsion_is_equivariant_differentiable_and_safe_for_collinear_ligands():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE
    from tricoopdiff.geometry.rigid import rotation_exp

    ligand, poi, body, data = _folded_torsion_decoder_fixture()
    for collinear in (False, True):
        current = ligand.clone()
        if collinear:
            current[:, :, 1:] = 0.
        current.requires_grad_()
        torsion = current.new_tensor([[2.]], requires_grad=True)
        velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE, 'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                      'e3_ligand': {'tr': current.new_tensor([[.2, -.3, .1]]), 'rot': current.new_tensor([[.1, .2, -.1]])},
                      'poi_ligand': {'tr': current.new_tensor([[-.1, .2, .3]]), 'rot': current.new_tensor([[.3, -.1, .2]])},
                      'torsion': torsion}
        dt = current.new_tensor([.17])
        expected = SDPIFlow.step(current, poi, body, velocities, dt, data)
        rotation = rotation_exp(current.new_tensor([[.3, -.5, .7]]))[0]
        shift = current.new_tensor([12., -3., 5.])
        transformed_velocities = {**velocities}
        for branch in ('e3_ligand', 'poi_ligand'):
            transformed_velocities[branch] = {name: value @ rotation.T for name, value in velocities[branch].items()}
        moved = SDPIFlow.step(current @ rotation.T + shift, poi @ rotation.T + shift,
                             rotation[None] @ body, transformed_velocities, dt, data)
        for first, second in zip(expected[:2], moved[:2]):
            assert torch.allclose(first @ rotation.T + shift, second, atol=1e-9, rtol=0.)
        assert torch.allclose(rotation[None] @ expected[2], moved[2], atol=1e-9, rtol=0.)
        expected[0].square().sum().backward()
        assert torch.isfinite(current.grad).all() and torch.isfinite(torsion.grad).all()


def test_continuous_horizontal_multitorsion_decoder_matches_teacher_tangent():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE

    flow, data, _, native_poi, coupling = fixture('CCCOCC')
    for time in (.1, .47, .9):
        t = torch.tensor([time], dtype=torch.float64)
        ligand, poi, body, velocities = flow.path(data, native_poi, coupling, t)
        velocities['torsion_update_mode'] = CONTINUOUS_TORSION_MODE
        errors = []
        for h in (1e-3, 5e-4):
            dt = t.new_tensor([h])
            actual = flow.step(ligand, poi, body, velocities, dt, data)
            expected = flow.path(data, native_poi, coupling, t - dt)
            errors.append(max(float((actual[index] - expected[index]).abs().max()) for index in (0, 1)))
        assert errors[1] < .4 * errors[0] + 1e-10
        assert errors[1] < 3e-4


def test_continuous_torsion_model_and_sampler_callback_cannot_drop_update_tag():
    from tricoopdiff.initialization.candidates import FixedCandidateInitializer
    from tricoopdiff.sampler import sample_single
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    ligand, poi, body, data = _folded_torsion_decoder_fixture()
    data = {**data, 'coords': ligand}

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.expert = SimpleNamespace(model=SimpleNamespace(no_torsion=False, dynamic_max_cross=False))
            self.trifuse = SimpleNamespace(direct_ternary_motion=True, continuous_torsion=True)
            self.omit = False

        def forward(self, *args, **kwargs):
            velocity = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                        'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                        'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                        'torsion': ligand.new_zeros(1, 1)}
            if not self.omit:
                velocity['torsion_update_mode'] = CONTINUOUS_TORSION_MODE
            return {'normalized_velocity': velocity}

    model = Predictor()
    flow = SDPIFlow(inference_steps=1)
    initializer = FixedCandidateInitializer({'ligand': ligand, 'poi': poi, 'rotation': body,
                                             'candidate_id': ['sdpi', 0]}, no_torsion=False)
    ed, pd = {'coords': ligand}, {'coords': poi}

    def drop_tag(velocities, *args, **kwargs):
        velocities = dict(velocities)
        velocities.pop('torsion_update_mode')
        return velocities

    for omit in (True, False):
        model.omit = omit
        try:
            sample_single(model, flow, initializer, ed, pd, data, device='cpu',
                          init_mode='fixed_candidate', velocity_transform=None if omit else drop_tag)
        except ValueError as error:
            assert 'torsion' in str(error)
        else:
            raise AssertionError('continuous torsion motion silently fell back to the old Kabsch chart')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
