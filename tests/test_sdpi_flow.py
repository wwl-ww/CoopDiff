"""Sdpi flow regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import math
from types import SimpleNamespace

import numpy as np
import torch

from helpers import flow_example
from tricoopdiff.experiments.joint_endpoint_experiment import (
    joint_rigid_endpoint_targets,
    stable_so3_log,
)
from tricoopdiff.experiments.offpath_experiments import (
    BodyFrameTransportRolloutTrainer,
    BoundedJointEndpointRolloutTrainer,
    GAUGE_CORRECT_CONTRACT_MODE,
    GAUGE_CORRECT_TORSION_CONTRACT,
    GaugeCorrectBodyFrameTransportRolloutTrainer,
    JOINT_ENDPOINT_CONTRACT_MODE,
    JOINT_ENDPOINT_TORSION_CONTRACT,
    OFFPATH_CONTRACT_MODE,
    TORSION_GAUGE_CONTRACT,
    _state_torsion_gauge,
    transport_poi_body_frame_targets,
    transport_targets_with_recomputed_gauge,
)
from tricoopdiff.experiments.trajectory_experiment import (
    DISTANCE_STAGED_POI_CONTRACT_MODE,
    DeterministicPriorReplayTrajectoryTrainer,
    PRIOR_REPLAY_CONTRACT_MODE,
    PRIOR_REPLAY_EVAL_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE,
    PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE,
    PriorReplayJointTranslationAgreementTrainer,
    PriorReplayJointTranslationDirectionTrainer,
    PriorReplayJointTranslationReplaceTrainer,
    PriorReplayJointTranslationSoftTimeTrainer,
    PriorReplayJointTranslationTrainer,
    PriorReplayJointTranslationWindowReplaceTrainer,
    PriorReplayTrajectoryTrainer,
    STAGED_POI_CONTRACT_MODE,
    StagedPOITrajectoryTrainer,
    TRAJECTORY_CONTRACT_MODE,
    TrajectoryConsistencyTrainer,
    differentiable_sdpi_step,
    joint_endpoint_translation_loss,
    prior_replay_schedule,
    remaining_centroid_loss,
    staged_poi_state_loss,
)
from tricoopdiff.geometry.rigid import (
    apply_rigid,
    assemble_interface_pair,
    rotation_exp as _rotation_exp,
)
from tricoopdiff.sdpi_flow import SDPIFlow


# Sdpi flow

def test_flow_clean_and_prior_endpoints():
    flow, data, a, b, coupling = flow_example()
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
    flow, ligand_data, e3, poi, _ = flow_example()
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
        flow, data, a, b, coupling = flow_example(smiles)
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
    flow, data, a, b, coupling = flow_example()
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
        flow, data, _, b, coupling = flow_example(smiles)
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
    flow, data, a, b, _ = flow_example()
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

    flow, data, a, b, _ = flow_example()
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
    flow, data, a, b, coupling = flow_example()
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
    flow, data, a, b, coupling = flow_example()
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

    flow, data, _, native_poi, coupling = flow_example()
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

    flow, data, e3, poi, _ = flow_example()

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


def sdpi_flow_folded_torsion_decoder_fixture(dtype=torch.float64):
    ligand = torch.tensor([[[-1., 1., 0.], [0., 0., 0.], [1., 0., 0.], [2., 1., 0.]]], dtype=dtype)
    poi = torch.tensor([[[0., 0., 2.], [1., 1., 2.], [2., 0., 3.]]], dtype=dtype)
    data = {'edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            'edge_mask': torch.tensor([False, False, True, False, False, False]),
            'mask_rotate': torch.tensor([[False, False, True, True]])}
    return ligand, poi, torch.eye(3, dtype=dtype)[None], data


def test_continuous_horizontal_torsion_preserves_tags_and_rejects_unknown_decoder():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture()
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
        ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture(dtype)
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

    ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture()
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

    flow, data, _, native_poi, coupling = flow_example('CCCOCC')
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

    ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture()
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


def test_rotation_exponential_keeps_large_finite_angles_on_so3_without_dtype_or_gradient_change():
    from tricoopdiff.geometry.rigid import is_proper_rotation, rotation_exp

    for scale in (1e3, 1e4, 1e5, 1e8, 1e12, 1e20):
        vector = torch.tensor([[scale, .3 * scale, -.7 * scale]], requires_grad=True)
        rotation = rotation_exp(vector)
        assert rotation.dtype == vector.dtype and is_proper_rotation(rotation, atol=1e-6), (
            'large finite rotation angles produced non-rigid scale/shear')
        reference = rotation_exp(vector.detach().double()).float()
        assert torch.equal(rotation, reference)
        (rotation * rotation.new_tensor([[1., 2., 3.], [4., 2., 5.], [6., 7., 3.]])).sum().backward()
        assert vector.grad.dtype == vector.dtype and torch.isfinite(vector.grad).all()


def test_rotation_rodrigues_matches_matrix_exponential_reference_in_normal_and_small_angle_charts():
    from tricoopdiff.geometry.rigid import rotation_exp

    for scale in (0., 1e-8, .999e-4, 1.001e-4, .1, 1., 10., 100.):
        vector = torch.tensor([[scale, .3 * scale, -.7 * scale]], dtype=torch.float64, requires_grad=True)
        skew = vector.new_zeros(1, 3, 3)
        x, y, z = vector.unbind(-1)
        skew[:, 0, 1], skew[:, 0, 2] = -z, y
        skew[:, 1, 0], skew[:, 1, 2] = z, -x
        skew[:, 2, 0], skew[:, 2, 1] = -y, x
        actual, expected = rotation_exp(vector), torch.linalg.matrix_exp(skew)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=0.)
        probe = actual.new_tensor([[1., 2., 3.], [4., 2., 5.], [6., 7., 3.]])
        actual_gradient, = torch.autograd.grad((actual * probe).sum(), vector, retain_graph=True)
        expected_gradient, = torch.autograd.grad((expected * probe).sum(), vector)
        assert torch.isfinite(actual_gradient).all()
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-11, rtol=0.)


def test_large_angle_rigid_decoder_preserves_distance_and_valid_body_over_repeated_steps():
    from tricoopdiff.geometry.rigid import is_proper_rotation
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture(torch.float32)
    original_distances = torch.cdist(poi, poi, compute_mode='donot_use_mm_for_euclid_dist')
    velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE, 'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                  'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_tensor([[1e4, 3e3, -7e3]])},
                  'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_tensor([[-7e3, 1e4, 3e3]])},
                  'torsion': ligand.new_empty(1, 0)}
    center = poi.mean(1)
    for _ in range(20):
        ligand, poi, body = SDPIFlow.step(ligand, poi, body, velocities, ligand.new_ones(1), data)
        assert is_proper_rotation(body, atol=5e-6)
    distances = torch.cdist(poi, poi, compute_mode='donot_use_mm_for_euclid_dist')
    assert torch.allclose(distances, original_distances, atol=5e-6, rtol=0.)
    assert torch.allclose(poi.mean(1), center, atol=5e-6, rtol=0.)


def test_continuous_horizontal_multitorsion_large_angles_have_correct_coordinate_gradients():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    _, data, _, poi, _ = flow_example('CCCOCC')
    ligand = data['coords'].clone().requires_grad_()
    body = torch.eye(3, dtype=ligand.dtype)[None]
    k = int(data['edge_mask'].sum())
    assert k >= 2, 'this regression requires coupled rotatable bonds'
    torsion = torch.linspace(-30., 40., k, dtype=ligand.dtype)[None].requires_grad_()

    def decode(coords, values):
        velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                      'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                      'e3_ligand': {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)},
                      'poi_ligand': {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)},
                      'torsion': values}
        return SDPIFlow.step(coords, poi, body, velocities, coords.new_tensor([.23]), data)[0]

    assert torch.autograd.gradcheck(decode, (ligand, torsion), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_continuous_horizontal_spin_prior_preserves_geometric_scale_and_torsion_gradient():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    ligand, poi, body, data = sdpi_flow_folded_torsion_decoder_fixture()
    reference = reference_gradient = None
    for scale in (1., 1e-5, 1e5):
        coordinates = ligand * scale
        angle = ligand.new_tensor([[2.]], requires_grad=True)
        velocities = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                      'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                      'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                      'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                      'torsion': angle}
        output = SDPIFlow.step(coordinates, poi * scale, body, velocities, ligand.new_tensor([.17]), data)[0] / scale
        gradient, = torch.autograd.grad((output * torch.arange(12, dtype=ligand.dtype).reshape_as(output)).sum(), angle)
        if reference is None:
            reference, reference_gradient = output, gradient
        assert torch.allclose(output, reference, atol=1e-10, rtol=0.)
        assert torch.isfinite(gradient).all()
        assert torch.allclose(gradient, reference_gradient, atol=1e-9, rtol=0.)


def test_rotation_exponential_handles_representable_float64_norm_without_squaring_overflow():
    from tricoopdiff.geometry.rigid import is_proper_rotation, rotation_exp

    vector = torch.tensor([[1e200, 3e199, -7e199]], dtype=torch.float64, requires_grad=True)
    rotation = rotation_exp(vector)
    assert torch.isfinite(rotation).all() and is_proper_rotation(rotation, atol=1e-12)
    probe = rotation.new_tensor([[1., 2., 3.], [4., 2., 5.], [6., 7., 3.]])
    (rotation * probe).sum().backward()
    assert torch.isfinite(vector.grad).all()
    for value in (torch.tensor([[torch.finfo(torch.float64).max] * 3], dtype=torch.float64),
                  torch.tensor([[float('inf'), 0., 0.]], dtype=torch.float64)):
        try:
            rotation_exp(value.double())
        except ValueError as error:
            assert 'norm' in str(error)
        else:
            raise AssertionError('unrepresentable rotation-vector norm was silently accepted')


def test_horizontal_ridge_prior_has_no_first_order_rotor_orientation_dependence():
    from rdkit import Chem
    from tricoopdiff.initialization import validate_ligand_graph
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    flow, data, _, poi, _ = flow_example()
    edges = data['edge_index']
    selected = torch.where(data['edge_mask'])[0]
    reversed_indices = [int(torch.where((edges[0] == edges[1, i]) &
                                        (edges[1] == edges[0, i]))[0]) for i in selected]
    order = np.argsort(reversed_indices)
    reversed_mask = torch.zeros_like(data['edge_mask'])
    reversed_mask[reversed_indices] = True
    reversed_data = {**data, 'edge_mask': reversed_mask,
                     'mask_rotate': (~data['mask_rotate'])[order]}
    validate_ligand_graph(Chem.Mol(data['mol']), edges, reversed_mask,
                          reversed_data['mask_rotate'])
    ligand = data['coords']
    body = torch.eye(3, dtype=ligand.dtype)[None]
    speeds = torch.linspace(-2., 3., len(selected), dtype=ligand.dtype)[None]
    velocity = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                'e3_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                'poi_ligand': {'tr': ligand.new_zeros(1, 3), 'rot': ligand.new_zeros(1, 3)},
                'torsion': speeds}
    reverse_velocity = {**velocity, 'torsion': speeds[:, order]}
    errors = []
    for h in (1e-5, 5e-6):
        a = flow.step(ligand, poi, body, velocity, ligand.new_tensor([h]), data)[0]
        b = flow.step(ligand, poi, body, reverse_velocity, ligand.new_tensor([h]), reversed_data)[0]
        errors.append(float((a - b).abs().max()))
    assert errors[1] < .3 * errors[0], 'rotor orientation left a first-order gauge residual'
    assert errors[1] / 5e-6 < 2e-5


def test_horizontal_ridge_prior_multistep_near_collinear_gradients_match_finite_difference():
    from tricoopdiff.sdpi_flow import CONTINUOUS_TORSION_MODE, INDEPENDENT_RIGID_MODE

    flow, data, _, poi, _ = flow_example()
    n = data['coords'].shape[1]
    index = torch.arange(n, dtype=torch.float64)
    body = torch.eye(3, dtype=torch.float64)[None]
    probe = torch.arange(n * 3, dtype=torch.float64).reshape(1, n, 3) / (n * 3)
    for epsilon in (1e-3, 0.):
        coords = torch.stack((index, index.sin() * epsilon, index.cos() * epsilon), -1)[None]
        coords.requires_grad_()
        speeds = torch.linspace(-9., 13., int(data['edge_mask'].sum()), dtype=coords.dtype)[None]
        speeds.requires_grad_()

        def decode(initial, parameters):
            ligand = initial
            for _ in range(20):
                velocity = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                            'torsion_update_mode': CONTINUOUS_TORSION_MODE,
                            'e3_ligand': {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)},
                            'poi_ligand': {'tr': coords.new_zeros(1, 3), 'rot': coords.new_zeros(1, 3)},
                            'torsion': parameters * (1 + .05 * ligand[:, :, 2].mean().tanh())}
                ligand, _, _ = flow.step(ligand, poi, body, velocity, coords.new_tensor([.05]), data)
            return (ligand * probe).sum()

        value = decode(coords, speeds)
        gx, gv = torch.autograd.grad(value, (coords, speeds))
        assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
        dx, dv = probe / probe.norm(), torch.ones_like(speeds) / speeds.numel() ** .5
        h = 1e-7
        difference = (decode(coords + h * dx, speeds + h * dv) -
                      decode(coords - h * dx, speeds - h * dv)) / (2 * h)
        exact = (gx * dx).sum() + (gv * dv).sum()
        assert torch.allclose(difference, exact, atol=2e-6, rtol=1e-5)


# Trajectory experiment

def trajectory_experiment_example(dtype=torch.float64):
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
    ligand, poi, body, ligand_data, velocities = trajectory_experiment_example()
    dt = torch.tensor([.05], dtype=ligand.dtype)
    expected = SDPIFlow.step(
        ligand, poi, body, velocities, dt, ligand_data)
    actual = differentiable_sdpi_step(
        ligand, poi, body, velocities, dt, ligand_data)
    for value, reference in zip(actual, expected):
        assert torch.allclose(value, reference, atol=2e-7, rtol=2e-7), (
            value - reference).abs().max()


def test_remaining_centroid_loss_uses_exact_independent_motion_and_gradients():
    from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE

    ligand, poi, body, data, velocities = trajectory_experiment_example()
    velocities['rigid_output_mode'] = INDEPENDENT_RIGID_MODE
    time = ligand.new_tensor([.4])
    scales = ligand.new_tensor([10., 15.])
    target_ligand, target_poi, _ = SDPIFlow.step(ligand, poi, body, velocities, time, data)
    loss = remaining_centroid_loss(ligand, poi, velocities, time, target_ligand, target_poi, scales)
    assert loss < 1e-20, 'rotation and torsion must not change the supervised centroid'
    rotation = torch.linalg.matrix_exp(torch.tensor([[0., -.3, .2], [.3, 0., -.1], [-.2, .1, 0.]], dtype=ligand.dtype))
    moved = {**velocities, **{b: {**velocities[b], 'tr': velocities[b]['tr'] @ rotation.T}
                            for b in ('e3_ligand', 'poi_ligand')}}
    reference = remaining_centroid_loss(ligand, poi, velocities, time, ligand, poi, scales)
    transformed = remaining_centroid_loss(ligand @ rotation.T + 10, poi @ rotation.T + 10,
                                         moved, time, ligand @ rotation.T + 10,
                                         poi @ rotation.T + 10, scales)
    assert torch.allclose(reference, transformed, atol=1e-12)
    for branch in ('e3_ligand', 'poi_ligand'):
        velocities[branch]['tr'] = velocities[branch]['tr'].clone().requires_grad_()
    loss = remaining_centroid_loss(ligand, poi, velocities, time, ligand, poi, scales)
    loss.backward()
    for branch in ('e3_ligand', 'poi_ligand'):
        grad = velocities[branch]['tr'].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    zero = remaining_centroid_loss(ligand, poi, velocities, time * 0, ligand, poi, scales)
    assert zero == 0 and torch.isfinite(zero), 'no inverse-time singularity is allowed'


def test_remaining_centroid_loss_rejects_legacy_motion():
    ligand, poi, _, _, velocities = trajectory_experiment_example()
    try:
        remaining_centroid_loss(ligand, poi, velocities, ligand.new_tensor([.5]),
                                ligand, poi, ligand.new_tensor([10., 15.]))
    except ValueError as error:
        assert 'independent' in str(error)
    else:
        raise AssertionError('dual-interface velocities do not independently move the POI centroid')


def test_gradients_cross_two_differentiable_steps():
    ligand, poi, body, ligand_data, velocities = trajectory_experiment_example()
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
    ligand, poi, body, ligand_data, velocities = trajectory_experiment_example()
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


def test_no_torsion_output_can_disable_existing_rotatable_bonds():
    ligand, poi, body, ligand_data, velocities = trajectory_experiment_example()
    assert bool(ligand_data['edge_mask'].any())
    velocities['torsion'] = ligand.new_empty((1, 0))
    dt = ligand.new_tensor([.05])
    actual = SDPIFlow.differentiable_step(ligand, poi, body, velocities, dt, ligand_data)
    expected = SDPIFlow.step(ligand, poi, body, velocities, dt, ligand_data)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(actual, expected))


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


def test_staged_poi_loss_jointly_penalizes_center_and_orientation():
    poi = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]],
        dtype=torch.float64, requires_grad=True)
    target = poi.detach() + torch.tensor([[[2., -1., 3.]]], dtype=torch.float64)
    native = target + torch.tensor([[[20., 0., 0.]]], dtype=torch.float64)
    body = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).requires_grad_()
    target_body = body.detach().clone()
    target_body[:, 0, 0] = 0
    loss, parts = staged_poi_state_loss(
        poi, target, native, body, target_body, poi.new_tensor(10.),
        distance_weighted=True, distance_weight_cap=3.)
    assert parts['center'] > 0 and parts['orientation'] > 0
    assert 1 < parts['distance_weight'] <= 3
    loss.backward()
    assert float(poi.grad.abs().sum()) > 0
    assert float(body.grad.abs().sum()) > 0


def test_staged_poi_contracts_are_distinct_and_complete():
    for mode, weighted in (
            (STAGED_POI_CONTRACT_MODE, False),
            (DISTANCE_STAGED_POI_CONTRACT_MODE, True)):
        trainer = StagedPOITrajectoryTrainer.__new__(StagedPOITrajectoryTrainer)
        trainer.trajectory_contract_mode = mode
        trainer.trajectory_steps = 5
        trainer.trajectory_weight = 1.
        trainer.trajectory_min_time = .25
        trainer.trajectory_max_time = .65
        trainer.distance_weighted = weighted
        trainer.distance_weight_cap = 3.
        trainer.activation_clamp = 100.
        trainer.activation_clamp_mode = 'irrep_norm'
        trainer.residual_weight = 1e-3
        trainer.ema_decay = .999
        trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
        contract = trainer.runtime_contract()
        assert contract['trajectory_objective'] == mode
        assert contract['trajectory_max_time'] == .65
        assert contract['trajectory_distance_weighted'] is weighted
        assert contract['trajectory_distance_weight_cap'] == 3.


def test_prior_replay_schedule_reaches_requested_time():
    time = torch.tensor([.43], dtype=torch.float64)
    count, step = prior_replay_schedule(
        time, inference_steps=20, max_steps=15)
    assert count == 12
    assert torch.allclose(torch.ones_like(time) - count * step, time)
    capped_count, capped_step = prior_replay_schedule(
        time, inference_steps=20, max_steps=4)
    assert capped_count == 4
    assert torch.allclose(torch.ones_like(time) - capped_count * capped_step, time)
    endpoint_count, endpoint_step = prior_replay_schedule(
        torch.ones_like(time), inference_steps=20, max_steps=15)
    assert endpoint_count == 0
    assert torch.equal(endpoint_step, torch.zeros_like(time))


def test_prior_replay_contract_records_exposure_policy():
    trainer = PriorReplayTrajectoryTrainer.__new__(PriorReplayTrajectoryTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .25
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_CONTRACT_MODE
    assert contract['trajectory_replay_start_time'] == 1.
    assert contract['trajectory_replay_max_steps'] == 15
    assert contract['trajectory_replay_gradients'] is False


def test_deterministic_prior_replay_uses_eval_and_restores_training():
    trainer = DeterministicPriorReplayTrajectoryTrainer.__new__(
        DeterministicPriorReplayTrajectoryTrainer)
    trainer.model = torch.nn.Sequential(torch.nn.Dropout(.5))
    trainer.model.train()
    with trainer._replay_context():
        assert not trainer.model.training
    assert trainer.model.training

    trainer.trajectory_contract_mode = PRIOR_REPLAY_EVAL_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .25
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_EVAL_CONTRACT_MODE
    assert contract['trajectory_replay_model_mode'] == 'eval'


def test_joint_endpoint_translation_loss_only_differentiates_translations():
    _, _, _, _, velocities = trajectory_experiment_example()
    targets = {}
    leaves = []
    for branch in ('e3_ligand', 'poi_ligand'):
        targets[branch] = {}
        for component in ('tr', 'rot'):
            value = velocities[branch][component].clone().requires_grad_()
            velocities[branch][component] = value
            leaves.append((component, value))
            targets[branch][component] = torch.zeros_like(value)
    loss = joint_endpoint_translation_loss(
        velocities, targets,
        (torch.tensor(2., dtype=torch.float64),
         torch.tensor(4., dtype=torch.float64)))
    loss.backward()
    for component, value in leaves:
        if component == 'tr':
            assert value.grad is not None
            assert float(value.grad.abs().sum()) > 0
        else:
            assert value.grad is None


def test_prior_replay_translation_contract_is_explicit():
    trainer = PriorReplayJointTranslationTrainer.__new__(
        PriorReplayJointTranslationTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = .1
    trainer.trajectory_min_time = .3
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE
    assert contract['trajectory_min_time'] == .3
    assert contract['trajectory_auxiliary_branches'] == [
        'e3_ligand.tr', 'poi_ligand.tr']
    assert contract['trajectory_auxiliary_state_gradients'] is False
    assert contract['trajectory_auxiliary_native_reading'] == 'training_only'
    assert contract['trajectory_rotation_torsion_target'] == 'analytic_teacher_only'


def test_prior_replay_translation_replacement_contract_is_explicit():
    trainer = PriorReplayJointTranslationReplaceTrainer.__new__(
        PriorReplayJointTranslationReplaceTrainer)
    trainer.trajectory_contract_mode = PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .3
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == (
        PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE)
    assert contract['trajectory_translation_teacher_policy'] == (
        'replace_when_prior_replay_target_available')


def test_prior_replay_translation_window_contract_is_explicit():
    trainer = PriorReplayJointTranslationWindowReplaceTrainer.__new__(
        PriorReplayJointTranslationWindowReplaceTrainer)
    trainer.trajectory_contract_mode = (
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE)
    trainer.trajectory_steps = 3
    trainer.trajectory_weight = 1.
    trainer.trajectory_min_time = .3
    trainer.trajectory_max_time = .65
    trainer.replay_max_steps = 15
    trainer.activation_clamp = 100.
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 5e-5, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['trajectory_objective'] == (
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE)
    assert contract['trajectory_min_time'] == .3
    assert contract['trajectory_max_time'] == .65
    assert contract['trajectory_translation_teacher_policy'] == (
        'replace_only_inside_prior_replay_time_window')


def test_prior_replay_translation_window_keeps_early_teacher_target():
    trainer = PriorReplayJointTranslationWindowReplaceTrainer.__new__(
        PriorReplayJointTranslationWindowReplaceTrainer)
    trainer.trajectory_max_time = .65
    trainer._translation_auxiliary_applied = True
    time = torch.tensor([.8])
    trainer._trajectory_context = (None, None, None, None, time, None, None)
    loss = trainer._trajectory_loss()
    assert loss.item() == 0
    assert not loss.requires_grad
    assert trainer._translation_auxiliary_applied is False


def test_soft_time_translation_mix_has_expected_endpoints():
    trainer = PriorReplayJointTranslationSoftTimeTrainer.__new__(
        PriorReplayJointTranslationSoftTimeTrainer)
    trainer.endpoint_zero_time = .8
    trainer.endpoint_full_time = .3
    for time, expected in ((.8, 0.), (.55, .5), (.3, 1.)):
        trainer._trajectory_context = (
            None, None, None, None, torch.tensor([time]), None, None)
        assert torch.allclose(
            trainer._translation_replacement_mix(), torch.tensor(expected))


def test_direction_translation_uses_teacher_magnitude():
    trainer = PriorReplayJointTranslationDirectionTrainer.__new__(
        PriorReplayJointTranslationDirectionTrainer)
    endpoint = torch.tensor([[0., 2., 0.]])
    analytic = torch.tensor([[3., 0., 0.]])
    prediction = torch.tensor([[0., 3., 0.]])
    branches = ('e3_ligand', 'poi_ligand')
    velocities = {branch: {'tr': prediction} for branch in branches}
    endpoint_targets = {branch: {'tr': endpoint} for branch in branches}
    analytic_targets = {branch: {'tr': analytic} for branch in branches}
    loss = trainer._translation_loss(
        velocities, endpoint_targets, (torch.ones(1), torch.ones(1)),
        analytic_targets)
    assert torch.allclose(loss, torch.zeros(()))


def test_agreement_translation_falls_back_for_opposite_target():
    trainer = PriorReplayJointTranslationAgreementTrainer.__new__(
        PriorReplayJointTranslationAgreementTrainer)
    endpoint = torch.tensor([[-2., 0., 0.]])
    analytic = torch.tensor([[3., 0., 0.]])
    branches = ('e3_ligand', 'poi_ligand')
    velocities = {branch: {'tr': analytic} for branch in branches}
    endpoint_targets = {branch: {'tr': endpoint} for branch in branches}
    analytic_targets = {branch: {'tr': analytic} for branch in branches}
    loss = trainer._translation_loss(
        velocities, endpoint_targets, (torch.ones(1), torch.ones(1)),
        analytic_targets)
    assert torch.allclose(loss, torch.zeros(()))


def test_new_translation_contract_names_are_distinct():
    assert len({
        PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE,
        PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE,
    }) == 5


# Joint endpoint experiment

ENDPOINT_DTYPE = torch.float64


def joint_endpoint_experiment_raises(message, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f'expected ValueError containing {message!r}')


def joint_endpoint_experiment_random_rotation(generator):
    axis = torch.randn(3, dtype=ENDPOINT_DTYPE, generator=generator)
    axis /= axis.norm()
    angle = (torch.rand((), dtype=ENDPOINT_DTYPE, generator=generator) * 2 - 1) * 2.8
    return _rotation_exp(axis * angle)


def joint_endpoint_experiment_fixture(seed=0):
    generator = torch.Generator().manual_seed(seed)
    native_ligand = torch.randn(1, 13, 3, dtype=ENDPOINT_DTYPE, generator=generator) * torch.tensor(
        [2., 3., 4.], dtype=ENDPOINT_DTYPE)
    native_poi = torch.randn(1, 19, 3, dtype=ENDPOINT_DTYPE, generator=generator) * torch.tensor(
        [5., 7., 9.], dtype=ENDPOINT_DTYPE)
    ra, rb = joint_endpoint_experiment_random_rotation(generator)[None], joint_endpoint_experiment_random_rotation(generator)[None]
    ta, tb = (torch.randn(1, 3, dtype=ENDPOINT_DTYPE, generator=generator) * 10 for _ in range(2))
    ligand = apply_rigid(native_ligand, ra, ta)
    poi, body = assemble_interface_pair(native_poi, ra, ta, rb, tb)
    data = {
        'coords': native_ligand,
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'edge_mask': torch.empty(0, dtype=torch.bool),
        'mask_rotate': torch.empty((0, native_ligand.shape[1]), dtype=torch.bool),
    }
    return ligand, poi, body, native_ligand, native_poi, data


def test_exact_one_step_recovers_joint_endpoint_across_random_poses():
    for seed in range(24):
        ligand, poi, body, native_ligand, native_poi, data = joint_endpoint_experiment_fixture(seed)
        time = torch.tensor([0.17 + (seed % 7) * 0.1], dtype=ENDPOINT_DTYPE)
        target, diagnostic = joint_rigid_endpoint_targets(
            ligand, poi, body, native_ligand, native_poi, time)
        result = SDPIFlow.step(ligand, poi, body, target, time, data)
        assert torch.allclose(result[0], native_ligand, atol=2e-9, rtol=0)
        assert torch.allclose(result[1], native_poi, atol=2e-9, rtol=0)
        assert torch.allclose(result[2], torch.eye(3, dtype=ENDPOINT_DTYPE)[None], atol=2e-9, rtol=0)
        assert diagnostic['ligand_fit_rmsd'].max() < 1e-12
        assert diagnostic['poi_fit_rmsd'].max() < 1e-12


def test_target_is_equivariant_under_global_rigid_motion():
    ligand, poi, body, native_ligand, native_poi, _ = joint_endpoint_experiment_fixture(31)
    generator = torch.Generator().manual_seed(91)
    q = joint_endpoint_experiment_random_rotation(generator)[None]
    shift = torch.tensor([[80., -31., 12.]], dtype=ENDPOINT_DTYPE)
    original, _ = joint_rigid_endpoint_targets(
        ligand, poi, body, native_ligand, native_poi, torch.tensor([.6], dtype=ENDPOINT_DTYPE))
    moved, _ = joint_rigid_endpoint_targets(
        apply_rigid(ligand, q, shift), apply_rigid(poi, q, shift), q @ body @ q.transpose(-1, -2),
        apply_rigid(native_ligand, q, shift), apply_rigid(native_poi, q, shift),
        torch.tensor([.6], dtype=ENDPOINT_DTYPE))
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert torch.allclose(moved[branch][component],
                                  original[branch][component] @ q.transpose(-1, -2),
                                  atol=2e-9, rtol=0)


def test_local_step_matches_cartesian_finite_difference_tangent():
    ligand, poi, body, native_ligand, native_poi, data = joint_endpoint_experiment_fixture(8)
    time = torch.tensor([.7], dtype=ENDPOINT_DTYPE)
    target, _ = joint_rigid_endpoint_targets(
        ligand, poi, body, native_ligand, native_poi, time)
    h = 1e-6
    plus = SDPIFlow.step(ligand, poi, body, target, torch.tensor([h], dtype=ENDPOINT_DTYPE), data)
    # Directly evaluate the same smooth one-parameter branch at negative h.
    center = ligand.mean(1)
    transforms = []
    for branch in ('e3_ligand', 'poi_ligand'):
        rotation = _rotation_exp(torch.tensor([h], dtype=ENDPOINT_DTYPE)[:, None] * target[branch]['rot'])
        destination = center + h * target[branch]['tr']
        translation = destination - (rotation @ center[..., None]).squeeze(-1)
        transforms.append((rotation, translation))
    minus_ligand = apply_rigid(ligand, *transforms[0])
    minus_poi, _ = assemble_interface_pair(poi, *transforms[0], *transforms[1])
    derivative = ((plus[0] - minus_ligand) / (2 * h), (plus[1] - minus_poi) / (2 * h))
    for step in (2e-4, 1e-4):
        actual = SDPIFlow.step(ligand, poi, body, target, torch.tensor([step], dtype=ENDPOINT_DTYPE), data)
        errors = [((value - start) / step - tangent).abs().max()
                  for value, start, tangent in zip(actual[:2], (ligand, poi), derivative)]
        if step == 2e-4:
            coarse = max(errors)
        else:
            assert max(errors) < coarse * .51


def test_low_time_is_rejected_or_velocity_is_bounded():
    fixture = joint_endpoint_experiment_fixture(12)
    args = fixture[:5] + (torch.tensor([1e-5], dtype=ENDPOINT_DTYPE),)
    joint_endpoint_experiment_raises('minimum', joint_rigid_endpoint_targets, *args, min_time=.1)
    bounded, diagnostic = joint_rigid_endpoint_targets(*args, min_time=.1, low_time='clamp')
    reference, _ = joint_rigid_endpoint_targets(
        *fixture[:5], torch.tensor([.1], dtype=ENDPOINT_DTYPE))
    for branch in ('e3_ligand', 'poi_ligand'):
        for component in ('tr', 'rot'):
            assert torch.allclose(bounded[branch][component], reference[branch][component])
    assert diagnostic['low_time_clamped'].item()
    assert diagnostic['effective_time'].item() == .1


def test_blend_and_per_component_correction_caps_are_explicit():
    fixture = joint_endpoint_experiment_fixture(5)
    time = torch.tensor([.4], dtype=ENDPOINT_DTYPE)
    oracle, _ = joint_rigid_endpoint_targets(*fixture[:5], time)
    teacher = {branch: {component: torch.zeros_like(oracle[branch][component])
                        for component in ('tr', 'rot')}
               for branch in ('e3_ligand', 'poi_ligand')}
    bounded, diagnostic = joint_rigid_endpoint_targets(
        *fixture[:5], time, teacher_targets=teacher, blend=.25,
        correction_cap={'tr': .5, 'rot': .2})
    for branch in ('e3_ligand', 'poi_ligand'):
        assert bounded[branch]['tr'].norm(dim=-1).max() <= .125 + 1e-12
        assert bounded[branch]['rot'].norm(dim=-1).max() <= .05 + 1e-12
        for component in ('tr', 'rot'):
            key = f'correction_norm_{branch}_{component}'
            assert torch.allclose(diagnostic[key], oracle[branch][component].norm(dim=-1))
    assert diagnostic['bounded_target'].item()


def test_pi_branch_is_rejected_by_default_but_stable_log_is_finite():
    native_ligand = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                                   [0., 0., 3.]]], dtype=ENDPOINT_DTYPE)
    native_poi = native_ligand * 2 + 4
    rotation = _rotation_exp(torch.tensor([[math.pi, 0., 0.]], dtype=ENDPOINT_DTYPE))
    ligand = apply_rigid(native_ligand, rotation, torch.zeros(1, 3, dtype=ENDPOINT_DTYPE))
    body = torch.eye(3, dtype=ENDPOINT_DTYPE)[None]
    joint_endpoint_experiment_raises('branch cut', joint_rigid_endpoint_targets,
            ligand, native_poi, body, native_ligand, native_poi,
            torch.tensor([.5], dtype=ENDPOINT_DTYPE))
    log = stable_so3_log(rotation, reject_near_pi=False)
    assert torch.isfinite(log).all()
    assert torch.allclose(log.norm(dim=-1), torch.tensor([math.pi], dtype=ENDPOINT_DTYPE), atol=1e-9)
    assert torch.allclose(_rotation_exp(log), rotation, atol=1e-9, rtol=0)


def test_shape_or_body_inconsistency_is_rejected_in_exact_mode():
    ligand, poi, body, native_ligand, native_poi, _ = joint_endpoint_experiment_fixture(4)
    changed = native_ligand.clone()
    changed[:, 0] += torch.tensor([2., -1., 3.], dtype=ENDPOINT_DTYPE)
    joint_endpoint_experiment_raises('rigid-equivalent', joint_rigid_endpoint_targets,
            ligand, poi, body, changed, native_poi, torch.tensor([.5], dtype=ENDPOINT_DTYPE))
    joint_endpoint_experiment_raises('body frame', joint_rigid_endpoint_targets,
            ligand, poi, torch.eye(3, dtype=ENDPOINT_DTYPE)[None], native_ligand, native_poi,
            torch.tensor([.5], dtype=ENDPOINT_DTYPE))


# Offpath experiments

def offpath_experiments_rotation(vector):
    return _rotation_exp(torch.as_tensor(vector, dtype=torch.float64).reshape(1, 3))


def offpath_experiments_targets(dtype=torch.float64):
    return {
        'e3_ligand': {
            'tr': torch.tensor([[7., 8., 9.]], dtype=dtype),
            'rot': torch.tensor([[1., 4., 2.]], dtype=dtype),
        },
        'poi_ligand': {
            'tr': torch.tensor([[2., -3., 5.]], dtype=dtype),
            'rot': torch.tensor([[-4., 1., 6.]], dtype=dtype),
        },
        'torsion': torch.tensor([[.2, -.7]], dtype=dtype),
    }


def test_identity_when_teacher_and_rollout_bodies_equal():
    body = offpath_experiments_rotation([.3, -.5, .7])
    original = offpath_experiments_targets()
    result = transport_poi_body_frame_targets(original, body, body)
    assert torch.allclose(result['poi_ligand']['tr'], original['poi_ligand']['tr'], atol=1e-12)
    assert torch.allclose(result['poi_ligand']['rot'], original['poi_ligand']['rot'], atol=1e-12)
    assert result['e3_ligand'] is original['e3_ligand']
    assert result['torsion'] is original['torsion']


def test_transport_is_globally_rotation_equivariant():
    teacher_body = offpath_experiments_rotation([.2, -.4, .1])
    rollout_body = offpath_experiments_rotation([-.7, .3, .5])
    global_rotation = offpath_experiments_rotation([.6, .1, -.2])
    original = offpath_experiments_targets()
    result = transport_poi_body_frame_targets(original, teacher_body, rollout_body)
    rotated_targets = {
        **original,
        'poi_ligand': {
            **original['poi_ligand'],
            'tr': torch.einsum('...i,...ij->...j', original['poi_ligand']['tr'],
                               global_rotation.transpose(-1, -2)),
            'rot': torch.einsum('...i,...ij->...j', original['poi_ligand']['rot'],
                                global_rotation.transpose(-1, -2)),
        },
    }
    rotated = transport_poi_body_frame_targets(
        rotated_targets, global_rotation @ teacher_body, global_rotation @ rollout_body)
    for component in ('tr', 'rot'):
        expected = torch.einsum('...i,...ij->...j', result['poi_ligand'][component],
                                global_rotation.transpose(-1, -2))
        assert torch.allclose(rotated['poi_ligand'][component], expected, atol=1e-12)


def test_transport_preserves_rigid_label_norms():
    original = offpath_experiments_targets()
    result = transport_poi_body_frame_targets(
        original, offpath_experiments_rotation([.8, -.2, .4]), offpath_experiments_rotation([-.1, .9, -.3]))
    for component in ('tr', 'rot'):
        before = torch.linalg.vector_norm(original['poi_ligand'][component], dim=-1)
        after = torch.linalg.vector_norm(result['poi_ligand'][component], dim=-1)
        assert torch.allclose(after, before, atol=1e-12)


def offpath_experiments_torsion_example():
    teacher = torch.tensor([[[0., 0., 0.], [1., 0., 0.],
                             [1., 1., 0.], [2., 1., .5]]], dtype=torch.float64)
    rollout = teacher.clone()
    rollout[0, 2:] = rollout[0, 2:] @ offpath_experiments_rotation([.6, 0., 0.])[0].T
    coupling = {
        'bonds': torch.tensor([[0, 1]]).numpy(),
        'masks': torch.tensor([[False, False, True, True]]).numpy(),
    }
    return teacher, rollout, coupling


def test_gauge_correct_transport_removes_and_recomputes_common_spin():
    teacher_ligand, rollout_ligand, coupling = offpath_experiments_torsion_example()
    teacher_body, rollout_body = offpath_experiments_rotation([.3, -.2, .1]), offpath_experiments_rotation([-.4, .2, .5])
    original = offpath_experiments_targets()
    teacher_gauge = _state_torsion_gauge(
        teacher_ligand, coupling, original['torsion'][:, :1])
    rollout_gauge = _state_torsion_gauge(
        rollout_ligand, coupling, original['torsion'][:, :1])
    # Match the one active bond in this synthetic graph.
    one_torsion = {**original, 'torsion': original['torsion'][:, :1]}
    result = transport_targets_with_recomputed_gauge(
        one_torsion, teacher_body, rollout_body,
        teacher_ligand, rollout_ligand, coupling)
    assert torch.allclose(result['e3_ligand']['rot'],
                          original['e3_ligand']['rot'] - teacher_gauge + rollout_gauge,
                          atol=1e-12)
    old_global = torch.einsum('...i,...ij->...j',
                              original['poi_ligand']['rot'], teacher_body)
    new_global = torch.einsum('...i,...ij->...j',
                              result['poi_ligand']['rot'], rollout_body)
    assert torch.allclose(new_global, old_global - teacher_gauge + rollout_gauge,
                          atol=1e-12)


def test_gauge_correct_contract_has_distinct_checkpoint_identity():
    trainer = GaugeCorrectBodyFrameTransportRolloutTrainer.__new__(
        GaugeCorrectBodyFrameTransportRolloutTrainer)
    trainer.offpath_contract_mode = GAUGE_CORRECT_CONTRACT_MODE
    trainer.rollout_probability = .5
    trainer.rollout_steps = 5
    trainer.rollout_min_time = .25
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_contract_mode'] == GAUGE_CORRECT_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == GAUGE_CORRECT_TORSION_CONTRACT


def test_bounded_joint_endpoint_changes_only_rigid_targets_within_caps():
    native_ligand, rollout_ligand, coupling = offpath_experiments_torsion_example()
    native_poi = torch.tensor(
        [[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]],
        dtype=torch.float64)
    poi_rotation = offpath_experiments_rotation([-.2, .3, .1])
    rollout_poi = native_poi @ poi_rotation.transpose(-1, -2) + torch.tensor(
        [[[2., -1., .5]]], dtype=torch.float64)
    teacher_body = offpath_experiments_rotation([.3, -.2, .1])
    rollout_body = offpath_experiments_rotation([-.4, .2, .5])
    original = offpath_experiments_targets()
    original = {**original, 'torsion': original['torsion'][:, :1]}
    teacher = (native_ligand, native_poi, teacher_body, original)

    trainer = BoundedJointEndpointRolloutTrainer.__new__(
        BoundedJointEndpointRolloutTrainer)
    trainer.rollout_min_time = .25
    trainer.endpoint_blend = .25
    trainer.endpoint_translation_cap = 2.
    trainer.endpoint_rotation_cap = .4
    trainer.joint_endpoint_applied_states = 0
    result = trainer._map_targets(
        teacher, rollout_ligand, rollout_poi, rollout_body, coupling,
        ligand_data={'coords': native_ligand},
        poi_data={'coords': native_poi},
        time=torch.tensor([.5], dtype=torch.float64))
    base = transport_targets_with_recomputed_gauge(
        original, teacher_body, rollout_body,
        native_ligand, rollout_ligand, coupling)

    assert trainer.joint_endpoint_applied_states == 1
    assert result['torsion'] is base['torsion']
    assert torch.equal(result['torsion'], original['torsion'])
    for branch in ('e3_ligand', 'poi_ligand'):
        for component, cap in (('tr', 2.), ('rot', .4)):
            change = torch.linalg.vector_norm(
                result[branch][component] - base[branch][component], dim=-1)
            assert bool((change <= .25 * cap + 1e-12).all()), (
                branch, component, change)


def test_bounded_joint_endpoint_contract_has_distinct_checkpoint_identity():
    trainer = BoundedJointEndpointRolloutTrainer.__new__(
        BoundedJointEndpointRolloutTrainer)
    trainer.offpath_contract_mode = JOINT_ENDPOINT_CONTRACT_MODE
    trainer.rollout_probability = .5
    trainer.rollout_steps = 5
    trainer.rollout_min_time = .25
    trainer.endpoint_blend = .25
    trainer.endpoint_translation_cap = 10.
    trainer.endpoint_rotation_cap = .5
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_target'] == JOINT_ENDPOINT_CONTRACT_MODE
    assert contract['offpath_contract_mode'] == JOINT_ENDPOINT_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == JOINT_ENDPOINT_TORSION_CONTRACT
    assert contract['endpoint_blend'] == .25
    assert contract['endpoint_translation_cap'] == 10.
    assert contract['endpoint_rotation_cap'] == .5


class FakeFlow:
    def __init__(self, teacher_body, rollout_body, teacher_targets):
        self.teacher_body = teacher_body
        self.rollout_body = rollout_body
        self.teacher_targets = teacher_targets
        self.path_calls = 0

    def path(self, ligand_data, poi_coords, coupling, time):
        self.path_calls += 1
        body = (self.teacher_body if torch.equal(time, torch.tensor([.5], dtype=time.dtype))
                else offpath_experiments_rotation([-.1, .2, .3]))
        ligand = ligand_data['coords'] + self.path_calls
        poi = poi_coords + self.path_calls
        return ligand, poi, body, self.teacher_targets

    def normalized_predictions(self, output, time):
        return output

    def velocities(self, normalized, e3_coords, poi_coords):
        return normalized

    def step(self, ligand, poi, body, velocities, dt, ligand_data):
        return ligand + 10, poi + 20, self.rollout_body


def test_training_state_maps_only_generated_offpath_poi_targets():
    teacher_body = offpath_experiments_rotation([.4, -.3, .2])
    rollout_body = offpath_experiments_rotation([-.6, .1, .5])
    original = offpath_experiments_targets()
    trainer = BodyFrameTransportRolloutTrainer.__new__(BodyFrameTransportRolloutTrainer)
    trainer.diffusion = FakeFlow(teacher_body, rollout_body, original)
    trainer.rollout_probability = 1.0
    trainer.rollout_steps = 1
    trainer.rollout_min_time = 0.0
    trainer.rollout_augmented_states = 41  # Must not be used to identify this state.
    trainer._forward_state = lambda *args, **kwargs: {'deterministic': True}
    coords = torch.zeros(1, 2, 3, dtype=torch.float64)
    state = trainer._training_state(
        {'coords': coords}, {'coords': coords}, {'coords': coords}, {},
        torch.tensor([.5], dtype=torch.float64),
        torch.tensor([.05], dtype=torch.float64))
    expected = transport_poi_body_frame_targets(original, teacher_body, rollout_body)
    assert trainer.rollout_augmented_states == 42
    assert torch.equal(state[2], rollout_body)
    for component in ('tr', 'rot'):
        assert torch.allclose(state[3]['poi_ligand'][component],
                              expected['poi_ligand'][component], atol=1e-12)
    assert state[3]['e3_ligand'] is original['e3_ligand']
    assert state[3]['torsion'] is original['torsion']


def test_teacher_state_is_not_mapped_and_contract_records_mode():
    original = offpath_experiments_targets()
    trainer = BodyFrameTransportRolloutTrainer.__new__(BodyFrameTransportRolloutTrainer)
    trainer.diffusion = FakeFlow(offpath_experiments_rotation([.1, .2, .3]), offpath_experiments_rotation([.7, -.2, .4]), original)
    trainer.rollout_probability = 0.0
    trainer.rollout_steps = 1
    trainer.rollout_min_time = 0.0
    trainer.rollout_augmented_states = 99
    coords = torch.zeros(1, 2, 3, dtype=torch.float64)
    state = trainer._training_state(
        {'coords': coords}, {'coords': coords}, {'coords': coords}, {},
        torch.tensor([.5], dtype=torch.float64),
        torch.tensor([.05], dtype=torch.float64))
    assert state[3] is original
    assert trainer.rollout_augmented_states == 99

    trainer.offpath_contract_mode = OFFPATH_CONTRACT_MODE
    trainer.activation_clamp = 100.0
    trainer.activation_clamp_mode = 'irrep_norm'
    trainer.residual_weight = 1e-3
    trainer.ema_decay = .999
    trainer.optimizer_config = {'trifuse': {'lr': 1e-4, 'weight_decay': 1e-4}}
    contract = trainer.runtime_contract()
    assert contract['offpath_target'] == OFFPATH_CONTRACT_MODE
    assert contract['offpath_contract_mode'] == OFFPATH_CONTRACT_MODE
    assert contract['offpath_torsion_gauge'] == TORSION_GAUGE_CONTRACT
    assert contract['rollout_probability'] == 0.0
