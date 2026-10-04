"""CPU checks for training-stage contracts that must fail explicitly."""

import sys
import os
import random
import tempfile
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tricoopdiff.train import TriCoopDiffTrainer
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.sampler import actual_candidate_source
from train_entry import (
    TernaryDataLoader, build_trainer, parse_args, prepare_source_batch,
    swap_protein_reference, iter_training_batches, validate,
)


def _head():
    return torch.nn.Linear(4, 2)


def test_phase1_freezes_unsupervised_heads():
    trainer = SimpleNamespace(model=SimpleNamespace(
        trifuse=SimpleNamespace(judge_head=_head(), state_head=_head()),
    ))
    TriCoopDiffTrainer.set_phase(trainer, 1)
    assert not any(
        parameter.requires_grad
        for head in (trainer.model.trifuse.judge_head,
                     trainer.model.trifuse.state_head)
        for parameter in head.parameters()
    )


def test_undefined_phase_is_rejected():
    trainer = SimpleNamespace(model=SimpleNamespace(
        trifuse=SimpleNamespace(judge_head=_head(), state_head=_head()),
    ))
    try:
        TriCoopDiffTrainer.set_phase(trainer, 2)
    except ValueError as error:
        assert "labels and losses" in str(error)
    else:
        raise AssertionError("an unsupervised training phase must not run")


def test_protein_role_swap_changes_reference_without_changing_geometry():
    e3 = {'coords': torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0]]]), 'tag': 'e3'}
    poi = {'coords': torch.tensor([[[9.0, 0, 0], [11.0, 0, 0]]]), 'tag': 'poi'}
    ligand = {'coords': torch.tensor([[[4.0, 2.0, 0]]])}
    new_e3, new_poi, new_ligand = swap_protein_reference(e3, poi, ligand)
    assert new_e3['tag'] == 'poi'
    assert new_poi['tag'] == 'e3'
    assert torch.allclose(new_e3['coords'].mean(dim=1), torch.zeros(1, 3))
    assert torch.allclose(
        new_ligand['coords'] - new_poi['coords'][:, :1],
        ligand['coords'] - e3['coords'][:, :1],
    )


def test_featurization_cache_state_cannot_change_training_rng():
    loader = TernaryDataLoader.__new__(TernaryDataLoader)
    loader.score_model_args = object(); loader.device = 'cpu'; loader.cache_dir = 'unused'
    loader.cache_read_only = False; loader.cache_wait_seconds = 3600.0
    loader.random_flip_proteins = False
    entry = {'case': 'fixture', 'protein1': 'a', 'protein2': 'b', 'ligand': 'c'}

    def consuming(*args, **kwargs):
        torch.rand(5); np.random.random(5); random.random()
        data = {'coords': torch.zeros(1, 1, 3)}
        return data, data, data, 1

    torch.manual_seed(3); np.random.seed(3); random.seed(3)
    expected = (torch.rand(1), np.random.random(), random.random())
    torch.manual_seed(3); np.random.seed(3); random.seed(3)
    with patch('train_entry.featurize_complex', side_effect=consuming):
        loader._load_one(entry)
    actual = (torch.rand(1), np.random.random(), random.random())
    assert torch.equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]


def test_trainer_owns_recorded_activation_clamp_runtime():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trifuse = torch.nn.Linear(2, 2)
            self.expert = torch.nn.Module()
            self.expert.model = SimpleNamespace()

    with patch.dict('os.environ', {}, clear=True):
        trainer = TriCoopDiffTrainer(
            DummyModel(), SDPIFlow(), device='cpu', activation_clamp=37)
        assert trainer.activation_clamp == 37
        assert trainer.model.expert.model.activation_clamp == 37.
        assert 'NAN_CLAMP' not in os.environ


def test_consensus_builder_passes_only_active_model_dimensions():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused',
    ])
    expert = SimpleNamespace(num_conv_layers=11)
    with patch('tricoopdiff.models.expert.build_diffdock_expert', return_value=expert), \
            patch('tricoopdiff.models.consensus.ConsensusFuse') as fuse, \
            patch('tricoopdiff.models.consensus.ConsensusCoopDiff') as model, \
            patch('tricoopdiff.train.ConsensusTrainer') as trainer:
        result, _, trained = build_trainer(args, 'cpu', input_dim=1281)
        fuse.assert_called_once_with(
            node_dim=args.node_dim, input_dim=1281,
            num_transformer_layers=args.num_transformer_layers,
            direct_ternary_motion=True,
            global_node_context=True,
            spatial_geometry=True,
            motion_attention=True,
            global_node_attention=True,
            rotation_scale_normalization=True,
            dropout=0., source_conditioning=False,
            model_profile='ternary_endpoint_v1',
        )
        assert model.call_args.kwargs['expert_input_grad'] is True
        assert result is model.return_value
        assert trained is trainer.return_value
        assert trainer.call_args.kwargs['activation_clamp_mode'] == 'irrep_norm'
        assert trainer.call_args.kwargs['training_endpoint_protocol'] == 'none'
        assert trainer.call_args.kwargs['steric_loss_weight'] == .05
        assert args.resolved_model_config['dropout'] == 0


def _endpoint_trainer_fixture():
    from tricoopdiff.models.consensus import DIRECT_ARCHITECTURE
    from tricoopdiff.train import ConsensusTrainer
    from test_sdpi_flow import fixture

    flow, ligand, e3, poi, coupling = fixture()
    flow.inference_steps = 3

    class Model(torch.nn.Module):
        architecture = DIRECT_ARCHITECTURE
        expert_input_grad = True

        def __init__(self):
            super().__init__()
            self.trifuse = torch.nn.Module()
            self.trifuse.direct_ternary_motion = True
            self.trifuse.config = {'direct_ternary_motion': True, 'node_dim': 4}
            self.trifuse.motion = torch.nn.Parameter(torch.full((2, 6), .01, dtype=torch.float64))
            self.trifuse.torsion = torch.nn.Parameter(torch.full((1, int(ligand['edge_mask'].sum())), .01, dtype=torch.float64))
            self.trifuse.judge_head, self.trifuse.state_head = _head(), _head()
            self.expert = torch.nn.Module()
            self.expert.model = SimpleNamespace(no_torsion=False, dynamic_max_cross=False)
            self.expert.provenance = {'diffdock_commit': 'test', 'score_checkpoint_sha256': 'test', 'no_torsion': False}
            self.calls = []

        def forward(self, *args, **kwargs):
            self.calls.append((float(kwargs['t_e3l']), kwargs['source']))
            return {'normalized_velocity': {
                'rigid_output_mode': 'independent_centroids_v1',
                **{b: {'tr': self.trifuse.motion[i:i+1, :3],
                       'rot': self.trifuse.motion[i:i+1, 3:], 'tor': self.trifuse.torsion}
                   for i, b in enumerate(('e3_ligand', 'poi_ligand'))},
                'torsion': self.trifuse.torsion}}

    model = Model()
    trainer = ConsensusTrainer(model, flow, device='cpu', ema_decay=0.)
    batch = {'e3_data': {'coords': e3}, 'poi_data': {'coords': poi}, 'ligand_data': ligand}
    return trainer, batch, coupling


def test_direct_trainer_supervises_executed_endpoint_not_teacher_velocities():
    from tricoopdiff.train import ENDPOINT_OBJECTIVE

    trainer, batch, coupling = _endpoint_trainer_fixture()
    assert trainer.training_objective == ENDPOINT_OBJECTIVE
    result = trainer.compute_loss(**batch, coupling_override=coupling, diagnostics=True, source=1)
    assert len(trainer.model.calls) == 3
    assert all(source == 1 for _, source in trainer.model.calls)
    assert np.allclose([clock for clock, _ in trainer.model.calls], [1., 2/3, 1/3])
    assert set(result['components']) == {'loss_ligand_geometry', 'loss_protein_geometry',
                                         'loss_interface_geometry', 'loss_contact_geometry'}
    scales = trainer.diffusion.scales(batch['e3_data']['coords'], batch['poi_data']['coords'])
    expected = .1 * (((result['predicted_ligand'] - batch['ligand_data']['coords']) / scales[0]).square().mean()
                     + ((result['predicted_poi'] - batch['poi_data']['coords']) / scales[1]).square().mean()
                     + sum(result['components'][key] for key in ('loss_interface_geometry', 'loss_contact_geometry')))
    assert torch.allclose(result['loss_total'], expected)
    result['loss_total'].backward()
    for parameter in (trainer.model.trifuse.motion, trainer.model.trifuse.torsion):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(p.requires_grad for group in trainer.optimizer.param_groups for p in group['params'])


def test_endpoint_training_and_sampling_execute_identical_full_trajectory():
    from tricoopdiff.sampler import sample_single

    trainer, batch, coupling = _endpoint_trainer_fixture()
    flow = trainer.diffusion
    flow.inference_steps = 20
    state = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling,
                      batch['ligand_data']['coords'].new_ones(1))[:3]
    original_forward = trainer.model.forward

    def geometry_dependent_forward(ed, pd, ld, **kwargs):
        output = original_forward(ed, pd, ld, **kwargs)
        velocity = output['normalized_velocity']
        velocity['e3_ligand']['tr'] = velocity['e3_ligand']['tr'] + ld['coords'].mean(1) * .001
        velocity['poi_ligand']['tr'] = velocity['poi_ligand']['tr'] + pd['coords'].mean(1) * .001
        return output

    trainer.model.forward = geometry_dependent_forward
    trained = trainer.compute_loss(**batch, coupling_override=coupling, diagnostics=True)
    training_calls = list(trainer.model.calls)
    trainer.model.calls.clear()
    initializer = SimpleNamespace(no_torsion=False,
        initialize=lambda *args, **kwargs: (*[value.detach().clone() for value in state], {}))
    sampled = sample_single(trainer.model, flow, initializer, **batch,
                            device='cpu', return_trace=True, return_trajectory=True)
    assert sampled['valid'] and len(sampled['trajectory']) == flow.inference_steps
    assert trainer.model.calls == training_calls
    assert np.allclose(sampled['ligand_coords'], trained['predicted_ligand'][0].detach().numpy(), atol=1e-10)
    assert np.allclose(sampled['poi_coords'], trained['predicted_poi'][0].detach().numpy(), atol=1e-10)


def test_rank_features_are_read_from_actual_final_pose_and_zero_clock():
    from tricoopdiff.sampler import sample_single

    trainer, batch, coupling = _endpoint_trainer_fixture()
    trainer.diffusion.inference_steps = 1
    state = trainer.diffusion.path(batch['ligand_data'], batch['poi_data']['coords'], coupling,
                                  batch['ligand_data']['coords'].new_ones(1))[:3]
    forward = trainer.model.forward

    def read(ed, pd, ld, **kwargs):
        out = forward(ed, pd, ld, **kwargs)
        out['rank_features'] = ld['coords'].mean(1)
        out['consensus_diagnostics'] = {'rigid_output_mode': 'independent_centroids_v1'}
        return out

    trainer.model.forward = read
    initializer = SimpleNamespace(no_torsion=False,
        initialize=lambda *args, **kwargs: (*[value.clone() for value in state], {}))
    sampled = sample_single(trainer.model, trainer.diffusion, initializer, **batch,
                            device='cpu', return_rank_features=True, return_trace=True)
    assert np.allclose(sampled['rank_features'], sampled['ligand_coords'].mean(0))
    assert trainer.model.calls[-1][0] == 0.
    assert sampled['rank_feature_state'] == 'final_pose_t0_dt0_v1'
    assert sampled['trace'][-1]['consensus']['rigid_output_mode'] == 'independent_centroids_v1'


def test_endpoint_objective_cannot_silently_use_a_random_teacher_clock():
    trainer, batch, coupling = _endpoint_trainer_fixture()
    try:
        trainer.compute_loss(**batch, coupling_override=coupling, t_override=[.5])
    except ValueError as error:
        assert 't=1' in str(error)
    else:
        raise AssertionError('endpoint supervision must execute from the actual prior')


def test_endpoint_checkpoint_records_objective_weight_and_execution_length():
    from tricoopdiff.train import ENDPOINT_OBJECTIVE, validate_consensus_checkpoint

    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    runtime = payload['runtime_config']
    assert runtime['training_objective'] == ENDPOINT_OBJECTIVE
    assert runtime['endpoint_loss_weight'] == .1 and runtime['inference_steps'] == 3
    validate_consensus_checkpoint(payload)
    del runtime['inference_steps']
    try:
        validate_consensus_checkpoint(payload)
    except ValueError as error:
        assert 'rollout steps' in str(error)
    else:
        raise AssertionError('a checkpoint must not silently change the executed training horizon')


def test_endpoint_checkpoint_roundtrip_and_objective_mismatch_guard():
    from tricoopdiff.train import VELOCITY_OBJECTIVE

    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    restored, _, _ = _endpoint_trainer_fixture()
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.training_objective == trainer.training_objective
    assert all(torch.equal(a, b) for a, b in zip(
        restored.model.trifuse.parameters(), trainer.model.trifuse.parameters()))
    payload['runtime_config']['training_objective'] = VELOCITY_OBJECTIVE
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        try:
            restored.load_checkpoint('unused.pt')
        except ValueError as error:
            assert 'objective differs' in str(error)
        else:
            raise AssertionError('resume must not silently exchange endpoint and velocity supervision')


def test_real_endpoint_uses_raw_geometry_and_an_attainable_target():
    from tricoopdiff.train import REAL_ENDPOINT_OBJECTIVE
    from tricoopdiff.initialization.endpoints import reachable_ligand_target

    trainer, batch, _ = _endpoint_trainer_fixture()
    trainer.training_objective = REAL_ENDPOINT_OBJECTIVE
    trainer.model.expert.model.no_torsion = True
    with torch.no_grad():
        for parameter in trainer.model.trifuse.parameters():
            parameter.zero_()
    native = batch['ligand_data']['coords']
    initial = (native - native.mean(1, keepdim=True)) * 1.2 + native.mean(1, keepdim=True)
    target, report = reachable_ligand_target(
        batch['ligand_data'], initial, batch['poi_data']['coords'],
        batch['poi_data']['coords'], no_torsion=True)
    override = {'initial_state': {'ligand': initial[0], 'poi': batch['poi_data']['coords'][0],
                                  'body_frame': torch.eye(3, dtype=native.dtype)},
                'target_ligand': target}
    result = trainer.compute_loss(**batch, coupling_override=override, diagnostics=True)
    assert torch.allclose(result['predicted_ligand'], initial, atol=1e-10)
    assert torch.allclose(result['target_ligand'], initial, atol=1e-8)
    assert result['components']['loss_ligand_geometry'] < 1e-16
    assert (result['predicted_ligand'] - native).square().mean() > 1e-3
    assert not report['endpoint_exactly_represented']
    try:
        trainer.compute_loss(**batch, coupling_override={'angles': np.zeros(1)})
    except ValueError as error:
        assert 'crystal-projected' in str(error)
    else:
        raise AssertionError('real endpoint trainer accepted a legacy projected starting state')


def test_checkpoint_resume_restores_next_random_coupling_streams():
    import random
    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    expected = (random.random(), np.random.random(), torch.rand(4))
    restored, _, _ = _endpoint_trainer_fixture()
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])


def test_resume_rejects_incomplete_ema_before_mutating_parameters():
    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    payload['ema_trifuse_state_dict'] = dict(payload['ema_trifuse_state_dict'])
    payload['ema_trifuse_state_dict'].pop(next(iter(payload['ema_trifuse_state_dict'])))
    before = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        try:
            trainer.load_checkpoint('unused.pt')
        except ValueError as error:
            assert 'EMA state' in str(error)
        else:
            raise AssertionError('incomplete EMA weights were accepted')
    assert all(torch.equal(value, before[name]) for name, value in trainer.model.trifuse.state_dict().items())


def test_interface_checkpoint_preserves_batch_progress_and_rejects_loss_changes():
    from tricoopdiff.train import validate_consensus_checkpoint

    trainer, _, _ = _endpoint_trainer_fixture()
    trainer.samples_seen = 7
    trainer.effective_batch_size = 2
    payload = trainer.checkpoint_payload(best_val_dockq=.67)
    restored, _, _ = _endpoint_trainer_fixture()
    restored.effective_batch_size = 2
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.samples_seen == 7 and restored.resume_best_val_dockq == .67
    payload['runtime_config']['interface_loss_weight'] = .5
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        try:
            restored.load_checkpoint('unused.pt')
        except ValueError as error:
            assert 'interface_loss_weight differs' in str(error)
        else:
            raise AssertionError('resume must never silently change interface supervision')
    # Earlier objective checkpoints remain readable with their original contract.
    payload['checkpoint_version'] = 21
    for key in ('interface_loss_weight', 'interface_cutoff', 'effective_batch_size'):
        del payload['runtime_config'][key]
    validate_consensus_checkpoint(payload)


def test_resume_builder_uses_saved_model_and_loss_recipe_not_current_defaults():
    from tricoopdiff.models.consensus import production_model_config

    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    payload['model_config'] = production_model_config(32, 16, 2)
    payload['runtime_config']['steric_loss_weight'] = .05
    payload['runtime_config']['interface_loss_weight'] = .7
    args = parse_args(['--train_data', 'unused', '--val_data', 'unused',
                       '--output_dir', 'unused', '--resume', 'saved.pt'])
    with patch('train_entry.torch.load', return_value=payload), \
            patch('tricoopdiff.models.expert.build_diffdock_expert', return_value=SimpleNamespace(num_conv_layers=11)), \
            patch('tricoopdiff.models.consensus.ConsensusFuse') as fuse, \
            patch('tricoopdiff.models.consensus.ConsensusCoopDiff'), \
            patch('tricoopdiff.train.ConsensusTrainer') as builder:
        build_trainer(args, 'cpu', input_dim=16)
    assert fuse.call_args.kwargs == payload['model_config']
    assert builder.call_args.kwargs['steric_loss_weight'] == .05
    assert builder.call_args.kwargs['interface_loss_weight'] == .7
    assert args.model_profile == 'production' and args.resolved_model_config['dropout'] == 0


def test_terminal_interface_supervision_is_zero_at_native_and_penalizes_errors():
    from tricoopdiff.train import terminal_interface_losses

    e3 = torch.tensor([[[0., 0., 0.], [0., 2., 0.]]])
    native = torch.tensor([[[3., 0., 0.], [50., 0., 0.]]])
    exact, count = terminal_interface_losses(e3, native, native, 10.)
    assert count == 2 and all(value == 0 for value in exact.values())
    predicted = (native + torch.tensor([[[0., 4., 0.], [0., 0., 0.]]])).requires_grad_()
    losses, count = terminal_interface_losses(e3, native, predicted, 10.)
    assert count == 2 and all(value > 0 for value in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad[0, 0].abs().sum() > 0
    assert predicted.grad[0, 1].abs().sum() == 0, 'non-interface nodes must not dilute interface loss'
    empty, count = terminal_interface_losses(e3, native + 100, predicted, 10.)
    assert count == 0 and all(torch.isfinite(value) and value == 0 for value in empty.values())


def test_coarse_repulsion_is_target_preserving_short_range_and_differentiable():
    from tricoopdiff.train import coarse_intercomponent_repulsion

    e3 = torch.tensor([[[0., 0., 0.], [0., 0., 3.]]])
    native_poi = torch.tensor([[[5., 0., 0.]]])
    ligand = torch.tensor([[[0., 6., 0.]]])
    native = coarse_intercomponent_repulsion(e3, native_poi, ligand, native_poi, ligand)
    assert native == 0
    overlapped = torch.tensor([[[.5, 0., 0.]]], requires_grad=True)
    penalty = coarse_intercomponent_repulsion(e3, native_poi, ligand, overlapped, ligand)
    assert penalty > 0
    penalty.backward()
    assert torch.isfinite(overlapped.grad).all() and overlapped.grad[0, 0, 0] < 0
    # A native close contact is legal under this conservative coarse surrogate.
    close_native = overlapped.detach()
    assert coarse_intercomponent_repulsion(e3, close_native, ligand, close_native, ligand) == 0
    far = native_poi + 100
    assert coarse_intercomponent_repulsion(e3, native_poi, ligand, far, ligand) == 0


def test_microbatch_training_matches_mean_gradient_and_updates_ema_once():
    accumulated, batch, coupling = _endpoint_trainer_fixture()
    reference, _, _ = _endpoint_trainer_fixture()
    reference.model.load_state_dict(accumulated.model.state_dict())
    reference.optimizer.zero_grad(set_to_none=True)
    arguments = {**batch, 'coupling_override': coupling}
    # Two independently computed gradients, followed by exactly one AdamW step.
    losses = [reference.compute_loss(**arguments)['loss_total'] for _ in range(2)]
    (sum(losses) / 2).backward()
    parameters = [p for group in reference.optimizer.param_groups for p in group['params']]
    torch.nn.utils.clip_grad_norm_(parameters, 1.)
    reference.optimizer.step()
    accumulated.ema_decay = .5
    accumulated.effective_batch_size = 2
    old_ema = {key: value.clone() for key, value in accumulated.ema.items()}
    result = accumulated.train_batch([arguments, arguments])
    assert len(result) == 2 and accumulated.global_step == 1 and accumulated.samples_seen == 2
    assert all(row['optimizer_batch_size'] == 2 for row in result)
    for actual, expected in zip(accumulated.model.parameters(), reference.model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    for key, value in accumulated.ema.items():
        assert torch.allclose(value, .5 * (old_ema[key] + accumulated.model.trifuse.state_dict()[key]))
    assert [len(group) for group in iter_training_batches(range(5), 2)] == [2, 2, 1]


def test_failed_microbatch_never_updates_parameters_or_retains_gradients():
    from tricoopdiff.train import NonFiniteTrainingError

    trainer, batch, coupling = _endpoint_trainer_fixture()
    before = [p.clone() for p in trainer.model.parameters()]
    real_compute = trainer.compute_loss
    def compute(**arguments):
        if arguments.get('case') == 'bad':
            raise NonFiniteTrainingError('test bad sample')
        return real_compute(**arguments)
    trainer.compute_loss = compute
    trainer.effective_batch_size = 2
    try:
        trainer.train_batch([{**batch, 'coupling_override': coupling}, {**batch, 'case': 'bad'}])
    except NonFiniteTrainingError:
        pass
    else:
        raise AssertionError('a partial effective batch must not step the optimizer')
    assert trainer.global_step == trainer.samples_seen == 0
    assert all(torch.equal(p, old) and p.grad is None for p, old in zip(trainer.model.parameters(), before))


def test_fullatom_validation_reuses_terminal_rollout_and_restores_raw_weights():
    trainer, batch, coupling = _endpoint_trainer_fixture()
    weights = {key: value.clone() for key, value in trainer.model.trifuse.state_dict().items()}
    metric = {'dockq': .6, 'fnat': .7, 'irms': 1.2, 'lrms': 3., 'method': 'test_exact_bridge'}
    with patch('tricoopdiff.metrics.dockq.compute_fullatom_dockq', return_value=metric) as scorer:
        result = trainer.validate_step(**batch, coupling_override=coupling,
                                       dockq_context={'native_complex_pdb': 'native.pdb'})
    assert result['dockq'] == .6 and len(trainer.model.calls) == 3
    assert scorer.call_count == 1 and not trainer._ema_active
    assert all(torch.equal(value, weights[key]) for key, value in trainer.model.trifuse.state_dict().items())


def test_validation_reports_actual_dockq_and_preserves_rng():
    from tricoopdiff.train import ENDPOINT_OBJECTIVE

    class Loader:
        entries_by_case = {'sample': {'case': 'sample'}}
        def __iter__(self):
            yield {'case': 'sample'}
    class Trainer:
        training_objective = ENDPOINT_OBJECTIVE
        model = torch.nn.Linear(1, 1)
        def validate_step(self, **batch):
            assert 'dockq_context' in batch
            return {'loss_total': .1, 'dockq': .6, 'fnat': .7, 'irms': 1., 'lrms': 2.}
    before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    with patch('train_entry.validation_metric_context', return_value={'template': 'native'}):
        result = validate(Trainer(), Loader(), torch.device('cpu'))
    assert result['selection_loss'] == .1 and result['selection_dockq'] == .6
    assert result['per_case'][0]['dockq'] == .6
    assert result['dockq_protocol'] == 'training_coupling_protein_fullatom_v2'
    assert torch.equal(torch.get_rng_state(), before[0])
    assert np.array_equal(np.random.get_state()[1], before[1][1]) and random.getstate() == before[2]


def test_training_entry_keeps_dockq_best_separate_from_loss_best():
    import train_entry
    from tricoopdiff.train import ENDPOINT_OBJECTIVE

    class Trainer:
        training_objective = ENDPOINT_OBJECTIVE
        global_step = samples_seen = resume_epoch = 0
        resume_best_val_loss = float('inf')
        resume_best_val_dockq = float('-inf')
        optimizer = SimpleNamespace(param_groups=[{'lr': .001}])
        def __init__(self):
            self.saved = {}
        def set_phase(self, phase):
            assert phase == 1
        def train_batch(self, batches):
            self.global_step += 1
            self.samples_seen += len(batches)
            return [{'loss_total': .2, 'loss_ligand_geometry': .1,
                     'loss_protein_geometry': .1} for _ in batches]
        def scheduler_step(self, value):
            assert value > 0
        def save_checkpoint(self, path, **state):
            self.saved[Path(path).name] = state
    class Loader:
        def __iter__(self):
            yield from [{'case': 'sample'}] * 3
        def _epoch_entries(self):
            return [entry]
    entry = {'case': 'sample', 'protein1': 'p1', 'protein2': 'p2', 'ligand': 'lig'}
    trainer = Trainer()
    validation = [{'selection_loss': .1, 'selection_dockq': .7, 'by_source': {'sdpi': .1}},
                  {'selection_loss': .05, 'selection_dockq': .6, 'by_source': {'sdpi': .05}}]
    for row in validation:
        row['dockq_protocol'] = 'test_fullatom'
    with tempfile.TemporaryDirectory(prefix='tricoopdiff-training-test-') as output:
        args = parse_args(['--train_data', 'unused', '--val_data', 'unused',
                           '--output_dir', output, '--device', 'cpu', '--epochs', '2', '--batch_size', '2'])
        args.resolved_model_config = {'dropout': 0.}
        with patch.dict(os.environ, {}, clear=False), \
                patch('train_entry.parse_args', return_value=args), \
                patch('train_entry._load_entries', return_value=[entry]), \
                patch('train_entry._load_score_model_args', return_value=SimpleNamespace()), \
                patch('train_entry.featurize_complex', return_value=(None, None, None, 16)), \
                patch('train_entry.build_trainer', return_value=(None, None, trainer)), \
                patch('train_entry.TernaryDataLoader', return_value=Loader()), \
                patch('train_entry.validation_metric_context', return_value={}), \
                patch('train_entry.validate', side_effect=validation), redirect_stdout(io.StringIO()):
            train_entry.main()
    assert trainer.saved['best_model.pt']['epoch'] == 1
    assert trainer.saved['best_loss_model.pt']['epoch'] == 2
    assert trainer.saved['tricoopdiff_zero.pt']['best_val_dockq'] == .7
    assert trainer.global_step == 4 and trainer.samples_seen == 6


def test_legacy_architecture_and_detached_expert_are_explicitly_selectable():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--no-protein-pair-motion', '--no-expert-input-grad',
        '--no-direct-ternary-motion',
    ])
    assert not args.protein_pair_motion and not args.expert_input_grad
    assert not args.direct_ternary_motion


def test_c_builder_records_online_single_pair_endpoint_protocol():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--training-arm', 'C',
    ])
    args.training_endpoint_protocol = 'online_bitseed_training_c1_m1_v1'
    expert = SimpleNamespace(num_conv_layers=11)
    with patch('tricoopdiff.models.expert.build_diffdock_expert', return_value=expert), \
            patch('tricoopdiff.models.consensus.ConsensusFuse'), \
            patch('tricoopdiff.models.consensus.ConsensusCoopDiff'), \
            patch('tricoopdiff.train.ConsensusTrainer') as trainer:
        build_trainer(args, 'cpu', input_dim=1281)
        assert trainer.call_args.kwargs['training_endpoint_protocol'] == (
            'online_bitseed_raw_resampled_v2'
        )


def test_legacy_clamp_mode_is_explicitly_selectable():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--activation-clamp-mode', 'componentwise',
    ])
    assert args.activation_clamp_mode == 'componentwise'


def test_mixed_arms_share_exact_source_and_endpoint_sequence():
    class Store:
        def coupling(self, case, occurrence=0):
            return ({'case_marker': case, 'occurrence': occurrence},
                    {'actual_source': 'bitseed'})

    batch = {'case': 'case-a'}
    for step in range(6):
        arm_b = prepare_source_batch(batch, step, 'B', Store())
        arm_c = prepare_source_batch(batch, step, 'C', Store())
        assert arm_b == arm_c
        assert arm_b['source'] == step % 2
        if step % 2:
            assert arm_b['coupling_override']['occurrence'] == step // 2


def test_a1_never_consumes_cached_endpoint():
    batch = prepare_source_batch({'case': 'case-a'}, 3, 'A1', None)
    assert batch == {'case': 'case-a', 'source': 0}


def test_sampler_conditions_on_actual_fallback_source():
    assert actual_candidate_source('sdpi', {}) == 'sdpi'
    assert actual_candidate_source('expert_anchor', {'fallback': None}) == 'bitseed'
    assert actual_candidate_source(
        'expert_anchor', {'fallback': 'broad_sdpi_clash_filter'},
    ) == 'sdpi'
    assert actual_candidate_source(
        'fixed_candidate', {'candidate_source': 'bitseed'},
    ) == 'bitseed'


def test_mixed_training_relabels_ood_expert_fallback_as_sdpi():
    class Store:
        def coupling(self, case, occurrence=0):
            return {'cached': True}, {'actual_source': 'sdpi'}

    batch = prepare_source_batch({'case': 'ood'}, 1, 'C', Store())
    assert batch['source'] == 0
    assert batch['coupling_override'] == {'cached': True}


def test_training_entry_has_no_lora_configuration():
    try:
        parse_args([
            '--train_data', 'unused.json', '--val_data', 'unused.json',
            '--output_dir', 'unused', '--lora_rank', '4',
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('the production entry must not expose LoRA')


def test_gradient_clipping_does_not_overflow_on_finite_float32_gradients():
    from tricoopdiff.train import clip_grad_norm_finite, NonFiniteTrainingError
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([1e30, -1e30, 1e30, -1e30])
    norm = clip_grad_norm_finite([parameter])
    assert torch.isfinite(norm) and norm > 1e30
    assert torch.allclose(parameter.grad, torch.tensor([.5, -.5, .5, -.5]))
    parameter.grad[0] = float('inf')
    try:
        clip_grad_norm_finite([parameter])
    except NonFiniteTrainingError:
        pass
    else:
        raise AssertionError('genuinely non-finite gradients must still fail')


if __name__ == "__main__":
    torch.set_num_threads(2)
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check()
            print(name, 'passed')
    print("Training-contract tests passed")
