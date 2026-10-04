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


def test_training_index_rejects_duplicate_case_identities():
    import json
    from train_entry import _load_index

    entry = {'case': 'same', 'protein1': 'a.pdb', 'protein2': 'b.pdb', 'ligand': 'c.sdf'}
    with tempfile.TemporaryDirectory() as directory:
        for entries in ([entry, entry], [entry, {**entry, 'protein1': 'different.pdb'}]):
            path = Path(directory) / 'index.json'
            path.write_text(json.dumps({'entries': entries, 'clusters': [{'items': ['same']}]}))
            try:
                _load_index(path)
            except ValueError as error:
                assert 'duplicate case' in str(error)
            else:
                raise AssertionError('duplicate case identities silently changed the sampled dataset')


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
    expert = SimpleNamespace(num_conv_layers=11, node_scalar_dim=58, ns=48, node_vector_dim=20)
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
            model_profile='ternary_endpoint_v7', expert_scalar_dim=58, expert_film_dim=48, expert_vector_dim=20,
        )
        assert model.call_args.kwargs['expert_input_grad'] is True
        assert result is model.return_value
        assert trained is trainer.return_value
        assert trainer.call_args.kwargs['activation_clamp_mode'] == 'irrep_norm'
        assert trainer.call_args.kwargs['training_endpoint_protocol'] == 'none'
        assert trainer.call_args.kwargs['steric_loss_weight'] == .05
        assert args.resolved_model_config['dropout'] == 0


def _endpoint_trainer_fixture(profile=None, inference_steps=3):
    from tricoopdiff.models.consensus import DIRECT_ARCHITECTURE, production_model_config
    from tricoopdiff.train import ConsensusTrainer
    from test_sdpi_flow import fixture

    flow, ligand, e3, poi, coupling = fixture()
    flow.inference_steps = inference_steps

    class Model(torch.nn.Module):
        architecture = DIRECT_ARCHITECTURE
        expert_input_grad = True

        def __init__(self):
            super().__init__()
            self.trifuse = torch.nn.Module()
            self.trifuse.direct_ternary_motion = True
            self.trifuse.config = {'direct_ternary_motion': True, 'node_dim': 4}
            if profile is not None:
                self.trifuse.config = production_model_config(4, 4, 1, model_profile=profile)
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
                **({'torsion_update_mode': 'continuous_horizontal_v1'}
                   if profile == 'ternary_endpoint_v4' else {}),
                **{b: {'tr': self.trifuse.motion[i:i+1, :3],
                       'rot': self.trifuse.motion[i:i+1, 3:], 'tor': self.trifuse.torsion}
                   for i, b in enumerate(('e3_ligand', 'poi_ligand'))},
                'torsion': self.trifuse.torsion}}

    model = Model()
    trainer = ConsensusTrainer(model, flow, device='cpu', ema_decay=.999 if profile else 0.)
    batch = {'e3_data': {'coords': e3}, 'poi_data': {'coords': poi}, 'ligand_data': ligand}
    if profile == 'ternary_endpoint_v7':
        batch['fullatom_targets'] = {'e3': e3[0].clone(), 'poi': poi[0].clone()}
    initial = flow.path(ligand, poi, coupling, ligand['coords'].new_ones(1))[:3]
    override = {'initial_state': dict(zip(('ligand', 'poi', 'body_frame'), initial)),
                'target_ligand': ligand['coords']}
    return trainer, batch, override


def test_direct_trainer_supervises_executed_endpoint_not_teacher_velocities():
    from tricoopdiff.train import REAL_ENDPOINT_OBJECTIVE

    trainer, batch, coupling = _endpoint_trainer_fixture()
    assert trainer.training_objective == REAL_ENDPOINT_OBJECTIVE
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
    state = [coupling['initial_state'][key] for key in ('ligand', 'poi', 'body_frame')]
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


def test_sampling_rejects_direct_motion_with_missing_or_legacy_mode():
    from tricoopdiff.sampler import sample_single

    for mode in (None, 'dual_interfaces_v1'):
        trainer, batch, coupling = _endpoint_trainer_fixture()
        forward = trainer.model.forward

        def read(*args, **kwargs):
            output = forward(*args, **kwargs)
            normalized = output['normalized_velocity']
            if mode is None:
                normalized.pop('rigid_output_mode')
            else:
                normalized['rigid_output_mode'] = mode
            return output

        trainer.model.forward = read
        state = coupling['initial_state']
        initializer = SimpleNamespace(no_torsion=False, initialize=lambda *args, **kwargs:
            (*[state[key].clone() for key in ('ligand', 'poi', 'body_frame')], {}))
        for execute in (
                lambda: trainer.compute_loss(**batch, coupling_override=coupling),
                lambda: sample_single(trainer.model, trainer.diffusion, initializer,
                                      **batch, device='cpu')):
            try:
                execute()
            except ValueError as error:
                assert 'independent centroid' in str(error)
            else:
                raise AssertionError('training/sampling silently decoded direct motion as dual interfaces')


def test_sampling_restores_probe_flag_when_forward_fails():
    from tricoopdiff.sampler import sample_single

    trainer, batch, coupling = _endpoint_trainer_fixture()
    state = coupling['initial_state']
    initializer = SimpleNamespace(no_torsion=False, initialize=lambda *args, **kwargs:
        (*[state[key].clone() for key in ('ligand', 'poi', 'body_frame')], {}))
    trainer.model._debug_step0_probe = False

    def fail(*args, **kwargs):
        assert trainer.model._debug_step0_probe
        raise RuntimeError('forward failed')

    trainer.model.forward = fail
    try:
        sample_single(trainer.model, trainer.diffusion, initializer, **batch,
                      device='cpu', return_trace=True)
    except RuntimeError as error:
        assert str(error) == 'forward failed'
    else:
        raise AssertionError('sampling swallowed a model failure')
    assert not trainer.model._debug_step0_probe, 'failed trace leaked into subsequent model calls'


def test_sampling_rejects_initialization_atom_mismatch_before_model():
    from tricoopdiff.sampler import sample_single

    trainer, batch, coupling = _endpoint_trainer_fixture()
    state = coupling['initial_state']
    initializer = SimpleNamespace(no_torsion=False, initialize=lambda *args, **kwargs:
        (state['ligand'][:, :-1], state['poi'], state['body_frame'], {}))
    try:
        sample_single(trainer.model, trainer.diffusion, initializer, **batch, device='cpu')
    except ValueError as error:
        assert 'atom counts' in str(error)
    else:
        raise AssertionError('sampling accepted coordinates inconsistent with graph atom counts')
    assert trainer.model.calls == [], 'invalid geometry reached the model'


def test_rank_features_are_read_from_actual_final_pose_and_zero_clock():
    from tricoopdiff.sampler import sample_single

    trainer, batch, coupling = _endpoint_trainer_fixture()
    trainer.diffusion.inference_steps = 1
    state = [coupling['initial_state'][key] for key in ('ligand', 'poi', 'body_frame')]
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
    from tricoopdiff.train import REAL_ENDPOINT_OBJECTIVE, validate_consensus_checkpoint

    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    runtime = payload['runtime_config']
    assert runtime['training_objective'] == REAL_ENDPOINT_OBJECTIVE
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


def test_resume_rejects_invalid_weights_before_mutating_parameters():
    trainer, _, _ = _endpoint_trainer_fixture()
    before = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    for key, corruption in (('ema_trifuse_state_dict', 'missing'),
                            ('trifuse_state_dict', 'missing'), ('trifuse_state_dict', 'nan')):
        payload = trainer.checkpoint_payload()
        state = {name: value.clone() + 1 for name, value in payload[key].items()}
        if corruption == 'missing':
            state.pop(next(reversed(state)))
        else:
            state[next(iter(state))].fill_(float('nan'))
        payload[key] = state
        with patch('tricoopdiff.train.torch.load', return_value=payload):
            try:
                trainer.load_checkpoint('unused.pt')
            except ValueError as error:
                assert 'state' in str(error)
            else:
                raise AssertionError('invalid checkpoint weights were accepted')
        assert all(torch.equal(value, before[name])
                   for name, value in trainer.model.trifuse.state_dict().items())


def test_resume_rejects_invalid_optimization_without_partial_mutation():
    import copy
    from tricoopdiff.runtime import capture_rng_state

    trainer, batch, coupling = _endpoint_trainer_fixture()
    trainer.train_step(**batch, coupling_override=coupling, source=1)
    checkpoint = copy.deepcopy(trainer.checkpoint_payload())
    optimizer, scheduler = trainer.optimizer, trainer.scheduler
    before = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    ema = {name: value.clone() for name, value in trainer.ema.items()}
    rng = capture_rng_state()
    for corruption in ('group', 'rng', 'nan_moment', 'shape', 'negative_variance',
                       'scheduler', 'counter', 'dockq', 'fractional_step', 'future_step', 'sample_counter'):
        payload = copy.deepcopy(checkpoint)
        payload['trifuse_state_dict'] = {name: value + 1 for name, value in before.items()}
        moments = next(iter(payload['optimizer_state_dict']['state'].values()))
        if corruption == 'group':
            payload['optimizer_state_dict']['param_groups'][0]['params'].pop()
        elif corruption == 'rng':
            payload['rng_state']['torch_cpu'] = torch.zeros(2, dtype=torch.uint8)
        elif corruption == 'nan_moment':
            moments['exp_avg'].fill_(float('nan'))
        elif corruption == 'shape':
            moments['exp_avg'] = torch.zeros(1)
        elif corruption == 'negative_variance':
            moments['exp_avg_sq'].fill_(-1.)
        elif corruption == 'scheduler':
            payload['scheduler_state_dict']['best'] = float('nan')
        elif corruption == 'counter':
            payload['global_step'] = 1.5
        elif corruption == 'fractional_step':
            moments['step'].fill_(.5)
        elif corruption == 'future_step':
            moments['step'].fill_(payload['global_step'] + 1)
        elif corruption == 'sample_counter':
            payload['training_state']['samples_seen'] = payload['global_step'] - 1
        else:
            payload['training_state']['best_val_dockq'] = float('nan')
        with patch('tricoopdiff.train.torch.load', return_value=payload):
            try:
                trainer.load_checkpoint('unused.pt')
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError(f'invalid {corruption} checkpoint was accepted')
        assert trainer.optimizer is optimizer and trainer.scheduler is scheduler
        assert trainer.global_step == checkpoint['global_step']
        assert all(torch.equal(value, before[name])
                   for name, value in trainer.model.trifuse.state_dict().items())
        assert all(torch.equal(value, ema[name]) for name, value in trainer.ema.items())
        assert torch.equal(torch.get_rng_state(), rng['torch_cpu']), 'rejected resume changed RNG'


def test_resumed_optimizer_matches_uninterrupted_next_update():
    import copy

    trainer, batch, coupling = _endpoint_trainer_fixture()
    trainer.train_step(**batch, coupling_override=coupling, source=1)
    payload = copy.deepcopy(trainer.checkpoint_payload())
    restored, _, _ = _endpoint_trainer_fixture()
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.scheduler.optimizer is restored.optimizer
    trainer.train_step(**batch, coupling_override=coupling, source=1)
    restored.train_step(**batch, coupling_override=coupling, source=1)
    assert all(torch.equal(value, restored.model.trifuse.state_dict()[name])
               for name, value in trainer.model.trifuse.state_dict().items())


def test_resume_rejects_weights_that_overflow_the_active_dtype():
    import copy

    trainer, _, _ = _endpoint_trainer_fixture()
    trainer.model.trifuse.float()
    trainer.ema = {name: value.float() for name, value in trainer.ema.items()}
    before = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    checkpoint = copy.deepcopy(trainer.checkpoint_payload())
    for state_name in ('trifuse_state_dict', 'ema_trifuse_state_dict'):
        payload = copy.deepcopy(checkpoint)
        name = next(iter(payload[state_name]))
        payload[state_name][name] = torch.full_like(payload[state_name][name], 1e300,
                                                  dtype=torch.float64)
        assert torch.isfinite(payload[state_name][name]).all()
        with patch('tricoopdiff.train.torch.load', return_value=payload):
            try:
                trainer.load_checkpoint('unused.pt')
            except ValueError as error:
                assert 'overflows' in str(error)
            else:
                raise AssertionError('finite source weights overflowed the live model during resume')
        assert all(torch.equal(value, before[name])
                   for name, value in trainer.model.trifuse.state_dict().items())
        assert all(torch.equal(value, checkpoint['ema_trifuse_state_dict'][name])
                   for name, value in trainer.ema.items())


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
    old = trainer.checkpoint_payload(best_val_loss=.01, best_val_dockq=.99)
    old['runtime_config']['validation_dockq_protocol'] = 'outdated'
    old['scheduler_state_dict']['best'] = .01
    old['scheduler_state_dict']['num_bad_epochs'] = 9
    restored.scheduler.step(.001)
    assert restored.scheduler.best == .001 and restored.scheduler.last_epoch == 1
    with patch('tricoopdiff.train.torch.load', return_value=old):
        restored.load_checkpoint('unused.pt')
    assert restored.resume_best_val_loss == float('inf')
    assert restored.resume_best_val_dockq == float('-inf')
    assert restored.scheduler.best == float('inf') and restored.scheduler.num_bad_epochs == 0
    assert restored.scheduler.last_epoch == 0
    assert restored.scheduler._last_lr == [group['lr'] for group in restored.optimizer.param_groups]
    # Earlier objective checkpoints remain readable with their original contract.
    payload['checkpoint_version'] = 21
    payload['runtime_config']['training_objective'] = 'executed_endpoint_geometry_v1'
    for key in ('interface_loss_weight', 'interface_cutoff', 'effective_batch_size'):
        del payload['runtime_config'][key]
    validate_consensus_checkpoint(payload)


def test_resume_builder_uses_saved_model_and_loss_recipe_not_current_defaults():
    from tricoopdiff.models.consensus import production_model_config

    trainer, _, _ = _endpoint_trainer_fixture()
    payload = trainer.checkpoint_payload()
    payload['model_config'] = production_model_config(32, 16, 2, model_profile='ternary_endpoint_v1')
    payload['runtime_config']['steric_loss_weight'] = .05
    payload['runtime_config']['interface_loss_weight'] = .7
    args = parse_args(['--train_data', 'unused', '--val_data', 'unused',
                       '--output_dir', 'unused', '--resume', 'saved.pt'])
    with patch('train_entry.torch.load', return_value=payload), \
            patch('tricoopdiff.models.expert.build_diffdock_expert', return_value=SimpleNamespace(num_conv_layers=11, node_scalar_dim=58, ns=48, node_vector_dim=20)), \
            patch('tricoopdiff.models.consensus.ConsensusFuse') as fuse, \
            patch('tricoopdiff.models.consensus.ConsensusCoopDiff'), \
            patch('tricoopdiff.train.ConsensusTrainer') as builder:
        build_trainer(args, 'cpu', input_dim=16)
    assert fuse.call_args.kwargs == payload['model_config']
    assert builder.call_args.kwargs['steric_loss_weight'] == .05
    assert builder.call_args.kwargs['interface_loss_weight'] == .7
    assert args.resolved_model_config['dropout'] == 0


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


def test_interface_and_clash_supervision_are_translation_invariant():
    from tricoopdiff.train import terminal_interface_losses, coarse_intercomponent_repulsion

    line = torch.stack((torch.arange(32).float() / 8, torch.zeros(32), torch.zeros(32)), -1)[None]
    poi, ligand = line + torch.tensor([0., .125, 0.]), line + torch.tensor([0., 0., .125])
    predicted = poi + torch.tensor([0., -.0625, 0.])
    results = []
    for shift in (0., 1024.):
        interface, count = terminal_interface_losses(line + shift, poi + shift, predicted + shift, .2)
        clash = coarse_intercomponent_repulsion(line + shift, poi + shift, ligand + shift,
                                                predicted + shift, ligand + shift, cutoff=.2)
        results.append((interface, count, clash))
    assert results[0][1] == results[1][1], 'translation changed the native interface labels'
    assert torch.equal(results[0][2], results[1][2]), 'translation changed clash supervision'
    for key in results[0][0]:
        assert torch.equal(results[0][0][key], results[1][0][key]), key


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
    trainer.model.train()
    trainer.model.fixed_eval_child = torch.nn.Dropout().eval()
    weights = {key: value.clone() for key, value in trainer.model.trifuse.state_dict().items()}
    metric = {'dockq': .6, 'fnat': .7, 'irms': 1.2, 'lrms': 3., 'method': 'test_exact_bridge'}
    with patch('tricoopdiff.metrics.dockq.compute_fullatom_dockq', return_value=metric) as scorer:
        result = trainer.validate_step(**batch, coupling_override=coupling,
                                       dockq_context={'native_complex_pdb': 'native.pdb'})
    assert result['dockq'] == .6 and len(trainer.model.calls) == 3
    assert scorer.call_count == 1 and not trainer._ema_active
    assert trainer.model.training and not trainer.model.fixed_eval_child.training
    assert all(torch.equal(value, weights[key]) for key, value in trainer.model.trifuse.state_dict().items())


def test_direct_validation_restores_modes_rng_and_grad_context_on_errors():
    from tricoopdiff.runtime import capture_rng_state

    trainer, batch, coupling = _endpoint_trainer_fixture()
    compute = trainer.compute_loss
    for failure in (False, True):
        def consuming(**kwargs):
            assert not trainer.model.training and not torch.is_grad_enabled()
            random.random(); np.random.random(); torch.rand(3)
            if failure:
                raise RuntimeError('intentional validation failure')
            return compute(**kwargs)
        trainer.compute_loss = consuming
        for validate_method in (trainer.validate_step,
                                lambda **kwargs: TriCoopDiffTrainer.validate_step(trainer, **kwargs)):
            trainer.model.train()
            trainer.model.fixed_eval_child = torch.nn.Dropout().eval()
            before = capture_rng_state()
            try:
                validate_method(**batch, coupling_override=coupling)
            except RuntimeError as error:
                assert failure and 'intentional' in str(error)
            else:
                assert not failure
            assert trainer.model.training and not trainer.model.fixed_eval_child.training
            assert torch.is_grad_enabled(), 'validation left the caller in no_grad'
            assert torch.equal(torch.get_rng_state(), before['torch_cpu'])
            assert random.getstate() == before['python']
            assert np.array_equal(np.random.get_state()[1], before['numpy'][1])
            assert not getattr(trainer, '_ema_active', False)


def test_ema_context_rejects_optimizer_and_checkpoint_mutations():
    trainer, batch, coupling = _endpoint_trainer_fixture()
    checkpoint = trainer.checkpoint_payload()
    raw = {key: value.clone() for key, value in trainer.model.trifuse.state_dict().items()}
    with trainer.ema_parameters():
        with trainer.ema_parameters():
            operations = (lambda: trainer.train_step(**batch, coupling_override=coupling),
                          lambda: trainer.checkpoint_payload(),
                          lambda: trainer._restore_optimization(checkpoint))
            for operation in operations:
                try:
                    operation()
                except RuntimeError as error:
                    assert 'EMA weights' in str(error)
                else:
                    raise AssertionError('temporary EMA weights contaminated training state')
        assert trainer._ema_active, 'nested EMA context restored too early'
    assert trainer.global_step == trainer.samples_seen == 0 and not trainer.optimizer.state
    assert all(torch.equal(value, raw[key]) for key, value in trainer.model.trifuse.state_dict().items())


def test_failed_ema_entry_restores_all_raw_weights():
    trainer, _, _ = _endpoint_trainer_fixture()
    raw = {key: value.clone() for key, value in trainer.model.trifuse.state_dict().items()}
    trainer.ema = {key: value + 1 for key, value in trainer.ema.items()}
    trainer.ema.pop(next(reversed(trainer.ema)))
    try:
        with trainer.ema_parameters():
            raise AssertionError('invalid EMA state was accepted')
    except RuntimeError:
        pass
    assert not trainer._ema_active
    assert all(torch.equal(value, raw[key]) for key, value in trainer.model.trifuse.state_dict().items())


def test_validation_reports_actual_dockq_and_preserves_rng():
    from tricoopdiff.train import REAL_ENDPOINT_OBJECTIVE

    class Loader:
        entries_by_case = {'sample': {'case': 'sample'}}
        def __iter__(self):
            yield {'case': 'sample'}
    class Trainer:
        training_objective = REAL_ENDPOINT_OBJECTIVE
        model = torch.nn.Linear(1, 1)
        def validate_step(self, **batch):
            assert 'dockq_context' in batch
            return {'loss_total': .1, 'dockq': .6, 'fnat': .7, 'irms': 1., 'lrms': 2.}
    Trainer.model.fixed_eval_child = torch.nn.Dropout().eval()
    before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    with patch('train_entry.validation_metric_context', return_value={'template': 'native'}):
        result = validate(Trainer(), Loader(), torch.device('cpu'))
    assert result['selection_loss'] == .1 and result['selection_dockq'] == .6
    assert result['selection_loss_component'] == 'loss_endpoint_main'
    assert result['per_case'][0]['loss_endpoint_main'] == .1
    assert result['per_case'][0]['dockq'] == .6
    assert result['dockq_protocol'] == 'case_fixed_real_initialization_protein_fullatom_v4'
    assert Trainer.model.training and not Trainer.model.fixed_eval_child.training
    assert torch.equal(torch.get_rng_state(), before[0])
    assert np.array_equal(np.random.get_state()[1], before[1][1]) and random.getstate() == before[2]


    class SeedTrainer(Trainer):
        def validate_step(self, **batch):
            result = super().validate_step(**batch)
            result['loss_total'] = float(torch.randint(0, 2**31 - 1, (1,)).item())
            return result
    class OrderedLoader(Loader):
        entries_by_case = {case: {'case': case} for case in ('a', 'b')}
        def __init__(self, order):
            self.order = order
        def __iter__(self):
            yield from ({'case': case} for case in self.order)
    trainer = SeedTrainer()
    trainer.model.eval()
    with patch('train_entry.validation_metric_context', return_value={}):
        forward = validate(trainer, OrderedLoader(['a', 'b']), torch.device('cpu'))
        reverse = validate(trainer, OrderedLoader(['b', 'a']), torch.device('cpu'))
        alone = validate(trainer, OrderedLoader(['a']), torch.device('cpu'))
    first = {row['case']: row['loss'] for row in forward['per_case']}
    assert first == {row['case']: row['loss'] for row in reverse['per_case']}
    assert first['a'] == alone['per_case'][0]['loss']
    assert not trainer.model.training, 'validation must preserve the caller model mode'


def test_sparse_auxiliary_metrics_do_not_impute_zeros():
    from train_entry import summarize_metrics
    means, counts = summarize_metrics([
        {'loss_total': 2., 'loss_near_target': 4.},
        {'loss_total': 4., 'loss_state_recovery': 8.}])
    assert means == {'loss_total': 3., 'loss_near_target': 4., 'loss_state_recovery': 8.}
    assert counts == {'loss_total': 2, 'loss_near_target': 1, 'loss_state_recovery': 1}


def test_fullatom_terminal_loss_tracks_actual_rigid_motion():
    from tricoopdiff.train import fullatom_interface_target, terminal_fullatom_loss
    native = torch.tensor([[[2., 0., 0.], [3., 1., 0.]]], dtype=torch.double)
    target = fullatom_interface_target({'e3': torch.zeros(1, 3), 'poi': native[0]}, native)
    body = torch.eye(3, dtype=torch.double).unsqueeze(0)
    assert terminal_fullatom_loss(target, native, native, body).item() == 0.
    moved = (native + 1.).requires_grad_()
    loss = terminal_fullatom_loss(target, native, moved, body)
    gradient, = torch.autograd.grad(loss, moved)
    assert loss > 0 and torch.isfinite(gradient).all() and gradient.norm() > 0


def test_packed_budget_preserves_order_and_oversized_singletons():
    from tricoopdiff.models.consensus import packed_groups
    def item(size):
        return {key: {'coords': torch.zeros(size, 3)}
                for key in ('e3_data', 'poi_data', 'ligand_data')}
    items = [item(2), item(2), item(20), item(2)]
    groups = list(packed_groups(items, {'nodes': 16, 'pairs': 100000}))
    assert [len(group) for group in groups] == [2, 1, 1]
    assert all(a is b for a, b in zip(items, sum(groups, [])))


def test_training_entry_keeps_dockq_best_separate_from_loss_best():
    import train_entry
    from tricoopdiff.train import REAL_ENDPOINT_OBJECTIVE

    class Trainer:
        training_objective = REAL_ENDPOINT_OBJECTIVE
        fullatom_interface_weight = 0.
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


def test_production_entry_rejects_retired_architecture_switches():
    for option in ('--no-direct-ternary-motion', '--no-expert-input-grad',
                   '--protein-pair-motion', '--bitseed-endpoint-cache', '--ns'):
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            try:
                parse_args(['--train_data', 'unused', '--val_data', 'unused',
                            '--output_dir', 'unused', option])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f'production retained obsolete option {option}')


def test_c_builder_records_online_single_pair_endpoint_protocol():
    args = parse_args([
        '--train_data', 'unused.json', '--val_data', 'unused.json',
        '--output_dir', 'unused', '--training-arm', 'C',
    ])
    args.training_endpoint_protocol = 'online_bitseed_training_c1_m1_v1'
    expert = SimpleNamespace(num_conv_layers=11, node_scalar_dim=58, ns=48, node_vector_dim=20)
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


def test_base_velocity_trainer_initializes_residual_weight():
    from tricoopdiff.train import TriCoopDiffTrainer

    endpoint, batch, _ = _endpoint_trainer_fixture()
    trainer = TriCoopDiffTrainer(endpoint.model, endpoint.diffusion, device='cpu')
    forward = endpoint.model.forward
    def with_residual(*args, **kwargs):
        return {**forward(*args, **kwargs), 'residual_penalty': torch.tensor(1.)}
    with patch.object(endpoint.model, 'forward', side_effect=with_residual):
        result = trainer.compute_loss(**batch, t_override=torch.tensor([.5]))
    assert trainer.residual_weight == 0 and result['loss_residual'] == 1


def test_v2_training_adds_reachable_short_starts_and_updates_ema_once():
    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v2', inference_steps=20)
    trainer.effective_batch_size = 2
    old_ema = {name: value.clone() for name, value in trainer.ema.items()}
    arguments = {**batch, 'source': 1, 'coupling_override': coupling}
    with patch.object(trainer, '_endpoint_rollout_loss', wraps=trainer._endpoint_rollout_loss) as rollout:
        rows = trainer.train_batch([arguments, arguments])
    assert len(rows) == 2 and all('loss_near_target' in row for row in rows)
    assert trainer.global_step == 1 and trainer.samples_seen == 2
    assert len(trainer.model.calls) == 48, 'each microbatch must execute 20 full + 4 short steps'
    assert all(source == 1 for _, source in trainer.model.calls), 'short starts must inherit source'
    assert np.allclose([clock for clock, _ in trainer.model.calls[20:24]], [.2, .15, .1, .05])
    zero, perturbed = rollout.call_args_list[1], rollout.call_args_list[3]
    assert torch.equal(zero.args[3], coupling['target_ligand'])
    assert torch.equal(zero.args[4], batch['poi_data']['coords'])
    assert not torch.equal(perturbed.args[3], coupling['target_ligand'])
    assert not torch.equal(perturbed.args[4], batch['poi_data']['coords'])
    decay = 2 / 11
    assert trainer.effective_ema_decay == decay
    for name, value in trainer.ema.items():
        assert torch.allclose(value, decay * old_ema[name]
                              + (1 - decay) * trainer.model.trifuse.state_dict()[name])
    assert trainer.model.trifuse.motion.grad is not None
    assert torch.isfinite(trainer.model.trifuse.motion.grad).all()


def test_v2_exact_target_has_zero_auxiliary_loss_and_finite_gradient():
    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v2')
    with torch.no_grad():
        trainer.model.trifuse.motion.zero_()
        trainer.model.trifuse.torsion.zero_()
    output = trainer.compute_loss(**batch, coupling_override=coupling,
                                  near_target_zero=True, diagnostics=True)
    assert output['components']['loss_near_target'] < 1e-25
    output['loss_total'].backward()
    for parameter in (trainer.model.trifuse.motion, trainer.model.trifuse.torsion):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_v2_validation_uses_full_starts_and_can_explicitly_read_raw_weights():
    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v2')
    trainer.ema['motion'] = trainer.ema['motion'] + .1
    raw = trainer.validate_step(**batch, coupling_override=coupling, weights='raw')
    ema = trainer.validate_step(**batch, coupling_override=coupling)
    assert 'loss_near_target' not in raw and 'loss_near_target' not in ema
    assert len(trainer.model.calls) == 6
    assert raw['loss_total'] != ema['loss_total']
    assert raw['loss_endpoint_main'] == raw['loss_total']
    for auxiliary in ({'near_target_zero': True}, {'near_target_start_t': .2}):
        try:
            trainer.validate_step(**batch, coupling_override=coupling, **auxiliary)
        except ValueError as error:
            assert 'main endpoint only' in str(error)
        else:
            raise AssertionError('validation selection loss accidentally included the training auxiliary')
    with trainer.ema_parameters():
        try:
            trainer.validate_step(**batch, coupling_override=coupling, weights='raw')
        except RuntimeError as error:
            assert 'raw validation' in str(error)
        else:
            raise AssertionError('temporary EMA weights were misreported as raw')


def test_v2_checkpoint_records_recipe_and_resumes_warmup_and_entry_parity():
    import copy

    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v2')
    trainer.train_step(**batch, coupling_override=coupling)
    payload = copy.deepcopy(trainer.checkpoint_payload())
    assert payload['checkpoint_version'] == 31
    payload['checkpoint_version'] = 25  # Historical v2 weights retain their old recipe.
    assert payload['runtime_config']['ema_update_contract'] == 'optimizer_step_warmup_v1'
    assert payload['runtime_config']['near_target_supervision']['weight'] == .25
    assert payload['training_state']['ema_updates'] == payload['global_step'] == 1
    restored, _, _ = _endpoint_trainer_fixture('ternary_endpoint_v2')
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.effective_ema_decay == trainer.effective_ema_decay
    for corrupted in ('ema_updates', 'near_target_supervision'):
        changed = copy.deepcopy(payload)
        if corrupted == 'ema_updates':
            changed['training_state']['ema_updates'] += 1
        else:
            changed['runtime_config']['near_target_supervision']['weight'] = 1.
        with patch('tricoopdiff.train.torch.load', return_value=changed):
            try:
                restored.load_checkpoint('unused.pt')
            except ValueError:
                pass
            else:
                raise AssertionError(f'new recipe accepted corrupt {corrupted}')
    trainer.train_step(**batch, coupling_override=coupling)
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    restored.train_step(**batch, coupling_override=coupling)
    assert all(torch.equal(value, restored.model.trifuse.state_dict()[name])
               for name, value in trainer.model.trifuse.state_dict().items())
    assert all(torch.equal(value, restored.ema[name]) for name, value in trainer.ema.items())


def test_v1_version24_resume_preserves_fixed_ema_and_no_auxiliary():
    import copy

    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v1')
    payload = copy.deepcopy(trainer.checkpoint_payload())
    payload['checkpoint_version'] = 24
    for key in ('training_recipe', 'ema_update_contract', 'near_target_supervision', 'validation_weights'):
        payload['runtime_config'].pop(key)
    payload['training_state'].pop('ema_updates')
    restored, _, _ = _endpoint_trainer_fixture('ternary_endpoint_v1')
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.effective_ema_decay == .999
    assert restored.near_target_weight == 0
    result = restored.train_step(**batch, coupling_override=coupling)
    assert 'loss_near_target' not in result and len(restored.model.calls) == 3


def test_v4_near_entry_cycle_is_orthogonal_to_source_and_batch_boundaries():
    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v4')
    trainer.effective_batch_size = 8
    seen = []
    def record(**arguments):
        seen.append((arguments['source'], arguments['near_target_zero'], arguments['near_target_start_t']))
        return {'loss_total': trainer.model.trifuse.motion.square().sum()}
    entries = [{**batch, 'source': index % 2, 'coupling_override': coupling} for index in range(8)]
    with patch.object(trainer, 'compute_loss', side_effect=record):
        trainer.train_batch(entries[:3])
        trainer.train_batch(entries[3:])
    assert len(set(seen)) == 8, 'source, zero/perturbed entry and clock must cover the full factorial'
    assert seen == [(index % 2, (index // 2) % 2 == 0, (1., .2)[(index // 4) % 2])
                    for index in range(8)]
    assert trainer.samples_seen == 8 and trainer.global_step == 2


def test_v4_near_supervision_executes_both_inference_and_short_clocks():
    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v4', inference_steps=20)
    for start in (1., .2):
        trainer.model.calls.clear()
        result = trainer.compute_loss(**batch, coupling_override=coupling,
                                      near_target_zero=True, near_target_start_t=start)
        assert len(trainer.model.calls) == 20 + (20 if start == 1. else 4)
        assert trainer.model.calls[20][0] == start
        assert 'loss_near_target' in result and torch.isfinite(result['loss_total'])
        assert abs(result['loss_total'].item() - result['loss_endpoint_main']
                   - .25 * result['loss_near_target']) < 1e-10
    for invalid in (.5, float('nan'), 0.):
        try:
            trainer.compute_loss(**batch, coupling_override=coupling,
                                 near_target_zero=True, near_target_start_t=invalid)
        except ValueError as error:
            assert 'recorded training clocks' in str(error)
        else:
            raise AssertionError('unrecorded auxiliary clock was accepted')


def test_production_entry_rejects_old_v2_training_recipe_without_reinterpretation():
    from tricoopdiff.train import validate_consensus_checkpoint

    trainer, _, _ = _endpoint_trainer_fixture('ternary_endpoint_v2')
    payload = trainer.checkpoint_payload()
    payload['checkpoint_version'] = 25
    validate_consensus_checkpoint(payload)  # Historical inference remains legal.
    args = parse_args(['--train_data', 'unused', '--val_data', 'unused',
                       '--output_dir', 'unused', '--resume', 'saved.pt'])
    with patch('train_entry.torch.load', return_value=payload), \
            patch('tricoopdiff.models.expert.build_diffdock_expert',
                  return_value=SimpleNamespace(node_scalar_dim=58, ns=48, node_vector_dim=20)):
        try:
            build_trainer(args, 'cpu', input_dim=4)
        except ValueError as error:
            assert 'historical coupled near-target recipe' in str(error)
        else:
            raise AssertionError('production silently resumed the coupled v2 recipe')


def test_v4_resume_preserves_cycle_and_rejects_historical_recipe():
    import copy
    from tricoopdiff.train import LEGACY_NEAR_TARGET_SUPERVISION

    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v4')
    trainer.samples_seen = 5
    payload = copy.deepcopy(trainer.checkpoint_payload())
    assert payload['runtime_config']['training_recipe'] == 'endpoint_near_target_v2'
    assert payload['runtime_config']['near_target_supervision']['start_times'] == [1., .2]
    restored, _, _ = _endpoint_trainer_fixture('ternary_endpoint_v4')
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        restored.load_checkpoint('unused.pt')
    assert restored.samples_seen == 5
    with patch.object(restored, 'compute_loss', return_value={'loss_total': restored.model.trifuse.motion.sum()}) as loss:
        restored.train_step(**batch, coupling_override=coupling)
    assert loss.call_args.kwargs['near_target_zero'] is True
    assert loss.call_args.kwargs['near_target_start_t'] == .2
    payload['runtime_config']['training_recipe'] = 'endpoint_near_target_v1'
    payload['runtime_config']['near_target_supervision'] = LEGACY_NEAR_TARGET_SUPERVISION
    with patch('tricoopdiff.train.torch.load', return_value=payload):
        try:
            restored.load_checkpoint('unused.pt')
        except ValueError as error:
            assert 'recipe' in str(error)
        else:
            raise AssertionError('v3 silently resumed the obsolete coupled schedule')


def test_failed_stochastic_microbatch_restores_rng_and_entry_phase():
    from tricoopdiff.runtime import capture_rng_state

    trainer, batch, coupling = _endpoint_trainer_fixture('ternary_endpoint_v3')
    trainer.samples_seen = 2  # This phase uses actual random near-target perturbations.
    trainer.effective_batch_size = 2
    before = capture_rng_state()
    raw = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    ema = {name: value.clone() for name, value in trainer.ema.items()}
    original = trainer.compute_loss
    def compute(**arguments):
        random.random()
        np.random.random()
        torch.rand(1)
        if arguments.get('case') == 'bad':
            raise FloatingPointError('failed second stochastic microbatch')
        return original(**arguments)
    with patch.object(trainer, 'compute_loss', side_effect=compute):
        try:
            trainer.train_batch([{**batch, 'coupling_override': coupling},
                                 {**batch, 'coupling_override': coupling, 'case': 'bad'}])
        except FloatingPointError:
            pass
        else:
            raise AssertionError('failed stochastic group was accepted')
    after = capture_rng_state()
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])
    assert all(torch.equal(a, b) for a, b in zip(before['torch_cuda'], after['torch_cuda']))
    assert trainer.global_step == 0 and trainer.samples_seen == 2
    assert not trainer.optimizer.state
    assert all(torch.equal(value, raw[name]) for name, value in trainer.model.trifuse.state_dict().items())
    assert all(torch.equal(value, ema[name]) for name, value in trainer.ema.items())
    assert all(parameter.grad is None for parameter in trainer.model.trifuse.parameters())

    # A finite loss may still produce a non-finite Jacobian at backward/clip.
    with torch.no_grad():
        trainer.model.trifuse.motion.zero_()
    clip_rng = capture_rng_state()
    def bad_gradient(**arguments):
        torch.rand(1)
        return {'loss_total': trainer.model.trifuse.motion.sum().sqrt()}
    with patch.object(trainer, 'compute_loss', side_effect=bad_gradient):
        try:
            trainer.train_step(**batch, coupling_override=coupling)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('non-finite gradients were accepted')
    assert torch.equal(clip_rng['torch_cpu'], torch.get_rng_state())
    assert trainer.global_step == 0 and trainer.samples_seen == 2
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in trainer.model.trifuse.parameters())


def test_optimizer_manifest_rejects_same_shape_moment_swaps_before_live_mutation():
    import copy
    import warnings

    trainer, _, _ = _endpoint_trainer_fixture('ternary_endpoint_v4')
    trainer.model.trifuse.extra = torch.nn.Parameter(trainer.model.trifuse.motion.detach().clone())
    trainer.ema['extra'] = trainer.model.trifuse.extra.detach().clone()
    first, second = trainer.model.trifuse.motion, trainer.model.trifuse.extra
    trainer._configure_optimizer([{'params': [first, second], 'lr': 1e-4, 'name': 'trifuse'}], 1e-4)
    with patch.object(trainer, 'compute_loss', return_value={'loss_total': first.sum() + 2 * second.sum()}):
        trainer.train_step()
    assert not torch.equal(trainer.optimizer.state[first]['exp_avg'], trainer.optimizer.state[second]['exp_avg'])
    payload = copy.deepcopy(trainer.checkpoint_payload())
    assert payload['optimizer_parameter_manifest'] == {0: 'trifuse.motion', 1: 'trifuse.extra'}
    optimizer = trainer.optimizer
    before = {name: value.clone() for name, value in trainer.model.trifuse.state_dict().items()}
    for corruption in ('reverse_ids', 'swap_names', 'missing_manifest', 'missing_contract_and_manifest', 'orphan_state'):
        broken = copy.deepcopy(payload)
        broken['trifuse_state_dict'] = {name: value + 1 for name, value in before.items()}
        if corruption == 'reverse_ids':
            broken['optimizer_state_dict']['param_groups'][0]['params'].reverse()
        elif corruption == 'swap_names':
            broken['optimizer_parameter_manifest'] = {0: 'trifuse.extra', 1: 'trifuse.motion'}
        elif corruption == 'missing_manifest':
            broken.pop('optimizer_parameter_manifest')
        elif corruption == 'missing_contract_and_manifest':
            broken.pop('optimizer_mapping_contract')
            broken.pop('optimizer_parameter_manifest')
        else:
            broken['optimizer_state_dict']['state'][999] = copy.deepcopy(broken['optimizer_state_dict']['state'][0])
        with patch('tricoopdiff.train.torch.load', return_value=broken):
            try:
                trainer.load_checkpoint('unused.pt')
            except ValueError as error:
                assert 'optimizer' in str(error)
            else:
                raise AssertionError(f'optimizer silently accepted {corruption}')
        assert trainer.optimizer is optimizer
        assert all(torch.equal(value, before[name]) for name, value in trainer.model.trifuse.state_dict().items())
    # Old v4/27 checkpoints remain readable, without pretending names were known.
    legacy = copy.deepcopy(payload)
    legacy['checkpoint_version'] = 27
    legacy.pop('optimizer_mapping_contract')
    legacy.pop('optimizer_parameter_manifest')
    with warnings.catch_warnings(record=True) as caught, \
            patch('tricoopdiff.train.torch.load', return_value=legacy):
        warnings.simplefilter('always')
        trainer.load_checkpoint('unused.pt')
    assert any('trusts registration order' in str(item.message) for item in caught)
    assert torch.equal(trainer.optimizer.state[first]['exp_avg'], optimizer.state[first]['exp_avg'])
    assert torch.equal(trainer.optimizer.state[second]['exp_avg'], optimizer.state[second]['exp_avg'])


def test_state_recovery_uses_actual_prefix_and_remaining_clock():
    from tricoopdiff.models.consensus import PRODUCTION_MODEL_PROFILE
    from tricoopdiff.train import validate_consensus_checkpoint

    trainer, batch, coupling = _endpoint_trainer_fixture(PRODUCTION_MODEL_PROFILE, inference_steps=8)
    trainer.model.trifuse.enhanced_conditions = True
    result = trainer.compute_loss(**batch, coupling_override=coupling, recovery_start_t=.75, diagnostics=True)
    assert len(trainer.model.calls) == 14, 'recovery must execute only six remaining steps after the full eight'
    assert trainer.model.calls[8][0] == .75
    assert 'loss_state_recovery' in result['components'] and 'recovery_state' not in result
    result['loss_total'].backward()
    assert torch.isfinite(trainer.model.trifuse.motion.grad).all()
    trainer.effective_batch_size = 2
    trainer.model.calls.clear()
    metrics = trainer.train_batch([{**batch, 'coupling_override': coupling}] * 2)
    assert all('loss_state_recovery' in item for item in metrics)
    assert trainer.global_step == 1 and trainer.samples_seen == 2
    payload = trainer.checkpoint_payload()
    validate_consensus_checkpoint(payload)
    payload['runtime_config']['state_recovery_supervision'] = {
        **payload['runtime_config']['state_recovery_supervision'], 'weight': .9}
    try:
        validate_consensus_checkpoint(payload)
    except ValueError:
        pass
    else:
        raise AssertionError('unrecorded state-recovery loss was accepted')
    try:
        trainer.validate_step(**batch, coupling_override=coupling, recovery_start_t=.75)
    except ValueError:
        pass
    else:
        raise AssertionError('training-only recovery entered validation selection loss')


if __name__ == "__main__":
    torch.set_num_threads(2)
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check()
            print(name, 'passed')
    print("Training-contract tests passed")
