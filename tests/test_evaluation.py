"""Evaluation regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem

import evaluate_index
import evaluate_selected_refinement as evaluation
import evaluation_common
import summarize_evaluation_reports
from evaluate_binary_interfaces import fresh_ligand, interface_metrics
from evaluation_common import sha256_file
from helpers import flow_example
from summarize_official_benchmark import main
from tricoopdiff.conditioning.condition_features import MANIFEST_VERSION, canonical_isomeric_smiles
from tricoopdiff.evaluate import compute_poi_dockq, evaluate_model
from tricoopdiff.inputs.graphs import transform_ligand_to_poi_frame
from tricoopdiff.legacy.diffusion import InterfaceDiffusion
from tricoopdiff.metrics.dockq import _ligand_records, _replace_coords, compute_fullatom_dockq
from tricoopdiff.metrics.ensemble import (
    analyze_ensemble,
    cooperativity_from_ddg,
    cooperativity_from_kd,
    cooperativity_ranking,
    energy_population_association,
)


# Evaluation common

EVALUATION_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation_common.py"


def test_retired_condition_checkpoint_rejected_even_with_opt_in():
    from unittest.mock import patch

    from evaluation_runner import EvaluationRunner

    payload = {'format': 'conditioned_consensus_training_v1'}
    for opt_in in (False, True):
        with patch('evaluation_runner.sha256_file', return_value='a' * 64), \
                patch('evaluation_runner.torch.load', return_value=payload):
            try:
                EvaluationRunner(EVALUATION_SCRIPT.parent.parent, 'retired.pt', device='cpu',
                                 allow_conditioned_checkpoint=opt_in)
            except ValueError as error:
                assert 'retired condition-token' in str(error)
            else:
                raise AssertionError('retired weights must not enter the nodewise architecture')


def test_runtime_rejects_overflowing_ema_without_publishing_model():
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch

    from evaluation_runner import EvaluationRunner

    class Model(torch.nn.Module):
        def __init__(self, frozen_expert, trifuse):
            super().__init__()
            self.trifuse = trifuse

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.model = runner.input_dim = None
    runner.conditioned_checkpoint = runner.experimental_checkpoint = False
    runner.device = torch.device('cpu')
    runner.expert = SimpleNamespace()
    runner.checkpoint = {'model_config': {'input_dim': 3},
                         'ema_trifuse_state_dict': {'weight': torch.full((1, 3), 1e300,
                                                                       dtype=torch.float64)}}
    with patch('evaluation_runner.TernaryAssembly', side_effect=lambda **kwargs:
               torch.nn.Linear(3, 1, bias=False)), patch('evaluation_runner.TernaryAssemblyModel', Model):
        try:
            runner._ensure_model(3)
        except ValueError as error:
            assert 'finite' in str(error)
        else:
            raise AssertionError('evaluation accepted weights overflowing its active dtype')
        assert runner.model is None and runner.input_dim is None
        runner.checkpoint['ema_trifuse_state_dict']['weight'].fill_(1.)
        runner._ensure_model(3)
        assert torch.isfinite(runner.model.trifuse.weight).all()
        assert not runner.model.training and runner.input_dim == 3


def test_index_entry_and_paths(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    entry = {"case": "demo"}
    for key in ("protein1", "protein2", "ligand"):
        path = case_dir / f"{key}.pdb"
        path.write_text("ATOM\n", encoding="utf-8")
        entry[key] = str(path.relative_to(tmp_path))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")

    selected = evaluation_common.load_index_entry(index, "demo")
    index.write_text(json.dumps([entry]), encoding="utf-8")
    assert evaluation_common.load_index_entry(index, "demo") == selected
    paths = evaluation_common.resolve_entry_paths(selected, tmp_path)
    assert all(path.is_file() for path in paths.values())


def test_legacy_checkpoint_preflight_rejects_missing_lora():
    payload = {
        "checkpoint_version": 17,
        "generative_process": "sdpi_flow_v1",
        "architecture": "trifuse_v13",
        "runtime_config": {
            "activation_clamp": 100.0,
            "torsion_expert_aux_weight": 0.25,
            "lora_warmup_steps": 100,
            "optimizer_groups": {"trifuse": {"lr": 1e-4}},
        },
        "trifuse_state_dict": {"x": 1},
    }
    try:
        evaluation_common.validate_legacy_checkpoint_payload(payload)
    except ValueError as error:
        assert "LoRA" in str(error)
    else:
        raise AssertionError("checkpoint without LoRA must be rejected")


def test_checkpoint_preflight_rejects_legacy_architecture():
    payload = {
        "checkpoint_version": 3,
        "trifuse_state_dict": {"x": 1},
        "expert_lora_state_dict": {str(i): i for i in range(336)},
        "expert_lora_config": {"rank": 4},
    }
    try:
        evaluation_common.validate_checkpoint_payload(payload)
    except ValueError as error:
        assert "checkpoint_version=18" in str(error)
    else:
        raise AssertionError("legacy architecture checkpoint must be rejected")


def test_legacy_checkpoint_preflight_accepts_complete_payload():
    payload = {
        "checkpoint_version": 17,
        "generative_process": "sdpi_flow_v1",
        "architecture": "trifuse_v13",
        "runtime_config": {
            "activation_clamp": 100.0,
            "torsion_expert_aux_weight": 0.25,
            "lora_warmup_steps": 100,
            "optimizer_groups": {"trifuse": {"lr": 1e-4}},
        },
        "trifuse_state_dict": {"x": 1},
        "expert_lora_state_dict": {str(i): i for i in range(336)},
        "expert_lora_config": {"rank": 4},
    }
    evaluation_common.validate_legacy_checkpoint_payload(payload)


def test_legacy_checkpoint_preflight_rejects_missing_runtime_config():
    payload = {
        "checkpoint_version": 17,
        "generative_process": "sdpi_flow_v1",
        "architecture": "trifuse_v13",
        "trifuse_state_dict": {"x": 1},
        "expert_lora_state_dict": {str(i): i for i in range(336)},
        "expert_lora_config": {"rank": 4},
    }
    try:
        evaluation_common.validate_legacy_checkpoint_payload(payload)
    except ValueError as error:
        assert "activation_clamp" in str(error)
    else:
        raise AssertionError("checkpoint without runtime clamp must be rejected")


def test_completed_report_requires_matching_checkpoint(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "case": "demo",
        "run_config": {"trifuse_checkpoint": "best.pt", "seed": 0},
        "ensemble": {"n_valid_walkers": 31},
    }), encoding="utf-8")
    assert evaluation_common.completed_report(path, "demo", "best.pt")
    assert evaluation_common.completed_report(
        path, "demo", "best.pt", {"seed": 0},
    )
    assert not evaluation_common.completed_report(
        path, "demo", "best.pt", {"seed": 1},
    )
    assert not evaluation_common.completed_report(path, "demo", "other.pt")
    assert not evaluation_common.completed_report(
        path, "demo", "best.pt", require_exact_dockq=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    report["structure"] = {"dockq_best": 0.5}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not evaluation_common.completed_report(
        path, "demo", "best.pt", require_exact_dockq=True,
    )
    report['ensemble']['n_exact_dockq_walkers'] = 31
    report['walkers'] = [dict(seed=seed, dockq=.5, dockq_fnat=.4,
                              dockq_lrms=2., dockq_irms=1.) for seed in range(31)]
    path.write_text(json.dumps(report), encoding='utf-8')
    assert evaluation_common.completed_report(path, 'demo', 'best.pt', require_exact_dockq=True)
    report['ensemble']['n_exact_dockq_walkers'] = 1
    path.write_text(json.dumps(report), encoding='utf-8')
    assert not evaluation_common.completed_report(
        path, 'demo', 'best.pt', require_exact_dockq=True,
    )


def test_exact_resume_requires_real_unique_seed_records(tmp_path):
    path = tmp_path / 'report.json'
    rows = [dict(seed=seed, dockq=.5, dockq_fnat=.4, dockq_lrms=2., dockq_irms=1.)
            for seed in (7, 8)]
    report = {'case': 'demo', 'run_config': {'trifuse_checkpoint': 'best.pt', 'seed': 7},
              'ensemble': {'n_walkers': 3, 'n_valid_walkers': 2,
                           'n_exact_dockq_walkers': 2, 'invalid_seeds': [9]},
              'structure': {'dockq_best': .5}, 'walkers': rows}
    path.write_text(json.dumps(report))
    assert evaluation_common.completed_report(path, 'demo', 'best.pt', require_exact_dockq=True)
    for invalid_rows in ([], rows[:1], [rows[0], rows[0]], [{**rows[0], 'seed': 17}, rows[1]],
                         [{**rows[0], 'dockq_fnat': None}, rows[1]]):
        path.write_text(json.dumps({**report, 'walkers': invalid_rows}))
        assert not evaluation_common.completed_report(path, 'demo', 'best.pt', require_exact_dockq=True)


def test_resume_fingerprints_detect_same_path_replacement(tmp_path):
    source = tmp_path / 'ligand.sdf'
    source.write_text('original chemical graph')
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'original weights')
    paths = {'ligand': source}
    signature = evaluation_common.evaluation_input_signature(paths)
    digest = evaluation_common.sha256_file(checkpoint)
    config = {'input_signature': signature, 'checkpoint_sha256': digest}
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({
        'case': 'demo',
        'run_config': {'trifuse_checkpoint': str(checkpoint), **config},
        'ensemble': {'n_valid_walkers': 1},
    }))
    assert evaluation_common.completed_report(report, 'demo', str(checkpoint), config)
    source.write_text('replacement chemical graph')
    changed_input = evaluation_common.evaluation_input_signature(paths)
    assert changed_input != signature
    assert not evaluation_common.completed_report(
        report, 'demo', str(checkpoint), {**config, 'input_signature': changed_input},
    )
    checkpoint.write_bytes(b'replacement weights')
    assert not evaluation_common.completed_report(
        report, 'demo', str(checkpoint),
        {**config, 'checkpoint_sha256': evaluation_common.sha256_file(checkpoint)},
    )
    native = tmp_path / 'native.pdb'
    native.write_text('reference')
    assert evaluation_common.evaluation_input_signature(paths, native) != changed_input


def test_resume_rejects_nonfinite_metric_and_incomplete_ensemble(tmp_path):
    path = tmp_path / 'report.json'
    report = {
        'case': 'demo',
        'run_config': {'trifuse_checkpoint': 'best.pt', 'n_walkers': 32},
        'ensemble': {'n_walkers': 1, 'n_valid_walkers': 1},
        'structure': {'dockq_best': 0.1},
    }
    path.write_text(json.dumps(report))
    assert not evaluation_common.completed_report(
        path, 'demo', 'best.pt', {'n_walkers': 32}, require_exact_dockq=True,
    )
    report['ensemble']['n_walkers'] = 32
    report['ensemble']['n_exact_dockq_walkers'] = 1
    for invalid in (None, True, -.1, 1.1, float('nan'), float('inf')):
        report['structure']['dockq_best'] = invalid
        path.write_text(json.dumps(report))
        assert not evaluation_common.completed_report(
            path, 'demo', 'best.pt', {'n_walkers': 32}, require_exact_dockq=True,
        )
    report['structure']['dockq_best'] = .1
    for invalid in (None, True, '1', 33):
        report['ensemble']['n_valid_walkers'] = invalid
        path.write_text(json.dumps(report))
        assert not evaluation_common.completed_report(path, 'demo', 'best.pt')
    path.write_text('[]')
    assert not evaluation_common.completed_report(path, 'demo', 'best.pt')


def test_resume_rejects_changed_numerical_policy(tmp_path):
    path = tmp_path / 'report.json'
    policy = {'deterministic_algorithms': True,
              'torus_score_norm_version': 'wrapped_normal_quadrature_v1'}
    report = {'case': 'demo',
              'run_config': {'trifuse_checkpoint': 'best.pt', **policy},
              'ensemble': {'n_valid_walkers': 1}}
    path.write_text(json.dumps(report))
    assert evaluation_common.completed_report(path, 'demo', 'best.pt', policy)
    for key, changed in (('deterministic_algorithms', False),
                         ('torus_score_norm_version', 'old')):
        assert not evaluation_common.completed_report(
            path, 'demo', 'best.pt', {**policy, key: changed})
    report['run_config'].pop('torus_score_norm_version')
    path.write_text(json.dumps(report))
    assert not evaluation_common.completed_report(path, 'demo', 'best.pt', policy)


# Evaluation walkers

def test_walker_metrics_preserve_seeds_and_exact_failures():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    samples = [dict(ligand_coords=coords, poi_coords=coords,
                    valid=valid, initialization={}) for valid in (True, False, True)]
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    exact = {'dockq': .5, 'fnat': .4, 'lrms': 2., 'irms': 1.}
    with ExitStack() as stack:
        stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=samples))
        stack.enter_context(patch('tricoopdiff.metrics.dockq.compute_fullatom_dockq',
                                  side_effect=[exact, RuntimeError('metric failed')]))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_dockq', 'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.metrics.ensemble.analyze_ensemble', return_value={
            'mean_pairwise_rmsd': 0., 'std_pairwise_rmsd': 0., 'n_clusters': 1}))
        report = evaluate_model(
            None, data, data, data, None, initializer=object(), device='cpu',
            n_walkers=3, seed=7, native_complex_pdb='dummy', deepternary_dockq_paths={})
    assert report['ensemble']['invalid_seeds'] == [8]
    assert report['ensemble']['n_exact_dockq_walkers'] == 1
    assert [row['seed'] for row in report['walkers']] == [7, 9]
    assert report['walkers'][0]['dockq'] == .5
    assert 'metric failed' in report['walkers'][1]['exact_dockq_error']
    assert all('pred' not in row and 'pred_poi' not in row for row in report['walkers'])
    assert report['structure']['dockq_best'] == .5


def test_mismatched_poi_contacts_are_not_reported_as_recovery():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    native = {'ligand': coords[:2], 'poi_ca': coords[:2], 'e3_ca': coords}
    with patch('tricoopdiff.evaluate._parse_native_complex', return_value=native):
        metric = compute_poi_dockq(coords, coords, 'unused')
    assert not metric['atom_correspondence']
    assert 'dockq' not in metric, 'density is not native contact recovery'
    assert metric['n_recovered_contacts'] is None
    assert metric['contact_density_ratio'] == 9 / 4

    sample = dict(ligand_coords=coords, poi_coords=coords, valid=True,
                  initialization={})
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    with ExitStack() as stack:
        stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=[sample]))
        stack.enter_context(patch('tricoopdiff.evaluate.compute_poi_dockq', return_value=metric))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.metrics.ensemble.analyze_ensemble', return_value={}))
        report = evaluate_model(None, data, data, data, None, initializer=object(),
                                device='cpu', n_walkers=1, native_complex_pdb='unused')
    assert report['structure']['poi_dockq_best'] is None
    assert report['structure']['poi_dockq_mean'] is None
    assert report['structure']['poi_dockq_median'] is None
    assert report['walkers'][0]['poi_contact_density_ratio'] == 9 / 4


def test_rank_features_are_saved_only_when_requested():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    samples = [dict(ligand_coords=coords, poi_coords=coords, valid=True,
                    initialization={}, rank_features=np.array([1., 2., 3.]))]
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    conditions = {'contract': object()}
    with ExitStack() as stack:
        sampler = stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=samples))
        debug = stack.enter_context(patch('tricoopdiff.sampler.sample_single', return_value=samples[0]))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_dockq', 'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.metrics.ensemble.analyze_ensemble', return_value={
            'mean_pairwise_rmsd': 0., 'std_pairwise_rmsd': 0., 'n_clusters': 1}))
        report = evaluate_model(
            SimpleNamespace(expert=SimpleNamespace(model=SimpleNamespace(no_torsion=True))),
            data, data, data, None, device='cpu',
            n_walkers=1, return_rank_features=True, conditions=conditions,
            debug_trace=True, translation_velocity_scale=2., rotation_velocity_scale=.5,
        )
    assert report['walkers'][0]['rank_features'] == [1.0, 2.0, 3.0]
    assert sampler.call_args.kwargs['conditions'] is conditions
    assert sampler.call_args.args[2].no_torsion, 'default initializer must match the expert'
    assert debug.call_args.kwargs['conditions'] is conditions
    assert debug.call_args.kwargs['translation_velocity_scale'] == 2.
    assert debug.call_args.kwargs['rotation_velocity_scale'] == .5


def test_rank_request_keeps_rejected_walkers_without_aborting_other_seeds():
    from tricoopdiff.initialization import TernaryInitializer
    from tricoopdiff.sampler import sample_ensemble

    flow, ligand_data, e3, poi, _ = flow_example()
    model = torch.nn.Module()
    model.expert = SimpleNamespace(model=SimpleNamespace(no_torsion=False))
    velocity = {'e3_ligand': {'tr': torch.zeros(1, 3), 'rot': torch.zeros(1, 3)},
                'poi_ligand': {'tr': torch.zeros(1, 3), 'rot': torch.zeros(1, 3)},
                'torsion': torch.zeros(1, int(ligand_data['edge_mask'].sum()))}
    bad = {**velocity, 'e3_ligand': {**velocity['e3_ligand'],
                                   'tr': torch.full((1, 3), float('nan'))}}
    with patch.object(flow, 'forward_state', side_effect=[
            {'normalized_velocity': bad}, {'normalized_velocity': velocity},
            {'rank_features': torch.tensor([[1., 2.]])}]):
        results = sample_ensemble(model, flow, TernaryInitializer(), {'coords': e3},
                                  {'coords': poi}, ligand_data, steps=1, device='cpu',
                                  seeds=[7, 8], return_rank_features=True)
    assert len(results) == 2
    assert not results[0]['valid'] and results[0]['termination_reason'] == 'nonfinite_score'
    assert results[0]['nonfinite_score_steps'][0]['components'] == ['e3_ligand.tr']
    assert 'rank_features' not in results[0]
    assert results[1]['valid'] and results[1]['rank_features'].tolist() == [1., 2.]


def test_unbound_metric_templates_receive_only_predicted_rigid_motion():
    def records(coords, chain):
        return ''.join(
            f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
            for i, (x, y, z) in enumerate(coords, 1))

    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([3., 4., 5.])
    template = np.vstack((coords + 10., [[12., 12., 12.]]))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, points, chain in (
                ('p1', coords, 'I'), ('p2', coords + 10., 'J'),
                ('m1', coords, 'A'), ('m2', template, 'B')):
            (root / f'{name}.pdb').write_text(records(points, chain))
        (root / 'lig.pdb').write_text(records(coords, 'A').replace('ATOM  ', 'HETATM'))
        native_path = root / 'native.pdb'
        native_path.write_text((root / 'm1.pdb').read_text() + (root / 'm2.pdb').read_text())

        def capture(predicted, native):
            lines = Path(predicted).read_text().splitlines()
            points = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                               for line in lines if line.startswith('ATOM') and line[21] == 'B'])
            assert np.allclose(points, template @ rotation + translation, atol=1e-3)
            assert not any(line[21] in ('I', 'J') for line in lines)
            return .4, 1., 2., .5

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}):
            result = compute_fullatom_dockq(
                coords, (coords + 10.) @ rotation + translation, coords,
                root / 'p1.pdb', root / 'p2.pdb', root / 'lig.pdb', native_path, root,
                metric_protein1_pdb=root / 'm1.pdb', metric_protein2_pdb=root / 'm2.pdb')
        assert result['dockq'] == .5


def test_metric_export_rejects_nonfinite_and_overflowing_pdb_columns():
    line = ('ATOM      1  CA  ALA A   1       0.000   0.000   0.000'
            '  1.00  0.00           C')
    for value in (10000., -1000., 9999.9996, float('nan'), float('inf')):
        try:
            _replace_coords([line], [0], np.array([[value, 2., 3.]]))
        except ValueError as error:
            assert 'PDB coordinates' in str(error)
        else:
            raise AssertionError('invalid coordinates must never shift DockQ PDB columns')
    output = _replace_coords([line], [0], np.array([[9999.999, -999.999, 3.]]))[0]
    assert len(output) == len(line)
    assert [float(output[start:start + 8]) for start in (30, 38, 46)] == [9999.999, -999.999, 3.]


def test_sdf_metric_template_preserves_heavy_atom_order():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    expected = molecule.GetConformer().GetPositions()[:3]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'ligand.sdf'
        Chem.MolToMolFile(molecule, str(path))
        _, indices, coords = _ligand_records(path)
    assert len(indices) == 3
    assert np.allclose(coords, expected, atol=1e-3)


# Evaluation summary


def test_summary_tracks_failures_walkers_and_quantiles():
    rows = [
        {"case": "a", "n_valid_walkers": 31, "n_nan_walkers": 1,
         "dockq_best": 0.8},
        {"case": "b", "n_valid_walkers": 0, "n_nan_walkers": 32,
         "dockq_best": None},
        {"case": "c", "n_valid_walkers": 32, "n_nan_walkers": 0,
         "dockq_best": 0.6},
    ]
    summary = summarize_evaluation_reports.summarize_rows(rows)
    assert summary["failed_cases"] == ["b"]
    assert summary["n_valid_walkers"] == 63
    assert summary["n_requested_walkers"] == 96
    assert summary["metrics"]["dockq_best"]["median"] == 0.7


def test_report_reads_walker_counts_from_ensemble(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({
        "case": "demo",
        "run_config": {"trifuse_checkpoint": "/abs/best.pt"},
        "ensemble": {"n_valid_walkers": 31, "n_nan_walkers": 1,
                     "n_exact_dockq_walkers": 31,
                     "selection_metric": "dockq",
                     "conformational_analysis": {
                         "n_clusters": 4, "dominant_fraction": 0.5,
                         "mean_pairwise_rmsd": 3.0,
                     }},
        "structure": {"dockq_best": 0.75},
    }))
    row = summarize_evaluation_reports.load_row(path)
    assert row["n_valid_walkers"] == 31
    assert row["n_nan_walkers"] == 1
    assert row["exact_dockq"] is True
    assert row["ensemble_n_clusters"] == 4


def test_partial_exact_scores_and_missing_walkers_are_not_complete(tmp_path):
    path = tmp_path / 'partial.json'
    report = {'case': 'demo', 'run_config': {'n_walkers': 32},
              'ensemble': {'n_valid_walkers': 8, 'n_nan_walkers': 0,
                           'n_exact_dockq_walkers': 1},
              'structure': {'dockq_best': .9}}
    path.write_text(json.dumps(report))
    row = summarize_evaluation_reports.load_row(path)
    assert not row['exact_dockq'], 'one scored seed must not certify all eight'
    summary = summarize_evaluation_reports.summarize_rows([row])
    assert summary['n_requested_walkers'] == 32
    assert summary['valid_walker_rate'] == .25
    assert summary['n_exact_dockq_cases'] == 0
    report['ensemble']['n_exact_dockq_walkers'] = 8
    path.write_text(json.dumps(report))
    assert not summarize_evaluation_reports.load_row(path)['exact_dockq'], '24 unaccounted seeds remain missing'
    report['ensemble']['n_nan_walkers'] = 24
    for invalid in (True, float('nan'), float('inf'), 1.1, -.1):
        report['structure']['dockq_best'] = invalid
        path.write_text(json.dumps(report))
        assert not summarize_evaluation_reports.load_row(path)['exact_dockq']


# Evaluate index


def test_select_entries_is_ordered_and_bounded(tmp_path):
    index = tmp_path / "test.json"
    index.write_text(json.dumps({
        "entries": [{"case": str(i)} for i in range(6)],
    }), encoding="utf-8")
    selected = evaluate_index.select_entries(index, start=2, limit=3)
    assert [entry["case"] for entry in selected] == ["2", "3", "4"]
    index.write_text(json.dumps([{"case": str(i)} for i in range(6)]))
    assert evaluate_index.select_entries(index, start=2, limit=3) == selected


def test_select_entries_rejects_duplicate_report_destinations(tmp_path):
    index = tmp_path / 'test.json'
    index.write_text(json.dumps({'entries': [{'case': 'same'}, {'case': 'same'}]}))
    try:
        evaluate_index.select_entries(index, start=0, limit=1)
    except ValueError as error:
        assert 'duplicate case' in str(error)
    else:
        raise AssertionError('duplicate cases must not overwrite or resume another report')


# Selected refinement contract

ROOT = Path(__file__).resolve().parents[1]


def selected_refinement_contract_metrics(dockq):
    return {'dockq': dockq, 'fnat': 0.3, 'lrms': 2.0, 'irms': 1.0}


def test_union_keeps_initialization_when_refinement_fails():
    row = evaluation.refinement_metric_fields(selected_refinement_contract_metrics(0.6))
    assert row['safe_dockq'] == 0.6 and row['safe_stage'] == 'initial'
    assert 'refined_dockq' not in row, 'a failed refinement is not a valid final pose'
    assert evaluation.summarize([row], 'safe_dockq')['hits_0.49'] == 1
    assert evaluation.summarize([row], 'refined_dockq')['count'] == 0


def test_runtime_cache_defaults_to_asset_repository_without_overriding_explicit_value():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, {}, clear=True):
            evaluation.configure_runtime_cache(root)
            assert os.environ['TRICOOPDIFF_CACHE_DIR'] == str(
                root / 'results/runtime_cache/diffdock')
        with patch.dict(os.environ, {'TRICOOPDIFF_CACHE_DIR': '/explicit'}, clear=True):
            evaluation.configure_runtime_cache(root)
            assert os.environ['TRICOOPDIFF_CACHE_DIR'] == '/explicit'


def test_union_selects_better_stage_and_rejects_nonfinite_scores():
    for final, expected, stage in ((0.2, 0.4, 'initial'), (0.7, 0.7, 'refined')):
        row = evaluation.refinement_metric_fields(selected_refinement_contract_metrics(0.4), selected_refinement_contract_metrics(final))
        assert row['safe_dockq'] == expected and row['safe_stage'] == stage
    try:
        evaluation.refinement_metric_fields(selected_refinement_contract_metrics(float('nan')))
    except ValueError:
        pass
    else:
        raise AssertionError('a non-finite initialization must not enter the union')


def test_resume_rejects_mismatch_legacy_and_overwrite_without_mutations():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        manifest = {'cases': ['demo'], 'input_signatures': {'demo': 'current'}}
        report = {'case': 'demo', 'status': 'complete', 'config': manifest}
        evaluation.atomic_json(manifest, out / 'manifest.json')
        evaluation.atomic_json(report, out / 'demo.json')
        assert 'demo' in evaluation.validate_resume_output(out, manifest)
        original = {path.name: path.read_bytes() for path in out.iterdir()}
        for config, resume in (
            ({**manifest, 'native_root': 'different'}, True),
            ({**manifest, 'input_signatures': {'demo': 'changed'}}, True),
            (manifest, False),
        ):
            try:
                evaluation.validate_resume_output(out, config, resume=resume)
            except ValueError as error:
                assert '--out-dir' in str(error)
            else:
                raise AssertionError('unsafe reuse or overwrite must be rejected')
            assert {path.name: path.read_bytes() for path in out.iterdir()} == original
        evaluation.atomic_json({'cases': ['demo']}, out / 'manifest.json')
        try:
            evaluation.validate_resume_output(out, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('unsigned legacy output must not be silently adopted')


def test_condition_preflight_catches_receptor_node_mismatch():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def pdb_line(serial, residue, number, x):
        return (f'ATOM  {serial:5d}  CA  {residue:>3} A{number:4d}    '
                f'{x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          C \n')

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        protein = pdb_line(1, 'ALA', 1, 0.) + pdb_line(2, 'GLY', 2, 10.) + 'END\n'
        for side in (1, 2):
            (root / f'protein{side}.pdb').write_text(protein)
        molecule = Chem.AddHs(Chem.MolFromSmiles('CC'))
        AllChem.EmbedMolecule(molecule, randomSeed=7)
        ligand = root / 'ligand.sdf'
        Chem.MolToMolFile(molecule, str(ligand))
        molecule = Chem.RemoveHs(molecule)
        manifest = {
            'schema': MANIFEST_VERSION, 'case': 'fixture',
            'ligand_atom_count': molecule.GetNumAtoms(),
            'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(molecule),
            'sides': [
                {'provenance': 'template', 'evidence_id': 'left',
                 'pocket_probabilities': [1., 0.]},
                {'provenance': 'template', 'evidence_id': 'right',
                 'pocket_probabilities': [1., 0.]},
            ],
            'linker_indices': [], 'protac': True,
        }
        condition = root / 'fixture.json'
        condition.write_text(json.dumps(manifest))
        paths = {'protein1': root / 'protein1.pdb', 'protein2': root / 'protein2.pdb',
                 'ligand': ligand}
        case_inputs = {'fixture': (paths, root / 'unused_native.pdb', {})}
        audit = evaluation.preflight_condition_manifests(
            {'fixture': condition}, case_inputs)
        assert audit['fixture']['protein_sizes'] == [2, 2]
        manifest['sides'][0]['pocket_probabilities'].append(0.)
        condition.write_text(json.dumps(manifest))
        try:
            evaluation.preflight_condition_manifests({'fixture': condition}, case_inputs)
        except ValueError as error:
            assert 'expected finite shape (2,)' in str(error)
        else:
            raise AssertionError('misaligned receptor pocket passed preflight')


def test_main_checks_all_inputs_before_writing_or_loading_models():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / 'out'
        entry = {'case': 'demo'}
        for key in ('protein1', 'protein2', 'ligand', 'metric_protein1', 'metric_protein2'):
            entry[key] = key + '.pdb'
            (root / entry[key]).write_text('original structure')
        native = root / 'native' / 'demo' / 'gt_complex.pdb'
        native.parent.mkdir(parents=True)
        native.write_text('original native')
        other_native = root / 'other_native' / 'demo' / 'gt_complex.pdb'
        other_native.parent.mkdir(parents=True)
        other_native.write_text('other native')
        (root / 'index.json').write_text(json.dumps({'entries': [entry]}))
        (root / 'model.pt').write_bytes(b'not loaded in this CPU test')
        expert = root / 'checkpoints' / 'score_model'
        expert.mkdir(parents=True)
        for name in ('model_parameters.yml', 'best_ema_inference_epoch_model.pt'):
            (expert / name).write_bytes(b'original expert asset')
        scorer = root / 'DeepTernary' / 'DockQ' / 'dockq_util.py'
        scorer.parent.mkdir(parents=True)
        scorer.write_text('# original metric implementation')
        argv = ['evaluate', '--index', str(root / 'index.json'), '--out-dir', str(out),
                '--checkpoint', str(root / 'model.pt'), '--device', 'cpu',
                '--native-root', 'native']
        old_cwd = Path.cwd()
        try:
            with patch.object(evaluation, 'ROOT', root), patch.object(sys, 'argv', argv), \
                    patch.object(evaluation, 'EvaluationRunner', side_effect=RuntimeError('CPU probe')) as runner, \
                    contextlib.redirect_stdout(io.StringIO()):
                # Obtain the actual manifest without constructing any model.
                try:
                    evaluation.main()
                except SystemExit as error:
                    assert error.code == 1
                assert runner.call_count == 1
                manifest = json.loads((out / 'manifest.json').read_text())
                evaluation.atomic_json({'case': 'demo', 'config': manifest, 'status': 'complete',
                                        'arms': {}, 'refinements': []}, out / 'demo.json')
                runner.reset_mock()
                evaluation.main()
                runner.assert_not_called()
                originals = {p.name: p.read_bytes() for p in out.iterdir()}
                # Same paths but changed contents, plus a changed native-root CLI argument.
                mutations = [root / entry[key] for key in entry if key != 'case']
                mutations += [native, scorer, expert / 'model_parameters.yml',
                              expert / 'best_ema_inference_epoch_model.pt', root / 'model.pt', None]
                for path in mutations:
                    saved = path.read_bytes() if path is not None else None
                    if path is not None:
                        path.write_bytes(saved + b' changed')
                    else:
                        argv[-1] = 'other_native'
                    try:
                        evaluation.main()
                    except ValueError as error:
                        assert '--out-dir' in str(error)
                    else:
                        raise AssertionError(f'changed input was silently reused: {path}')
                    finally:
                        if path is not None:
                            path.write_bytes(saved)
                        else:
                            argv[-1] = 'native'
                    runner.assert_not_called()
                    assert {p.name: p.read_bytes() for p in out.iterdir()} == originals
        finally:
            os.chdir(old_cwd)


# Official benchmark summary

def test_summary_accounts_for_missing_cases_and_seeds():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'model.pt'
        checkpoint.write_bytes(b'checkpoint')
        cases = root / 'cases.txt'
        cases.write_text('case1\ncase2\n')
        reports = root / 'reports'
        reports.mkdir()
        report = {
            'case': 'case1',
            'run_config': dict(checkpoint_sha256=sha256_file(checkpoint),
                               trifuse_checkpoint=str(checkpoint), n_walkers=2,
                               steps=20, seed=0, init_mode='sdpi', freeze_poi=False),
            'ensemble': dict(n_walkers=2, n_valid_walkers=2, n_nan_walkers=0,
                             n_exact_dockq_walkers=2, invalid_seeds=[], selection_metric='dockq'),
            'structure': {'dockq_best': .5, 'dockq_mean': .4},
            'walkers': [{'seed': 0, 'dockq': .5}, {'seed': 1, 'dockq': .3}],
        }
        (reports / 'case1.json').write_text(json.dumps(report))
        destination = root / 'summary.json'
        argv = ['summary', '--reports-root', str(reports), '--cases-file', str(cases),
                '--checkpoint', str(checkpoint), '--n-walkers', '2', '--out', str(destination)]

        def run(expected_exit):
            with patch.object(sys, 'argv', argv), patch('builtins.print'):
                try:
                    main()
                except SystemExit as error:
                    assert error.code == expected_exit
                else:
                    raise AssertionError('summary must report completeness via exit status')
            return json.loads(destination.read_text())['summary']

        summary = run(1)
        assert summary['expected_cases'] == 2
        assert summary['case_success_rates_including_missing']['0.23']['rate'] == .5
        report['case'] = 'case2'
        report['walkers'] = [{'seed': 0, 'dockq': .5}, {'seed': 1, 'exact_dockq_error': 'failed'}]
        report['ensemble']['n_exact_dockq_walkers'] = 1
        (reports / 'case2.json').write_text(json.dumps(report))
        summary = run(1)
        assert summary['exact_scored_walkers'] == 3
        assert not summary['complete_all_cases_and_seeds']


# Ensemble analysis

def test_report_diversity_keeps_rigid_pose_displacement():
    from tricoopdiff.evaluate import compute_ensemble_diversity

    pose = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    result = compute_ensemble_diversity([pose, pose + [10., 0., 0.]])
    assert result['mean_pairwise_rmsd'] == 10.
    assert result['n_clusters'] == 2
    assert result['coordinate_frame'] == 'fixed_e3_v1'


def test_ensemble_report_computes_pair_distances_once():
    from unittest.mock import patch
    from tricoopdiff.metrics import ensemble as ensemble_analysis

    with patch.object(ensemble_analysis, 'pairwise_pose_rmsd',
                      wraps=ensemble_analysis.pairwise_pose_rmsd) as distances:
        report = analyze_ensemble([ensemble_analysis_pose(0.), ensemble_analysis_pose(5.)])
    assert distances.call_count == 1 and report['n_clusters'] == 2


def ensemble_analysis_pose(offset):
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]) + offset


def test_ensemble_clusters_and_population_free_energy():
    poses = [ensemble_analysis_pose(0.0), ensemble_analysis_pose(0.1), ensemble_analysis_pose(0.2), ensemble_analysis_pose(5.0)]
    report = analyze_ensemble(poses, rmsd_cutoff=1.0)
    assert report['n_clusters'] == 2
    assert report['clusters'][0]['count'] == 3
    assert report['clusters'][0]['relative_free_energy_kcal_mol'] == 0.0
    assert report['clusters'][1]['relative_free_energy_kcal_mol'] > 0.0
    assert 'not absolute binding free energy' in report['free_energy_interpretation']


def test_cooperativity_conversions_and_ranking():
    assert cooperativity_from_kd(10.0, 2.0) == 5.0
    assert abs(cooperativity_from_ddg(0.0) - 1.0) < 1e-12
    assert cooperativity_from_ddg(-1.0) > 1.0
    ranking = cooperativity_ranking([-2., -1., 0., 1.], [8., 4., 2., 1.])
    assert abs(ranking['spearman_rho'] - 1.0) < 1e-12
    assert ranking['top_n_hit_rate'] == 1.0


def test_energy_population_uses_cluster_level_frequency():
    labels = [1, 1, 1, 2, 2, 3]
    energies = [-3.1, -3.0, -2.9, -2.0, -1.9, -1.0]
    report = energy_population_association(labels, energies)
    assert report['favorability_population_spearman_rho'] > 0.9
    assert report['dominant_cluster_energy_rank'] == 1


def test_energy_population_marks_undefined_correlation():
    report = energy_population_association([1, 1, 2, 2], [-2., -2., -1., -1.])
    assert report['favorability_population_spearman_rho'] is None


# Deepternary metric alignment


def deepternary_metric_alignment_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deepternary_metric_alignment_metrics = deepternary_metric_alignment_load("tricoopdiff_evaluate", ROOT / "tricoopdiff" / "evaluate.py")
deepternary_metric_alignment_exact = deepternary_metric_alignment_load("deepternary_metrics", ROOT / "tricoopdiff" / "metrics" / "dockq.py")
deepternary_metric_alignment_reference = deepternary_metric_alignment_load("deepternary_simple_dockq", ROOT / "DeepTernary" / "DockQ" / "simple_dockq.py")


def test_rigid_fit_recovers_known_transform():
    source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = source @ rotation + np.asarray([4., -2., 7.])
    fitted_r, fitted_t = deepternary_metric_alignment_exact._fit_rigid(source, target)
    assert np.max(np.abs(source @ fitted_r + fitted_t - target)) < 1e-10


def test_dockq_preserves_inserted_residues_without_duplicate_altloc_atoms():
    def atom(serial, insertion, alternate=' ', name='CA'):
        return (f'ATOM  {serial:5d} {name:^4}{alternate}GLY A  27{insertion}   '
                f'{serial:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}          C ')
    records = [atom(1, ' '), atom(2, 'A'), atom(3, 'B', 'A'), atom(4, 'B', 'B')]
    model, native = deepternary_metric_alignment_exact._dockq_protein_records(records, records)
    assert model == native and len(model) == 3
    assert [int(line[22:26]) for line in model] == [1, 2, 3]
    assert all(line[26] == ' ' and line[16] == ' ' for line in model)


def test_pose_dockq_aligns_extra_terminal_residues_before_numbering():
    residues = ('ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS',
                'CYS', 'MET', 'PRO', 'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE')

    def lines(chain, sequence):
        return [f'ATOM  {i:5d} CA   {name} {chain}{i:4d}    '
                f'{float(i):8.3f}{0.:8.3f}{0.:8.3f}'
                f'{1.:6.2f}{0.:6.2f}           C'
                for i, name in enumerate(sequence, 1)]

    native = lines('A', residues) + lines('B', residues)
    model = lines('A', residues) + lines('B', ('GLY', 'GLY') + residues)
    aligned = deepternary_metric_alignment_exact._align_pose_residue_numbers(model, native)
    chain_b = [line for line in aligned if line[21] == 'B']
    assert len(chain_b) == len(residues)
    assert [int(line[22:26]) for line in chain_b] == list(range(1, 26))
    assert chain_b[0][17:20] == 'ALA'


def test_exact_metric_maps_predictions_from_shared_model_frame_to_pdb_frame():
    e3_local = np.asarray([
        [0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.],
    ])
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.asarray([12., -7., 3.])
    e3_pdb = e3_local @ rotation + translation
    poi_pdb = np.asarray([[14., -3., 3.], [15., -3., 3.], [14., -2., 3.]])
    ligand_local = np.asarray([[1., 1., 2.], [3., 2., 1.]])
    poi_local = np.asarray([[4., 0., 0.], [5., 0., 0.], [4., 1., 0.]])

    def pdb_records(coords, chain, record, atom_name):
        return "\n".join(
            f"{record:<6}{i:5d} {atom_name:<4} ALA {chain}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C "
            for i, (x, y, z) in enumerate(coords, 1)
        ) + "\n"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1_path = root / "p1.pdb"
        p2_path = root / "p2.pdb"
        ligand_path = root / "ligand.pdb"
        p1_path.write_text(pdb_records(e3_pdb, "I", "ATOM", "CA"))
        p2_path.write_text(pdb_records(poi_pdb, "J", "ATOM", "CA"))
        ligand_path.write_text(pdb_records(ligand_local, "L", "HETATM", "C1"))
        native_path = root / 'native.pdb'
        native_path.write_text(p1_path.read_text() + p2_path.read_text() + ligand_path.read_text())

        def capture(predicted, native):
            assert all(not line.startswith('HETATM') for line in Path(native).read_text().splitlines())
            poi_lines = [
                line for line in Path(predicted).read_text().splitlines()
                if line.startswith('ATOM') and line[21] == 'J'
            ]
            assert 'HETATM' not in Path(predicted).read_text()
            exported = np.asarray([
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in poi_lines
            ])
            expected = poi_local @ rotation + translation
            assert np.allclose(exported, expected, atol=1e-3, rtol=0)
            return 1.0, 0.0, 0.0, 1.0

        with patch.dict(sys.modules, {
            "DockQ.dockq_util": SimpleNamespace(cal_dockq=capture),
        }), patch.object(
            deepternary_metric_alignment_exact, "_featurized_ca_coords", side_effect=[e3_pdb, poi_pdb],
        ):
            result = deepternary_metric_alignment_exact.compute_fullatom_dockq(
                ligand_local, poi_local, e3_local, p1_path, p2_path,
                ligand_path, native_path, root,
            )
        with patch.dict(sys.modules, {
            'DockQ.dockq_util': SimpleNamespace(cal_dockq=lambda *args: (1.5, 0., 0., .8)),
        }), patch.object(deepternary_metric_alignment_exact, '_featurized_ca_coords', side_effect=[e3_pdb, poi_pdb]):
            try:
                deepternary_metric_alignment_exact.compute_fullatom_dockq(
                    ligand_local, poi_local, e3_local, p1_path, p2_path,
                    ligand_path, native_path, root)
            except ValueError as error:
                assert 'invalid full-atom DockQ' in str(error)
            else:
                raise AssertionError('fnat > 1 must fail even if final DockQ is in range')
    assert result["dockq"] == 1.0


def test_sdf_ligand_records_preserve_atom_order():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(molecule, randomSeed=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ligand.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()
        lines, indices, coords = deepternary_metric_alignment_exact._ligand_records(path)
    heavy_atoms = [atom.GetIdx() for atom in molecule.GetAtoms()
                   if atom.GetAtomicNum() > 1]
    assert len(indices) == len(heavy_atoms)
    assert coords.shape == (len(heavy_atoms), 3)
    expected = molecule.GetConformer().GetPositions()[heavy_atoms]
    # SDF rounds to four decimals, then PDB to three; keep the heavy-atom order.
    assert np.allclose(coords, expected, atol=6e-4, rtol=0)
    assert all(lines[index].startswith(("ATOM", "HETATM")) for index in indices)


def test_approximate_metric_matches_deepternary_reference():
    ligand = np.asarray([[1., 0., 0.], [1., 1., 0.]])
    e3 = np.asarray([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    poi = np.asarray([[2., 0., 0.], [2., 1., 0.], [2., 2., 0.]])
    pred_ligand = ligand + np.asarray([0.3, 0.1, 0.])
    pred_poi = poi + np.asarray([0.7, 0.2, 0.])
    ours = deepternary_metric_alignment_metrics.compute_deepternary_dockq(
        ligand, e3, poi, pred_ligand, e3, pred_poi,
    )
    expected = deepternary_metric_alignment_reference.cal_approx_dockq(
        tuple(torch.as_tensor(x) for x in (ligand, e3, poi)),
        tuple(torch.as_tensor(x) for x in (pred_ligand, e3, pred_poi)),
    )
    for key, value in zip(("fnat", "lrms", "irms", "dockq"), expected):
        assert abs(ours[key] - float(value)) < 1e-7


def test_perfect_pose_is_one_and_interface_perturbation_degrades():
    ligand = np.asarray([[1., 0., 0.], [1., 1., 0.]])
    e3 = np.asarray([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    poi = np.asarray([[2., 0., 0.], [2., 1., 0.], [2., 2., 0.]])
    perfect = deepternary_metric_alignment_metrics.compute_deepternary_dockq(ligand, e3, poi, ligand, e3, poi)
    displaced = deepternary_metric_alignment_metrics.compute_deepternary_dockq(
        ligand, e3, poi, ligand + 10., e3, poi + 10.,
    )
    assert abs(perfect["dockq"] - 1.0) < 1e-12
    assert abs(perfect["fnat"] - 1.0) < 1e-12
    assert displaced["dockq"] < perfect["dockq"]


# Binary interfaces

def test_metrics_do_not_align_away_docking_error():
    native = torch.tensor([[[0., 0, 0], [20., 0, 0]]])
    receptor = torch.tensor([[[0., 2, 0], [0., -2, 0]]])
    result = interface_metrics(native + 3, native, receptor)
    assert result['interface_atom_count'] == 1
    assert abs(result['ligand_rmsd'] - 27 ** 0.5) < 1e-5
    assert abs(result['interface_atom_rmsd'] - 27 ** 0.5) < 1e-5


def test_initialization_ignores_native_coordinates():
    mol = Chem.MolFromSmiles('CC')
    ligand = {'mol': mol, 'features': torch.zeros(1, 2, 1),
              'coords': torch.zeros(1, 2, 3),
              'edge_index': torch.tensor([[0, 1], [1, 0]]),
              'edge_mask': torch.zeros(2, dtype=torch.bool),
              'mask_rotate': torch.zeros(0, 2, dtype=torch.bool)}
    receptor = torch.tensor([[0., 2, 0], [0., -2, 0]])
    first, _ = fresh_ligand(ligand, receptor, 42)
    ligand['coords'][:] = float('nan')
    second, _ = fresh_ligand(ligand, receptor, 42)
    assert torch.equal(first, second), 'native coordinates leaked into blind initialization'


def test_receptor_crop_preserves_all_geometrically_visible_nodes_with_empty_edges():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords = torch.arange(10).float()[:, None].expand(-1, 3) / 10
    features = torch.arange(10).float()[:, None]
    result = crop_receptor_to_ligand(coords, features, torch.zeros(1, 3),
                                     torch.empty(2, 0, dtype=torch.long), cutoff=10.,
                                     require_internal_edges=False)
    assert result['mask'].all(), 'fallback must not discard nodes already inside the radius'
    assert torch.equal(result['coords'], coords)
    assert torch.equal(result['features'], features)
    assert result['edge_index'].shape == (2, 0)


def test_receptor_crop_relabels_edges_without_reordering_nodes_or_detaching_inputs():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords = torch.tensor([[5., 0., 0.], [0., 0., 0.], [10., 0., 0.],
                           [1., 0., 0.], [20., 0., 0.]], requires_grad=True)
    features = torch.arange(5).float()[:, None].requires_grad_()
    edges = torch.tensor([[1, 3, 0, 2], [3, 1, 2, 0]])
    result = crop_receptor_to_ligand(coords, features, torch.zeros(1, 3), edges,
                                     cutoff=2., min_residues_with_edges=2)
    assert result['mask'].tolist() == [False, True, False, True, False]
    assert result['features'][:, 0].tolist() == [1., 3.]
    assert torch.equal(result['edge_index'], torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(edges, torch.tensor([[1, 3, 0, 2], [3, 1, 2, 0]]))
    (result['coords'].sum() + result['features'].sum()).backward()
    assert torch.equal(coords.grad, result['mask'][:, None].expand(-1, 3).float())
    assert torch.equal(features.grad, result['mask'][:, None].float())


def test_graph_entry_rejects_empty_components_but_preserves_singleton_inputs():
    from tricoopdiff.inputs.graphs import build_binary_graph, build_trifuse_graph, crop_receptor_to_ligand

    coords, features = torch.zeros(1, 3), torch.ones(1, 2)
    empty_coords, empty_features = torch.empty(0, 3), torch.empty(0, 2)
    invalid_calls = (
        lambda: build_binary_graph(empty_coords, empty_features, coords, features),
        lambda: build_binary_graph(coords, features, empty_coords, empty_features),
        lambda: build_binary_graph(coords, empty_features, coords, features),
        lambda: build_trifuse_graph(coords, features, empty_coords, empty_features, coords, features),
        lambda: crop_receptor_to_ligand(empty_coords, empty_features, coords, cutoff=None),
        lambda: crop_receptor_to_ligand(coords, features, empty_coords, cutoff=1.),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError as error:
            assert 'coordinates' in str(error) or 'features' in str(error)
        else:
            raise AssertionError('empty/mismatched nodes must fail at the shared graph boundary')
    graph = build_binary_graph(coords, features, coords, features,
                               rec_edge_index=torch.empty(2, 0, dtype=torch.long))
    assert graph['receptor'].pos.shape == graph['ligand'].pos.shape == (1, 3)
    crop = crop_receptor_to_ligand(coords, features, coords,
                                   rec_edge_index=torch.empty(2, 0, dtype=torch.long), cutoff=0.)
    assert crop['mask'].tolist() == [True] and crop['edge_index'].shape == (2, 0)


def test_receptor_crop_rejects_invalid_radius_and_fallback_counts():
    from tricoopdiff.inputs.graphs import crop_receptor_to_ligand

    coords, features = torch.zeros(2, 3), torch.ones(2, 1)
    for invalid in (-1., float('nan'), float('inf'), True, torch.tensor([1., 2.]), torch.tensor(True)):
        try:
            crop_receptor_to_ligand(coords, features, coords, cutoff=invalid)
        except ValueError as error:
            assert 'cutoff' in str(error)
        else:
            raise AssertionError('an invalid radius must not silently trigger a different crop')
    for name in ('min_residues', 'min_residues_with_edges'):
        for invalid in (0, -1, True, 1.5):
            try:
                crop_receptor_to_ligand(coords, features, coords, cutoff=1., **{name: invalid})
            except ValueError as error:
                assert 'minimum counts' in str(error)
            else:
                raise AssertionError('invalid fallback counts must not silently become another count')
    assert torch.equal(coords, torch.zeros(2, 3)) and torch.equal(features, torch.ones(2, 1))


# Interface coordinates

def interface_coordinates_triangle(center):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    ).unsqueeze(0) + torch.tensor(center).view(1, 1, 3)


def test_interface_composition_preserves_poi_side_relative_geometry():
    diffusion = InterfaceDiffusion(
        temp_sampling_tr=1.0,
        temp_sampling_rot=1.0,
    )
    ligand = interface_coordinates_triangle([2.0, -1.0, 0.5])
    poi = interface_coordinates_triangle([-3.0, 4.0, 1.0])
    target_ligand = interface_coordinates_triangle([7.0, 2.0, -3.0])
    tr_score = torch.tensor([[0.2, -0.3, 0.1]])
    rot_score = torch.tensor([[0.0, 0.0, 0.4]])
    time = torch.tensor([0.3])
    dt = torch.tensor([0.05])

    poi_side_proposal, _ = diffusion.reverse_step_protac(
        ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        tor_score=torch.empty(1, 0),
        t_tr=time,
        t_rot=time,
        t_tor=time,
        dt_tr=dt,
        dt_rot=dt,
        dt_tor=dt,
        add_noise=False,
    )
    updated_poi = diffusion.reverse_step_poi_ligand_interface(
        poi,
        source_ligand_pos=ligand,
        target_ligand_pos=target_ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        t_tr=time,
        t_rot=time,
        dt_tr=dt,
        dt_rot=dt,
        add_noise=False,
    )
    assert updated_poi.shape == poi.shape
    assert torch.isfinite(updated_poi).all()
    assert torch.equal(ligand, interface_coordinates_triangle([2.0, -1.0, 0.5]))
    # Applying one rigid transform to POI and the sub2 proposal preserves all
    # cross-interface distances when that proposal lands on the sub1 target.
    assert torch.allclose(
        torch.cdist(updated_poi, target_ligand),
        torch.cdist(poi, poi_side_proposal),
        atol=1e-5,
    )


def test_interface_pair_noise_reconstructs_finite_frames():
    torch.manual_seed(0)
    diffusion = InterfaceDiffusion()
    ligand = interface_coordinates_triangle([1.0, 2.0, 3.0])
    poi = interface_coordinates_triangle([8.0, -2.0, 1.0])
    noisy_ligand, noisy_poi, noise, global_to_poi = (
        diffusion.add_noise_interface_pair(
            ligand, poi, torch.tensor([0.4]), torch.tensor([0.4]),
        )
    )
    assert noisy_ligand.shape == ligand.shape
    assert noisy_poi.shape == poi.shape
    assert global_to_poi.shape == (1, 3, 3)
    assert torch.isfinite(noisy_ligand).all()
    assert torch.isfinite(noisy_poi).all()
    assert torch.allclose(
        torch.det(global_to_poi), torch.ones(1), atol=1e-5,
    )
    assert set(noise) == {"e3_ligand", "poi_ligand"}

    # The returned matrix is the inverse/body-frame map: the same rotation
    # that placed the clean POI into the global noisy frame must recover its
    # centered template coordinates when applied on the right to row vectors.
    recovered_poi = (
        noisy_poi - noisy_poi.mean(dim=1, keepdim=True)
    ) @ global_to_poi
    clean_centered = poi - poi.mean(dim=1, keepdim=True)
    assert torch.allclose(recovered_poi, clean_centered, atol=2e-5)

    local_vector = torch.tensor([[0.3, -0.7, 1.1]])
    global_vector = local_vector @ global_to_poi.transpose(-1, -2)
    assert torch.allclose(global_vector @ global_to_poi, local_vector, atol=1e-6)


def test_body_frame_transform_centers_poi_and_roundtrips_ligand():
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    e3_center = torch.tensor([0.0, 0.0, 0.0])
    poi_center = torch.tensor([6.0, -2.0, 3.0])
    ligand = torch.tensor([[7.0, -1.0, 3.5], [8.0, -3.0, 2.0]])
    local = transform_ligand_to_poi_frame(
        ligand, poi_center - e3_center, rotation,
    )
    recovered = local @ rotation.T + poi_center
    assert torch.allclose(recovered, ligand, atol=1e-6)


def test_poi_center_metric_uses_the_current_e3_coordinate_frame():
    import numpy as np
    from unittest.mock import patch
    from tricoopdiff.evaluate import compute_poi_com_dist

    e3 = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + np.array([5., 0., 0.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([20., -10., 3.])
    current_e3, current_poi = e3 @ rotation.T + translation, poi @ rotation.T + translation
    with patch('tricoopdiff.evaluate._parse_native_complex', return_value={'e3_ca': e3, 'poi_ca': poi}):
        legacy = compute_poi_com_dist(current_poi, 'unused')
        corrected = compute_poi_com_dist(current_poi, 'unused', current_e3)
        perturbed = compute_poi_com_dist(current_poi + np.array([0., 0., 2.]), 'unused', current_e3)
    assert legacy['poi_com_dist'] > 10, 'fixture must expose the legacy frame assumption'
    assert corrected['poi_com_dist'] < 1e-10
    assert abs(corrected['native_e3_poi_com_dist'] - 5.) < 1e-10
    assert abs(perturbed['poi_com_dist'] - 2.) < 1e-10


def test_exact_dockq_maps_shared_standalone_chain_ids_without_exiting():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from tricoopdiff.metrics.dockq import compute_fullatom_dockq

    def records(coords, chain):
        return ''.join(f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
                       f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
                       for i, (x, y, z) in enumerate(coords, 1))

    e3 = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + np.array([0., 0., 4.])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1, p2, native = (root / name for name in ('p1.pdb', 'p2.pdb', 'native.pdb'))
        p1.write_text(records(e3, 'A'))
        p2.write_text(records(poi, 'A'))
        native.write_text(records(e3, 'A') + records(poi, 'B'))
        visited = []

        def capture(model, reference):
            model_records = Path(model).read_text().splitlines()
            assert {line[21] for line in model_records} == {'A', 'B'}
            assert len(model_records) == 6
            visited.append(True)
            return 1., 0., 0., 1.

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[e3, poi, e3, poi]):
            result = compute_fullatom_dockq(e3, poi, e3, p1, p2, root / 'unused.pdb', native, root)
            assert result['dockq'] == 1. and visited == [True]
            assert {line[21] for line in p2.read_text().splitlines()} == {'A'}, 'metric must not mutate source PDBs'
            native.write_text(native.read_text() + records(e3, 'C'))
            try:
                compute_fullatom_dockq(e3, poi, e3, p1, p2, root / 'unused.pdb', native, root)
            except ValueError as error:
                assert 'exactly two' in str(error)
            else:
                raise AssertionError('unsupported native chain sets must raise, not sys.exit in DockQ')
            assert visited == [True]


def test_exact_dockq_preserves_explicit_native_chain_roles_when_swapped():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from tricoopdiff.metrics.dockq import compute_fullatom_dockq

    def records(coords, chain):
        return ''.join(f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
                       f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
                       for i, (x, y, z) in enumerate(coords, 1))

    a = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    b = a + np.array([0., 0., 4.])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p1, p2, native = (root / name for name in ('p1.pdb', 'p2.pdb', 'native.pdb'))
        p1.write_text(records(b, 'B'))
        p2.write_text(records(a, 'A'))
        native.write_text(records(a, 'A') + records(b, 'B'))

        def capture(model, reference):
            def coordinates(path, chain):
                return np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                                 for line in Path(path).read_text().splitlines() if line[21] == chain])
            for chain in ('A', 'B'):
                assert np.allclose(coordinates(model, chain), coordinates(reference, chain)), \
                    'explicit native chain identities must survive protein-role exchange'
            return 1., 0., 0., 1.

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[b, a]):
            result = compute_fullatom_dockq(a, a, b, p1, p2, root / 'unused.pdb', native, root)
        assert result['dockq'] == 1.
        for distorted_role in ('POI', 'E3'):
            distorted_poi = np.tile(a.mean(0), (len(a), 1)) if distorted_role == 'POI' else a
            distorted_e3 = np.tile(b.mean(0), (len(b), 1)) if distorted_role == 'E3' else b
            with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}), \
                    patch('tricoopdiff.metrics.dockq._featurized_ca_coords', side_effect=[b, a]):
                try:
                    compute_fullatom_dockq(a, distorted_poi, distorted_e3, p1, p2,
                                          root / 'unused.pdb', native, root)
                except ValueError as error:
                    assert f'{distorted_role} CA coordinates are not a rigid image' in str(error)
                else:
                    raise AssertionError('exact DockQ silently reconstructed a different protein shape')


def test_native_contact_parser_keeps_insertion_codes_and_default_altlocs():
    import tempfile
    import numpy as np
    from tricoopdiff.evaluate import _parse_native_complex

    def atom(serial, chain, insertion=' ', alternate=' ', x=0., record='ATOM  '):
        return (f'{record}{serial:5d}  CA {alternate}ALA {chain}  27{insertion}   '
                f'{x:8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00           C  \n')

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'native.pdb'
        path.write_text(atom(1, 'A') + atom(2, 'B', x=2.)
                        + atom(3, 'B', 'A', 'B', 99.)
                        + atom(4, 'B', 'A', 'A', 3.)
                        + atom(5, 'A', record='HETATM'))
        native = _parse_native_complex(path)
    assert native is not None
    assert np.array_equal(native['poi_ca'], np.array([[2., 0., 0.], [3., 0., 0.]])), \
        'inserted residues must remain separate, and altloc B must not replace featurized altloc A'


def test_equal_atom_counts_do_not_verify_poi_correspondence():
    coords = np.array([[0., 0., 0.], [1., 0., 0.]])
    native = {'ligand': coords, 'poi_ca': coords, 'e3_ca': coords}
    with patch('tricoopdiff.evaluate._parse_native_complex', return_value=native):
        metric = compute_poi_dockq(coords[::-1], coords, 'unused')
        verified = compute_poi_dockq(coords, coords, 'unused',
                                     atom_correspondence_verified=True)
    assert metric['atom_correspondence'] is False
    assert 'dockq' not in metric, 'matching counts cannot establish atom order'
    assert verified['atom_correspondence'] is True and verified['dockq'] == 1.


def test_evaluation_matcha_backend_preserves_identity_checks():
    from evaluation_runner import EvaluationRunner
    runtime = {'activation_clamp': 100., 'activation_clamp_mode': 'irrep_norm',
               'expert_runtime_identity': {'backend': 'matcha'}}
    checkpoint = {'model_config': {'expert_backend': 'matcha'},
                  'expert_base_identity': {'backend': 'matcha'},
                  'runtime_config': runtime, 'checkpoint_version': 24}
    expert = SimpleNamespace(provenance={'backend': 'matcha'})
    expert.to = lambda device: expert
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_dir = root/'model'
        model_dir.mkdir()
        (model_dir/'model_parameters.yml').write_text('num_conv_layers: 11\n')
        with ExitStack() as stack:
            stack.enter_context(patch('evaluation_runner.sha256_file', return_value='hash'))
            stack.enter_context(patch('evaluation_runner.torch.load', return_value=checkpoint))
            stack.enter_context(patch('evaluation_runner.validate_checkpoint_payload'))
            stack.enter_context(patch('evaluation_runner.validate_score_norm_version', return_value='norm'))
            stack.enter_context(patch('tricoopdiff.models.expert.configure_expert_policy'))
            stack.enter_context(patch('tricoopdiff.models.expert.expert_runtime_identity', return_value={'backend': 'matcha'}))
            base = stack.enter_context(patch('evaluation_runner.validate_expert_base_identity', return_value={'backend': 'matcha'}))
            matcha = stack.enter_context(patch('evaluation_runner.FrozenMatcha', return_value=expert))
            diffdock = stack.enter_context(patch('evaluation_runner.build_diffdock_expert'))
            runner = EvaluationRunner(root, 'checkpoint.pt', device='cpu', model_dir='model')
            assert runner.expert is expert and runner.expert_backend == 'matcha'
            matcha.assert_called_once()
            assert matcha.call_args.kwargs['device'] == 'cpu'
            diffdock.assert_not_called()
            base.assert_called_once_with(checkpoint, expert)
            runtime['expert_runtime_identity'] = {'backend': 'different'}
            try:
                EvaluationRunner(root, 'checkpoint.pt', device='cpu', model_dir='model')
            except ValueError as error:
                assert 'runtime differs' in str(error)
            else:
                raise AssertionError('Matcha runtime identity mismatch must fail')
