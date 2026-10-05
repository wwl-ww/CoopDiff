"""CPU-only tests for indexed and checkpointed evaluation preflight."""

import importlib.util
import json
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation_common.py"
_SPEC = importlib.util.spec_from_file_location("evaluation_common", _SCRIPT)
evaluation_common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation_common)


def test_runtime_rejects_overflowing_ema_without_publishing_model():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch

    sys.path.insert(0, str(_SCRIPT.parent))
    import tricoopdiff.diffdock_compat  # noqa: F401
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
    runner.checkpoint = {'model_config': {'input_dim': 3}, 'runtime_config': {},
                         'ema_trifuse_state_dict': {'weight': torch.full((1, 3), 1e300,
                                                                       dtype=torch.float64)}}
    with patch('evaluation_runner.ConsensusFuse', side_effect=lambda **kwargs:
               torch.nn.Linear(3, 1, bias=False)), patch('evaluation_runner.ConsensusCoopDiff', Model):
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
