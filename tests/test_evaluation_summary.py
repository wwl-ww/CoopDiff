"""CPU-only tests for evaluation aggregation."""

import importlib.util
import json
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts"
    / "summarize_evaluation_reports.py"
)
_SPEC = importlib.util.spec_from_file_location("evaluation_summary", _SCRIPT)
evaluation_summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation_summary)


def test_summary_tracks_failures_walkers_and_quantiles():
    rows = [
        {"case": "a", "n_valid_walkers": 31, "n_nan_walkers": 1,
         "dockq_best": 0.8},
        {"case": "b", "n_valid_walkers": 0, "n_nan_walkers": 32,
         "dockq_best": None},
        {"case": "c", "n_valid_walkers": 32, "n_nan_walkers": 0,
         "dockq_best": 0.6},
    ]
    summary = evaluation_summary.summarize_rows(rows)
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
    row = evaluation_summary.load_row(path)
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
    row = evaluation_summary.load_row(path)
    assert not row['exact_dockq'], 'one scored seed must not certify all eight'
    summary = evaluation_summary.summarize_rows([row])
    assert summary['n_requested_walkers'] == 32
    assert summary['valid_walker_rate'] == .25
    assert summary['n_exact_dockq_cases'] == 0
    report['ensemble']['n_exact_dockq_walkers'] = 8
    path.write_text(json.dumps(report))
    assert not evaluation_summary.load_row(path)['exact_dockq'], '24 unaccounted seeds remain missing'
    report['ensemble']['n_nan_walkers'] = 24
    for invalid in (True, float('nan'), float('inf'), 1.1, -.1):
        report['structure']['dockq_best'] = invalid
        path.write_text(json.dumps(report))
        assert not evaluation_summary.load_row(path)['exact_dockq']
