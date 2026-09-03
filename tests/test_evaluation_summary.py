"""CPU-only tests for evaluation aggregation."""

import importlib.util
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
