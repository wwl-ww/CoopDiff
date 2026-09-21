"""Incomplete coverage must not silently improve a reported panel mean."""

from pathlib import Path
import runpy


def test_panel_mean_requires_every_declared_case():
    summarize = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                   / 'scripts/evaluate_anchored_panel.py'))['summarize_panel']
    rows = [{'case': 'a', 'status': 'completed', 'oracle_dockq': .8},
            {'case': 'b', 'status': 'completed', 'oracle_dockq': .2}]
    full = summarize(rows, 2)
    assert full['complete'] and full['mean_case_oracle_dockq'] == .5
    assert full['model_top1_computed'] is False
    for partial in (rows[:1], [rows[0], {'case': 'b', 'status': 'failed'}]):
        report = summarize(partial, 2)
        assert not report['complete'] and report['mean_case_oracle_dockq'] is None
