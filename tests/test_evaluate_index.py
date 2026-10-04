"""CPU-only tests for deterministic evaluation slicing."""

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_index.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("evaluate_index", _SCRIPT)
evaluate_index = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_index)


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
