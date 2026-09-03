"""CPU-only tests for indexed and checkpointed evaluation preflight."""

import importlib.util
import json
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluation_common.py"
_SPEC = importlib.util.spec_from_file_location("evaluation_common", _SCRIPT)
evaluation_common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation_common)


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
    paths = evaluation_common.resolve_entry_paths(selected, tmp_path)
    assert all(path.is_file() for path in paths.values())


def test_checkpoint_preflight_rejects_missing_lora():
    payload = {"checkpoint_version": 3, "tribrain_state_dict": {"x": 1}}
    try:
        evaluation_common.validate_checkpoint_payload(payload)
    except ValueError as error:
        assert "LoRA" in str(error)
    else:
        raise AssertionError("checkpoint without LoRA must be rejected")


def test_checkpoint_preflight_accepts_complete_payload():
    payload = {
        "checkpoint_version": 3,
        "tribrain_state_dict": {"x": 1},
        "expert_lora_state_dict": {str(i): i for i in range(336)},
        "expert_lora_config": {"rank": 4},
    }
    evaluation_common.validate_checkpoint_payload(payload)


def test_completed_report_requires_matching_checkpoint(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "case": "demo",
        "run_config": {"tribrain_checkpoint": "best.pt"},
        "structure": {"n_valid_walkers": 31},
    }), encoding="utf-8")
    assert evaluation_common.completed_report(path, "demo", "best.pt")
    assert not evaluation_common.completed_report(path, "demo", "other.pt")
