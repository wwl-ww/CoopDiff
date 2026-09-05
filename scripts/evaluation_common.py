"""Pure-Python helpers shared by evaluation entry points and tests."""

import hashlib
import json
from pathlib import Path


def load_index_entry(index_path, case):
    """Return one uniquely named entry from a ternary JSON index."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    matches = [entry for entry in entries if entry.get("case") == case]
    if not matches:
        raise KeyError(f"case {case!r} is absent from {index_path}")
    if len(matches) != 1:
        raise ValueError(f"case {case!r} occurs {len(matches)} times in {index_path}")
    return matches[0]


def resolve_entry_paths(entry, project_root):
    """Resolve and validate the three structure paths in an index entry."""
    root = Path(project_root)
    resolved = {}
    for key in ("protein1", "protein2", "ligand"):
        value = Path(entry[key])
        path = value if value.is_absolute() else root / value
        if not path.is_file():
            raise FileNotFoundError(f"{key} file is missing: {path}")
        resolved[key] = path.resolve()
    return resolved


def validate_checkpoint_payload(payload, expected_lora_tensors=336):
    """Reject incomplete or legacy checkpoints before expensive inference."""
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("checkpoint_version", 0) < 3:
        raise ValueError("checkpoint_version must be at least 3")
    if not payload.get("tribrain_state_dict"):
        raise ValueError("checkpoint has no TriBrain weights")
    lora = payload.get("expert_lora_state_dict")
    if not lora:
        raise ValueError("checkpoint has no expert LoRA weights")
    if expected_lora_tensors is not None and len(lora) != expected_lora_tensors:
        raise ValueError(
            f"checkpoint has {len(lora)} LoRA tensors; "
            f"expected {expected_lora_tensors}"
        )
    if not payload.get("expert_lora_config"):
        raise ValueError("checkpoint has no LoRA architecture config")


def completed_report(path, case, checkpoint, expected_config=None,
                     require_exact_dockq=False):
    """Return true only for a valid, completed report with matching inputs."""
    destination = Path(path)
    if not destination.is_file():
        return False
    try:
        report = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    config = report.get("run_config", {})
    ensemble = report.get("ensemble", {})
    core_matches = bool(
        report.get("case") == case
        and config.get("tribrain_checkpoint") == checkpoint
        and ensemble.get("n_valid_walkers", 0) > 0
    )
    exact_dockq_matches = (
        not require_exact_dockq
        or report.get("structure", {}).get("dockq_best") is not None
    )
    return core_matches and exact_dockq_matches and all(
        config.get(key) == value for key, value in (expected_config or {}).items()
    )


def sha256_file(path, chunk_size=1024 * 1024):
    """Return a reproducibility hash without loading a large file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
