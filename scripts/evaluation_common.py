"""Pure-Python helpers shared by evaluation entry points and tests."""

import hashlib
import json
import math
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
    if payload.get("checkpoint_version") != 16:
        raise ValueError(
            "TriFuse evaluation requires checkpoint_version=16"
        )
    if payload.get("architecture") != "trifuse_v13":
        raise ValueError(
            "checkpoint architecture is not trifuse_v13"
        )
    activation_clamp = payload.get("runtime_config", {}).get("activation_clamp")
    if not isinstance(activation_clamp, (int, float)) or activation_clamp <= 0:
        raise ValueError("checkpoint has no valid runtime activation_clamp")
    torsion_aux_weight = payload.get("runtime_config", {}).get(
        "torsion_expert_aux_weight"
    )
    if not isinstance(torsion_aux_weight, (int, float)) or torsion_aux_weight < 0:
        raise ValueError("checkpoint has no valid torsion_expert_aux_weight")
    lora_warmup_steps = payload.get("runtime_config", {}).get(
        "lora_warmup_steps"
    )
    if not isinstance(lora_warmup_steps, int) or lora_warmup_steps < 0:
        raise ValueError("checkpoint has no valid lora_warmup_steps")
    optimizer_groups = payload.get("runtime_config", {}).get("optimizer_groups")
    if not isinstance(optimizer_groups, dict) or not optimizer_groups:
        raise ValueError("checkpoint has no optimizer-group training config")
    if not payload.get("trifuse_state_dict"):
        raise ValueError("checkpoint has no TriFuse weights")
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
        and config.get("trifuse_checkpoint") == checkpoint
        and ensemble.get("n_valid_walkers", 0) > 0
    )
    dockq = report.get("structure", {}).get("dockq_best")
    exact_dockq_matches = (
        not require_exact_dockq
        or (isinstance(dockq, (int, float)) and math.isfinite(dockq))
    )
    expected_walkers = (expected_config or {}).get("n_walkers")
    walker_count_matches = (
        expected_walkers is None
        or ensemble.get("n_walkers") == expected_walkers
    )
    return core_matches and exact_dockq_matches and walker_count_matches and all(
        config.get(key) == value for key, value in (expected_config or {}).items()
    )


def sha256_file(path, chunk_size=1024 * 1024):
    """Return a reproducibility hash without loading a large file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_input_signature(paths, native_complex_pdb=None):
    """Fingerprint actual case inputs, including the reference used for metrics."""
    sources = {**paths, "native_complex_pdb": native_complex_pdb}
    payload = {
        key: ({"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
              if path is not None else None)
        for key, path in sources.items()
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
