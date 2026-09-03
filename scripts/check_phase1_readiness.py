#!/usr/bin/env python
"""Check that sampling/frame gates and Phase 1 inputs are ready."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-report", default="results/sampling_parity_regression.json")
    parser.add_argument("--frame-report", default="results/initialization_frame_check.json")
    parser.add_argument("--data-index", default="data/protac_index.json")
    parser.add_argument("--model-dir", default="checkpoints/score_model")
    parser.add_argument("--out", default="results/phase1_readiness.json")
    parser.add_argument(
        "--training-only", action="store_true",
        help="Check data and expert weights only; defer sampling gates to inference",
    )
    return parser


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def report_gate(path, key, validator=None):
    if not path.is_file():
        return False, f"missing report: {path}"
    try:
        report = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid JSON report: {error}"
    passed = report.get(key) is True
    if not passed:
        return False, f"{key} is not true"
    if validator is not None:
        validation_error = validator(report)
        if validation_error:
            return False, validation_error
    return True, "pass"


def validate_parity_report(report):
    if report.get("init_mode") not in {"coupled5a", "anchor_local"}:
        return "parity report used an unsupported ternary initialization mode"
    if str(report.get("nan_clamp")).lower() != "none":
        return "parity report was not generated with nan_clamp=none"
    return None


def validate_frame_report(report):
    if report.get("init_mode") not in {"coupled5a", "anchor_local"}:
        return "frame report used an unsupported ternary initialization mode"
    checks = report.get("checks", {})
    if checks.get("poi_body_frame") is not True:
        return "frame report does not verify the POI body-frame rotation"
    metrics = report.get("metrics", {})
    if metrics.get("cache_version") != metrics.get("expected_cache_version"):
        return "frame report used a stale featurization cache"
    return None


def inspect_index(path):
    if not path.is_file():
        return False, 0, f"missing data index: {path}"
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return False, 0, f"invalid data index: {error}"
    entries = value.get("entries", []) if isinstance(value, dict) else value
    if not isinstance(entries, list) or not entries:
        return False, 0, "data index contains no entries"
    required = ("case", "protein1", "protein2", "ligand")
    for position, entry in enumerate(entries):
        missing = [key for key in required if not entry.get(key)]
        if missing:
            return False, len(entries), f"entry {position} missing keys: {missing}"
        resolved_paths = []
        for key in required[1:]:
            candidate = Path(entry[key])
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if not candidate.is_file():
                candidate = (
                    PROJECT_ROOT / "DeepTernary" / "output" / "protac22"
                    / entry["case"] / Path(entry[key]).name
                )
            resolved_paths.append(candidate)
        missing_files = [str(candidate) for candidate in resolved_paths if not candidate.is_file()]
        if missing_files:
            return False, len(entries), f"entry {position} missing files: {missing_files}"
    return True, len(entries), "pass"


def main():
    args = build_parser().parse_args()
    parity_path = resolve(args.parity_report)
    frame_path = resolve(args.frame_report)
    index_path = resolve(args.data_index)
    model_dir = resolve(args.model_dir)

    parity_ok, parity_detail = report_gate(
        parity_path, "parity_pass", validate_parity_report
    )
    frame_ok, frame_detail = report_gate(
        frame_path, "pass", validate_frame_report
    )
    index_ok, entry_count, index_detail = inspect_index(index_path)
    checkpoint_files = {
        "parameters": model_dir / "model_parameters.yml",
        "weights": model_dir / "best_ema_inference_epoch_model.pt",
    }
    checkpoint_ok = all(path.is_file() for path in checkpoint_files.values())

    checks = {"data_index": index_ok, "expert_checkpoint": checkpoint_ok}
    if not args.training_only:
        checks = {
            "sampling_parity": parity_ok,
            "initialization_frames": frame_ok,
            **checks,
        }
    details = {
        "data_index": index_detail,
        "data_entries": entry_count,
        "checkpoint_files": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in checkpoint_files.items()
        },
        "scope": "training" if args.training_only else "training_and_sampling",
    }
    if not args.training_only:
        details.update({
            "sampling_parity": parity_detail,
            "initialization_frames": frame_detail,
        })
    report = {
        "ready": all(checks.values()),
        "checks": checks,
        "details": details,
    }
    output_path = resolve(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
