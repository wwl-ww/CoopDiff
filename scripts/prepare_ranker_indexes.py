#!/usr/bin/env python
"""Prepare fixed non-benchmark indexes for the first pose-ranker experiment."""

import argparse
import json
from pathlib import Path


LEGACY_ROOT = Path("/tmp/coopdiff")


def relocate_entry(entry, artifacts):
    """Relocate immutable handoff paths without changing case membership."""
    result = dict(entry)
    for key in ("protein1", "protein2", "ligand"):
        path = Path(entry[key])
        if path.is_relative_to(LEGACY_ROOT):
            path = artifacts / path.relative_to(LEGACY_ROOT)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = str(path.resolve())
    return result


def prepare(source, artifacts, limit, split, *, offset=0):
    """Take a predeclared prefix from an already leak-controlled split."""
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    entries = payload["entries"][offset:offset + limit]
    if len(entries) != limit or len({entry["case"] for entry in entries}) != limit:
        raise ValueError(f"{split}: expected {limit} unique entries")
    return {
        "entries": [relocate_entry(entry, artifacts) for entry in entries],
        "rejected": [],
        "metadata": {
            "protocol": "internal_pose_ranker_labels_v1",
            "split": split,
            "selection": (
                f"entries_{offset + 1}_through_{offset + limit}_before_pose_label_generation"
            ),
            "benchmark_cases": False,
            "walkers_per_case": 8,
            "label": "DeepTernary full-atom DockQ",
            "features": "final-step invariant TriFuse rank_features",
        },
    }


def prepare_pdb_disjoint(source, artifacts, limit, split, exclude_pdb_ids):
    """Select the first label-blind entries with new, unique PDB identifiers."""
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    excluded = {value.upper() for value in exclude_pdb_ids}
    selected = []
    selected_pdb_ids = set()
    for entry in payload["entries"]:
        pdb_id = entry["case"].split("_", 1)[0].upper()
        if pdb_id in excluded or pdb_id in selected_pdb_ids:
            continue
        selected.append(entry)
        selected_pdb_ids.add(pdb_id)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"{split}: expected {limit} PDB-disjoint entries")
    return {
        "entries": [relocate_entry(entry, artifacts) for entry in selected],
        "rejected": [],
        "metadata": {
            "protocol": "internal_pose_ranker_labels_v1",
            "split": split,
            "selection": (
                f"first_{limit}_PDB_disjoint_entries_before_pose_label_generation"
            ),
            "excluded_pdb_ids": sorted(excluded),
            "benchmark_cases": False,
            "walkers_per_case": 8,
            "label": "DeepTernary full-atom DockQ",
            "features": "final-step invariant TriFuse rank_features",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("results/handoff_artifacts"))
    parser.add_argument("--source-dir", type=Path, default=Path(
        "results/handoff_artifacts/results/overnight_root_cause_v3_plan/indexes"))
    parser.add_argument("--out", type=Path, default=Path("results/ranker_pose_data/indexes"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = prepare(
        args.source_dir / "ternary_train_ccd.json",
        args.artifacts.resolve(), 32, "train",
    )
    train_pdb_ids = {entry["case"].split("_", 1)[0].upper()
                     for entry in train["entries"]}
    validation_source = args.source_dir / "ternary_val_ccd.json"
    val = prepare_pdb_disjoint(
        validation_source, args.artifacts.resolve(), 8, "val", train_pdb_ids,
    )
    val_pdb_ids = {entry["case"].split("_", 1)[0].upper()
                   for entry in val["entries"]}
    test = prepare_pdb_disjoint(
        validation_source, args.artifacts.resolve(), 10, "test",
        train_pdb_ids | val_pdb_ids,
    )
    for split, payload in (("train", train), ("val", val), ("test", test)):
        destination = args.out / f"ranker_{split}_v1.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"{split}: {len(payload['entries'])} cases -> {destination}")


if __name__ == "__main__":
    main()
