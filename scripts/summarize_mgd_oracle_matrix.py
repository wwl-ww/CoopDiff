#!/usr/bin/env python
"""Summarize the fixed MGD rigid-oracle causal matrix."""

import argparse
import json
import statistics
from pathlib import Path


MODES = ("none", "translations", "rotations", "all")


def summarize(root, seeds=range(8)):
    """Load complete paired reports and summarize DockQ by intervention."""
    values = {}
    for mode in MODES:
        for seed in seeds:
            path = Path(root) / f"mode_{mode}/seed_{seed}/report.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise ValueError(f"incomplete report: {path}")
            if payload.get("oracle_rigid") != mode or payload.get("seed") != seed:
                raise ValueError(f"oracle/seed mismatch: {path}")
            for row in payload["baseline"]:
                values[(mode, seed, row["case"])] = float(row["dockq"])
    baseline_keys = sorted(
        (seed, case) for mode, seed, case in values if mode == "none"
    )
    output = {"paired_rows": len(baseline_keys), "modes": {}}
    for mode in MODES:
        scores = [values[(mode, seed, case)] for seed, case in baseline_keys]
        deltas = [
            values[(mode, seed, case)] - values[("none", seed, case)]
            for seed, case in baseline_keys
        ]
        output["modes"][mode] = {
            "mean_dockq": statistics.mean(scores),
            "median_dockq": statistics.median(scores),
            "dockq_0_23_hits": sum(score >= 0.23 for score in scores),
            "mean_delta_vs_none": statistics.mean(deltas),
            "median_delta_vs_none": statistics.median(deltas),
            "wins_vs_none": sum(delta > 0 for delta in deltas),
        }
    output["interpretation"] = {
        "translations_dominant": (
            output["modes"]["translations"]["mean_delta_vs_none"]
            > output["modes"]["rotations"]["mean_delta_vs_none"]
        ),
        "joint_rigid_extra_over_best_component": (
            output["modes"]["all"]["mean_delta_vs_none"]
            - max(output["modes"]["translations"]["mean_delta_vs_none"],
                  output["modes"]["rotations"]["mean_delta_vs_none"])
        ),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
