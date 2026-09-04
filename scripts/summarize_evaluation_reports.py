#!/usr/bin/env python
"""Combine TriCoopDiff evaluation JSON files into a comparison table."""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


METRICS = (
    "dockq_best",
    "dockq_mean",
    "dockq_median",
    "dockq_fnat_best",
    "dockq_lrms_best",
    "dockq_irms_best",
    "protac_rmsd_best",
    "protac_rmsd_mean",
    "protac_pocket_dist_best",
    "protac_pocket_dist_mean",
    "e3_interface_dockq_best",
    "e3_interface_dockq_mean",
    "poi_dockq_best",
    "poi_dockq_mean",
    "poi_com_dist_best",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", help="Evaluation JSON files")
    parser.add_argument("--reports-dir", default=None,
                        help="Also read every JSON directly under this directory")
    parser.add_argument("--require-trained", action="store_true",
                        help="Reject reports without a trained checkpoint")
    parser.add_argument("--require-exact-dockq", action="store_true",
                        help="Reject reports lacking DeepTernary full-atom DockQ")
    parser.add_argument("--out-json", default="results/evaluation_summary.json")
    parser.add_argument("--out-csv", default="results/evaluation_summary.csv")
    return parser


def load_row(path):
    with path.open() as handle:
        report = json.load(handle)
    structure = report.get("structure", {})
    ensemble = report.get("ensemble", {})
    conformational = ensemble.get("conformational_analysis", {})
    config = report.get("run_config", {})
    checkpoint = config.get("tribrain_checkpoint")
    mode = "trained" if checkpoint else "zero_film"
    if config.get("freeze_poi"):
        mode += "_freeze_poi"
    row = {
        "report": str(path),
        "case": report.get("case"),
        "mode": mode,
        "checkpoint": checkpoint,
        "seed": config.get("seed"),
        "n_walkers": config.get("n_walkers"),
        "steps": config.get("steps"),
        "n_valid_walkers": ensemble.get("n_valid_walkers"),
        "n_nan_walkers": ensemble.get("n_nan_walkers"),
        "selection_metric": ensemble.get("selection_metric"),
        "exact_dockq": structure.get("dockq_best") is not None,
        "ensemble_n_clusters": conformational.get("n_clusters"),
        "ensemble_dominant_fraction": conformational.get("dominant_fraction"),
        "ensemble_mean_pairwise_rmsd": conformational.get("mean_pairwise_rmsd"),
    }
    row.update({metric: structure.get(metric) for metric in METRICS})
    return row


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_rows(rows):
    """Aggregate valid numeric metrics and list failed cases."""
    failed = [row["case"] for row in rows if not row.get("n_valid_walkers")]
    requested = sum(
        (row.get("n_valid_walkers") or 0) + (row.get("n_nan_walkers") or 0)
        for row in rows
    )
    valid = sum(row.get("n_valid_walkers") or 0 for row in rows)
    metrics = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows
                  if isinstance(row.get(metric), (int, float))
                  and math.isfinite(row[metric])]
        if values:
            metrics[metric] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p25": _percentile(values, 0.25),
                "p75": _percentile(values, 0.75),
            }
    return {
        "n_cases": len(rows),
        "n_failed_cases": len(failed),
        "failed_cases": failed,
        "n_requested_walkers": requested,
        "n_valid_walkers": valid,
        "valid_walker_rate": valid / requested if requested else 0.0,
        "n_exact_dockq_cases": sum(bool(row.get("exact_dockq")) for row in rows),
        "metrics": metrics,
    }


def main():
    args = build_parser().parse_args()
    paths = [Path(value).resolve() for value in args.reports]
    if args.reports_dir:
        paths.extend(
            path for path in sorted(Path(args.reports_dir).resolve().glob("*.json"))
            if path.name not in {"batch_status.json", "evaluation_manifest.json"}
        )
    paths = list(dict.fromkeys(paths))
    if not paths:
        print("FATAL: provide reports or --reports-dir", file=sys.stderr)
        sys.exit(2)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"FATAL: missing reports: {missing}", file=sys.stderr)
        sys.exit(2)
    try:
        rows = [load_row(path) for path in paths]
    except (OSError, json.JSONDecodeError) as error:
        print(f"FATAL: invalid evaluation report: {error}", file=sys.stderr)
        sys.exit(2)
    if args.require_trained and any(not row["checkpoint"] for row in rows):
        print("FATAL: zero-init report found in trained evaluation", file=sys.stderr)
        sys.exit(2)
    if args.require_exact_dockq and any(not row["exact_dockq"] for row in rows):
        print("FATAL: report without DeepTernary full-atom DockQ found",
              file=sys.stderr)
        sys.exit(2)

    summary = summarize_rows(rows)

    output_json = Path(args.out_json)
    output_csv = Path(args.out_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump({"summary": summary, "runs": rows}, handle, indent=2)
    fieldnames = list(rows[0])
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
