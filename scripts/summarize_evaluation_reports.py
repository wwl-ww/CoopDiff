#!/usr/bin/env python
"""Combine TriCoopDiff evaluation JSON files into a comparison table."""

import argparse
import csv
import json
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
    parser.add_argument("reports", nargs="+", help="Evaluation JSON files")
    parser.add_argument("--out-json", default="results/evaluation_summary.json")
    parser.add_argument("--out-csv", default="results/evaluation_summary.csv")
    return parser


def load_row(path):
    with path.open() as handle:
        report = json.load(handle)
    structure = report.get("structure", {})
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
    }
    row.update({metric: structure.get(metric) for metric in METRICS})
    return row


def main():
    args = build_parser().parse_args()
    paths = [Path(value).resolve() for value in args.reports]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"FATAL: missing reports: {missing}", file=sys.stderr)
        sys.exit(2)
    try:
        rows = [load_row(path) for path in paths]
    except (OSError, json.JSONDecodeError) as error:
        print(f"FATAL: invalid evaluation report: {error}", file=sys.stderr)
        sys.exit(2)

    output_json = Path(args.out_json)
    output_csv = Path(args.out_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump({"runs": rows}, handle, indent=2)
    fieldnames = list(rows[0])
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": rows}, indent=2))


if __name__ == "__main__":
    main()
