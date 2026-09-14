#!/usr/bin/env python
"""Merge full-pool shards and audit refinement, selectors, and geometry gates."""

import argparse
import json
from pathlib import Path

from tricoopdiff.refinement_audit import (
    aggregate_reports,
    audit_native_geometry,
    discover_report_directories,
    load_reports,
    validate_reports,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-root")
    source.add_argument("--input-dir", action="append")
    parser.add_argument("--expected-cases", type=int, default=94)
    parser.add_argument("--expected-pool", type=int, default=256)
    parser.add_argument("--expected-bitseed", type=int, default=192)
    parser.add_argument("--expected-sdpi", type=int, default=64)
    parser.add_argument("--index", help="Optional benchmark index for native gate audit")
    parser.add_argument(
        "--native-root", default="DeepTernary/data/TernaryDB/pdbs",
        help="Native complex root used with --index",
    )
    parser.add_argument("--out")
    return parser.parse_args()


def main():
    args = parse_args()
    directories = (
        discover_report_directories(args.results_root)
        if args.results_root else [Path(value).resolve() for value in args.input_dir]
    )
    reports, paths, failures = load_reports(directories)
    if failures:
        raise RuntimeError(f"input shards record {len(failures)} failed cases")
    integrity = validate_reports(
        reports, expected_cases=args.expected_cases,
        expected_pool=args.expected_pool,
        expected_bitseed=args.expected_bitseed,
        expected_sdpi=args.expected_sdpi,
    )
    integrity["input_directories"] = [str(path) for path in directories]
    integrity["case_files"] = dict(sorted(paths.items()))
    payload = aggregate_reports(
        reports, integrity=integrity, failures=failures,
    )
    if args.index:
        payload["acceptability"]["native_reference"] = audit_native_geometry(
            args.index, args.native_root,
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    print(rendered, end="")


if __name__ == "__main__":
    main()
