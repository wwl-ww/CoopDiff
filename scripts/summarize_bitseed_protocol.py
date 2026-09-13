#!/usr/bin/env python
"""Summarize fixed-budget BiTSeed proposal and selector audits."""

import argparse
import json
import statistics
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sdpi", required=True)
    result.add_argument("--hybrid", action="append", default=[])
    result.add_argument("--selector")
    result.add_argument("--selector-suite")
    result.add_argument("--allow-incomplete", action="store_true")
    result.add_argument("--out")
    return result


def load_directory(value, allow_incomplete):
    directory = Path(value).resolve()
    status_path = directory / "batch_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else None
    if not allow_incomplete and (status is None or not status.get("complete")):
        raise RuntimeError(f"incomplete experiment directory: {directory}")
    cases = {}
    for path in directory.glob("*.json"):
        if path.name in {"manifest.json", "batch_status.json"}:
            continue
        report = json.loads(path.read_text())
        if report.get("status") == "complete":
            cases[report["case"]] = report
    return directory, status, cases


def metric_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "hits_0.23": sum(value >= 0.23 for value in values),
        "hits_0.49": sum(value >= 0.49 for value in values),
    }


def proposal_summary(cases, common, baseline=None):
    values = [cases[case]["oracle_candidate"]["dockq"] for case in common]
    result = metric_summary(values)
    result.update({
        "acceptable_candidates": sum(cases[case]["n_acceptable"] for case in common),
        "fallback_events": sum(
            len(cases[case].get("fallback_reason") or []) for case in common
        ),
    })
    if baseline is not None:
        deltas = [
            cases[case]["oracle_candidate"]["dockq"]
            - baseline[case]["oracle_candidate"]["dockq"]
            for case in common
        ]
        tolerance = 1e-12
        result["paired_vs_sdpi"] = {
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "wins": sum(delta > tolerance for delta in deltas),
            "ties": sum(abs(delta) <= tolerance for delta in deltas),
            "losses": sum(delta < -tolerance for delta in deltas),
        }
    return result


def selector_summary(cases, common):
    result = {}
    for walker_count in (32, 64):
        key = str(walker_count)
        selections = [cases[case]["native_free_selections"][key] for case in common]
        selected = [item["selected_oracle_candidate"]["dockq"] for item in selections]
        result[key] = {
            **metric_summary(selected),
            "mean_oracle_retention_ratio": statistics.fmean(
                item["oracle_retention_ratio"] for item in selections
            ),
            "full_oracle_retained": sum(
                item["full_oracle_retained"] for item in selections
            ),
            "mean_cluster_count": statistics.fmean(
                item["cluster_count"] for item in selections
            ),
            "mean_unique_clusters_selected": statistics.fmean(
                item["unique_clusters_selected"] for item in selections
            ),
            "source_counts": {
                source: sum(
                    item["actual_source_counts"][source] for item in selections
                )
                for source in ("bitseed", "sdpi")
            },
        }
    return result


def selector_suite_summary(cases, common):
    result = {}
    for walker_count in ("32", "64"):
        names = sorted(cases[common[0]]["selector_audits"][walker_count])
        result[walker_count] = {}
        for name in names:
            audits = [
                cases[case]["selector_audits"][walker_count][name]
                for case in common
            ]
            selected = [
                audit["selected_oracle_candidate"]["dockq"] for audit in audits
            ]
            result[walker_count][name] = {
                **metric_summary(selected),
                "mean_oracle_retention_ratio": statistics.fmean(
                    audit["oracle_retention_ratio"] for audit in audits
                ),
                "full_oracle_retained": sum(
                    audit["full_oracle_retained"] for audit in audits
                ),
                "mean_unique_clusters_selected": statistics.fmean(
                    audit["unique_clusters_selected"] for audit in audits
                ),
                "mean_bitseed_selected": statistics.fmean(
                    audit["actual_source_counts"]["bitseed"] for audit in audits
                ),
                "mean_sdpi_selected": statistics.fmean(
                    audit["actual_source_counts"]["sdpi"] for audit in audits
                ),
            }
    return result


def main():
    args = parser().parse_args()
    sdpi_dir, sdpi_status, sdpi = load_directory(args.sdpi, args.allow_incomplete)
    experiments = [("sdpi", sdpi_dir, sdpi_status, sdpi)]
    for value in args.hybrid:
        directory, status, cases = load_directory(value, args.allow_incomplete)
        experiments.append((directory.name, directory, status, cases))
    selector = None
    auxiliary = []
    if args.selector:
        directory, status, cases = load_directory(args.selector, args.allow_incomplete)
        selector = (directory.name, directory, status, cases)
        auxiliary.append(cases)
    suite = None
    if args.selector_suite:
        directory, status, cases = load_directory(
            args.selector_suite, args.allow_incomplete,
        )
        suite = (directory.name, directory, status, cases)
        auxiliary.append(cases)
    case_sets = [set(item[3]) for item in experiments]
    case_sets.extend(set(cases) for cases in auxiliary)
    common = sorted(set.intersection(*case_sets))
    if not common:
        raise RuntimeError("experiment directories have no common completed cases")
    payload = {
        "scope": "complete cases shared by every supplied directory",
        "common_cases": common,
        "proposal": {
            label: proposal_summary(cases, common, None if label == "sdpi" else sdpi)
            for label, _, _, cases in experiments
        },
    }
    if selector is not None:
        payload["selector"] = selector_summary(selector[3], common)
    if suite is not None:
        payload["selector_suite"] = selector_suite_summary(suite[3], common)
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered)
    print(rendered, end="")

if __name__ == "__main__":
    main()
