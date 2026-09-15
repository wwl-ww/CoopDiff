#!/usr/bin/env python
"""Recompute BiTSeed refinement mismatch statistics from full-pool reports."""

import argparse
import json
import statistics
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results", required=True, type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--expected-cases", type=int, default=94)
    result.add_argument("--expected-candidates-per-case", type=int, default=256)
    result.add_argument("--examples", type=int, default=10)
    result.add_argument("--minimum-initial-dockq", type=float, default=0.30)
    return result


def _reports(root, expected_candidates):
    by_case = {}
    for path in sorted(root.glob("full94_*/*.json")):
        if path.name in {"manifest.json", "failures.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        rows = payload.get("refinements", [])
        if (payload.get("union_evaluated_count") != expected_candidates
                or len(rows) != expected_candidates):
            continue
        case = payload["case"]
        if case in by_case:
            raise ValueError(f"duplicate complete report for {case}")
        by_case[case] = (path, payload)
    return by_case


def _source_summary(reports, source):
    rows = []
    initial_oracles, final_oracles = [], []
    for _, payload in reports.values():
        selected = [row for row in payload["refinements"]
                    if row["source"] == source and row.get("valid")]
        if not selected:
            raise ValueError(f"{payload['case']} has no valid {source} candidates")
        rows.extend(selected)
        initial_oracles.append(max(row["initial_dockq"] for row in selected))
        final_oracles.append(max(row["refined_dockq"] for row in selected))
    deltas = [row["refined_dockq"] - row["initial_dockq"] for row in rows]
    return {
        "candidate_count": len(rows),
        "candidate_improved_count": sum(delta > 0 for delta in deltas),
        "candidate_improved_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
        "candidate_mean_delta_dockq": statistics.fmean(deltas),
        "case_initial_oracle_mean_dockq": statistics.fmean(initial_oracles),
        "case_final_oracle_mean_dockq": statistics.fmean(final_oracles),
        "case_oracle_delta_dockq": statistics.fmean(final_oracles) - statistics.fmean(initial_oracles),
    }


def _regression_examples(reports, minimum, count):
    candidates = []
    for path, payload in reports.values():
        config = payload["config"]
        for row in payload["refinements"]:
            if (row["source"] != "bitseed" or not row.get("valid")
                    or row["initial_dockq"] < minimum):
                continue
            expected_seed = (int(config["proposal_seed"]) + 2_000_000
                             + int(row["pool_index"])) % (2**31 - 1)
            if row["diffusion_seed"] != expected_seed:
                raise ValueError(f"unexpected diffusion seed in {payload['case']}")
            candidates.append({
                "case": payload["case"],
                "candidate_id": row["candidate_id"],
                "pool_index": row["pool_index"],
                "proposal_seed": config["proposal_seed"],
                "diffusion_seed": row["diffusion_seed"],
                "t_start": 1.0,
                "steps": config["steps"],
                "acceptable": bool(row["acceptable"]),
                "initial_dockq": row["initial_dockq"],
                "refined_dockq": row["refined_dockq"],
                "delta_dockq": row["refined_dockq"] - row["initial_dockq"],
                "checkpoint": config["checkpoint"],
                "checkpoint_sha256": config["checkpoint_sha256"],
                "source_report": str(path.resolve()),
            })
    candidates.sort(key=lambda row: (row["delta_dockq"], row["case"],
                                     repr(row["candidate_id"])))
    selected, seen = [], set()
    # Distinct cases make the compact replay panel less sensitive to one target.
    for row in candidates:
        if row["case"] in seen:
            continue
        selected.append(row)
        seen.add(row["case"])
        if len(selected) == count:
            break
    return selected


def build_report(args):
    reports = _reports(args.results.resolve(), args.expected_candidates_per_case)
    if len(reports) != args.expected_cases:
        raise ValueError(f"found {len(reports)} complete cases, expected {args.expected_cases}")
    return {
        "protocol": "bitseed_refinement_mismatch_audit_v1",
        "results_root": str(args.results.resolve()),
        "case_count": len(reports),
        "candidate_count": sum(len(payload["refinements"])
                               for _, payload in reports.values()),
        "sources": {
            source: _source_summary(reports, source)
            for source in ("sdpi", "bitseed")
        },
        "deterministic_regression_examples": _regression_examples(
            reports, args.minimum_initial_dockq, args.examples),
        "scope": (
            "Native-reading audit only; examples demonstrate regression but do not "
            "estimate deployable ranked performance or establish a unique cause."
        ),
    }


def main():
    args = parser().parse_args()
    report = build_report(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
