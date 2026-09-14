"""Aggregate full-pool refinement reports without regenerating candidates."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np


AUDIT_PROTOCOL = "full_pool_refinement_audit_v1"
GEOMETRY_GATE = {
    "version": "proposal_geometry_gate_v1",
    "protein_ca_pairs_below_3A": 0,
    "e3_ca_ligand_distance": [1.5, 8.0],
    "poi_ca_ligand_distance": [1.5, 8.0],
    "protein_ca_distance": [3.0, 15.0],
}
STAGES = {
    "initial": "initial_dockq",
    "final": "refined_dockq",
    "union": "safe_dockq",
}
HARD_SAFETY_CAUSES = {
    "protein_protein_clash_below_3A",
    "e3_ligand_below_1.5A",
    "poi_ligand_below_1.5A",
}


def metric_summary(values):
    """Return benchmark summaries for a non-empty sequence of DockQ values."""
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0, "mean": None, "median": None,
            "hits_0.23": 0, "hits_0.49": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "hits_0.23": sum(value >= 0.23 for value in values),
        "hits_0.49": sum(value >= 0.49 for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def numeric_summary(values):
    """Return ordinary numeric summaries without DockQ success thresholds."""
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def paired_summary(left, right):
    """Summarize paired ``left - right`` differences."""
    if len(left) != len(right) or not left:
        raise ValueError("paired summaries require equal non-empty inputs")
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    tolerance = 1e-12
    return {
        "count": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
    }


def geometry_gate_violations(geometry):
    """Return all clauses of the production proposal geometry gate that fail."""
    pairs = int(geometry["protein_ca_pairs_below_3A"])
    protein = float(geometry["minimum_protein_ca_distance"])
    e3_ligand = float(geometry["minimum_e3_ca_ligand_distance"])
    poi_ligand = float(geometry["minimum_poi_ca_ligand_distance"])
    violations = []
    if pairs != 0 or protein < 3.0:
        # The pair count and minimum-distance clauses are logically redundant
        # for exact cdist summaries. Report one physical rejection cause while
        # still checking both stored values against the production gate.
        violations.append("protein_protein_clash_below_3A")
    if e3_ligand < 1.5:
        violations.append("e3_ligand_below_1.5A")
    elif e3_ligand > 8.0:
        violations.append("e3_ligand_above_8A")
    if poi_ligand < 1.5:
        violations.append("poi_ligand_below_1.5A")
    elif poi_ligand > 8.0:
        violations.append("poi_ligand_above_8A")
    if protein > 15.0:
        violations.append("protein_separation_above_15A")
    return violations


def hard_safety_violations(geometry):
    """Return only steric-overlap failures, leaving separation as soft evidence."""
    return [
        violation for violation in geometry_gate_violations(geometry)
        if violation in HARD_SAFETY_CAUSES
    ]


def geometry_from_coordinates(e3_ca, poi_ca, ligand):
    """Compute the gate geometry from native or proposed Cartesian coordinates."""
    arrays = [np.asarray(value, dtype=float) for value in (e3_ca, poi_ca, ligand)]
    if any(value.ndim != 2 or value.shape[1] != 3 or not len(value)
           or not np.isfinite(value).all() for value in arrays):
        raise ValueError("geometry coordinates must be finite non-empty [N,3] arrays")
    e3_ca, poi_ca, ligand = arrays
    protein = np.linalg.norm(e3_ca[:, None] - poi_ca[None], axis=-1)
    e3_ligand = np.linalg.norm(e3_ca[:, None] - ligand[None], axis=-1)
    poi_ligand = np.linalg.norm(poi_ca[:, None] - ligand[None], axis=-1)
    return {
        "protein_ca_pairs_below_3A": int((protein < 3.0).sum()),
        "minimum_protein_ca_distance": float(protein.min()),
        "minimum_e3_ca_ligand_distance": float(e3_ligand.min()),
        "minimum_poi_ca_ligand_distance": float(poi_ligand.min()),
    }


def audit_native_geometry(index_path, native_root, *, project_root="."):
    """Apply the proposal gate to native complexes as a threshold sanity check."""
    from .evaluate import _parse_native_complex

    project_root = Path(project_root).resolve()
    index_path = Path(index_path)
    if not index_path.is_absolute():
        index_path = project_root / index_path
    native_root = Path(native_root)
    if not native_root.is_absolute():
        native_root = project_root / native_root
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    rows = []
    for entry in entries:
        value = entry.get("complex")
        path = Path(value) if value else native_root / entry["case"] / "gt_complex.pdb"
        if not path.is_absolute():
            path = project_root / path
        parsed = _parse_native_complex(str(path))
        if parsed is None or not len(parsed["e3_ca"]):
            raise ValueError(f"could not parse complete native ternary complex: {path}")
        geometry = geometry_from_coordinates(
            parsed["e3_ca"], parsed["poi_ca"], parsed["ligand"],
        )
        rows.append({
            "case": entry["case"],
            "path": str(path.resolve()),
            "geometry": geometry,
            "violations": geometry_gate_violations(geometry),
        })
    counts = Counter(
        violation for row in rows for violation in row["violations"]
    )
    return {
        "cases": len(rows),
        "acceptable": sum(not row["violations"] for row in rows),
        "hard_safety_acceptable": sum(
            not hard_safety_violations(row["geometry"]) for row in rows
        ),
        "violation_counts_nonexclusive": dict(counts.most_common()),
        "rejected_cases": [row for row in rows if row["violations"]],
    }


def discover_report_directories(results_root):
    """Discover only the completed full94 shard directories under a result root."""
    root = Path(results_root).resolve()
    directories = sorted(
        path for path in root.glob("full94_v*_gpu*") if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(f"no full94 shard directories under {root}")
    return directories


def load_reports(input_dirs):
    """Load complete per-case reports and reject duplicate cases."""
    reports = {}
    sources = {}
    failure_records = []
    for value in input_dirs:
        directory = Path(value).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        failure_path = directory / "failures.json"
        if failure_path.is_file():
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            failure_records.extend(payload.get("failures", []))
        for path in sorted(directory.glob("*.json")):
            if path.name in {"manifest.json", "failures.json", "batch_status.json"}:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "complete":
                continue
            case = report["case"]
            if case in reports:
                raise ValueError(
                    f"duplicate case {case}: {sources[case]} and {path}"
                )
            reports[case] = report
            sources[case] = str(path)
    if not reports:
        raise RuntimeError("no complete case reports found")
    return reports, sources, failure_records


def validate_reports(reports, *, expected_cases=None, expected_pool=None,
                     expected_bitseed=None, expected_sdpi=None):
    """Validate pool completeness, candidate identity, and stored gate labels."""
    if expected_cases is not None and len(reports) != expected_cases:
        raise ValueError(f"found {len(reports)} cases, expected {expected_cases}")
    protocols = set()
    contracts = set()
    total = Counter()
    for case, report in reports.items():
        rows = report["refinements"]
        if report.get("union_evaluated_count") != len(rows):
            raise ValueError(f"{case}: evaluated count does not match rows")
        if expected_pool is not None and len(rows) != expected_pool:
            raise ValueError(f"{case}: found {len(rows)} candidates, expected {expected_pool}")
        identifiers = [repr(row["candidate_id"]) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{case}: duplicate candidate ids")
        if not all(row.get("valid") for row in rows):
            raise ValueError(f"{case}: contains invalid refinement rows")
        counts = Counter(row["source"] for row in rows)
        if expected_bitseed is not None and counts["bitseed"] != expected_bitseed:
            raise ValueError(
                f"{case}: found {counts['bitseed']} BiTSeed, expected {expected_bitseed}"
            )
        if expected_sdpi is not None and counts["sdpi"] != expected_sdpi:
            raise ValueError(
                f"{case}: found {counts['sdpi']} SDPI, expected {expected_sdpi}"
            )
        for row in rows:
            violations = geometry_gate_violations(row["geometry"])
            if bool(row["acceptable"]) != (not violations):
                raise ValueError(
                    f"{case}/{row['candidate_id']}: stored geometry gate mismatch"
                )
            total[row["source"]] += 1
        protocols.add(report["config"]["protocol"])
        config = report["config"]
        contract = {
            key: config.get(key) for key in (
                "protocol", "index_sha256", "checkpoint_sha256",
                "candidate_pool", "steps", "proposal_seed",
                "refine_full_pool", "shared_trajectory_policy",
            )
        }
        contracts.add(json.dumps(contract, sort_keys=True))
    if len(contracts) != 1:
        raise ValueError("input reports use different evaluation contracts")
    return {
        "cases": len(reports),
        "candidates": sum(total.values()),
        "source_counts": dict(sorted(total.items())),
        "all_refinements_valid": True,
        "stored_geometry_labels_reproduced": True,
        "input_protocols": sorted(protocols),
        "evaluation_contract": json.loads(next(iter(contracts))),
    }


def _case_best(reports, selected_ids, key):
    by_id = {
        repr(row["candidate_id"]): row for row in reports["refinements"]
    }
    selected = [by_id[repr(candidate_id)] for candidate_id in selected_ids]
    return max(selected, key=lambda row: row[key])


def summarize_leaderboards(reports):
    """Summarize per-case best Initial, Final, and Union for every arm."""
    ordered = [reports[case] for case in sorted(reports)]
    arm_names = list(ordered[0]["arms"])
    if any(set(report["arms"]) != set(arm_names) for report in ordered):
        raise ValueError("arm names differ across cases")
    result = {}
    for arm in arm_names:
        stages = {}
        for label in STAGES:
            values = [report["arms"][arm][
                {"initial": "initial", "final": "refined", "union": "safe_union"}[label]
            ]["best"] for report in ordered]
            stages[label] = metric_summary(values)
        initial = [report["arms"][arm]["initial"]["best"] for report in ordered]
        final = [report["arms"][arm]["refined"]["best"] for report in ordered]
        stages["final_minus_initial"] = paired_summary(final, initial)
        result[arm] = stages
    baseline = result.get("sdpi_64")
    if baseline is not None:
        for arm, stages in result.items():
            if arm == "sdpi_64":
                continue
            stages["vs_sdpi"] = {}
            for label in STAGES:
                left = [report["arms"][arm][
                    {"initial": "initial", "final": "refined", "union": "safe_union"}[label]
                ]["best"] for report in ordered]
                right = [report["arms"]["sdpi_64"][
                    {"initial": "initial", "final": "refined", "union": "safe_union"}[label]
                ]["best"] for report in ordered]
                stages["vs_sdpi"][label] = paired_summary(left, right)
    return result


def summarize_sources(reports):
    """Report both candidate-level and per-case-best source behavior."""
    result = {}
    for source in ("bitseed", "sdpi"):
        rows = [
            row for report in reports.values() for row in report["refinements"]
            if row["source"] == source
        ]
        initial = [row["initial_dockq"] for row in rows]
        final = [row["refined_dockq"] for row in rows]
        case_initial, case_final, case_union = [], [], []
        for report in reports.values():
            selected = [row for row in report["refinements"] if row["source"] == source]
            case_initial.append(max(row["initial_dockq"] for row in selected))
            case_final.append(max(row["refined_dockq"] for row in selected))
            case_union.append(max(row["safe_dockq"] for row in selected))
        result[source] = {
            "candidate_level": {
                "initial": metric_summary(initial),
                "final": metric_summary(final),
                "final_minus_initial": paired_summary(final, initial),
            },
            "case_best": {
                "initial": metric_summary(case_initial),
                "final": metric_summary(case_final),
                "union": metric_summary(case_union),
                "final_minus_initial": paired_summary(case_final, case_initial),
            },
        }
    return result


def summarize_retention(reports):
    """Measure how often each selected arm preserves the full-pool oracle."""
    result = {}
    for arm in reports[next(iter(reports))]["arms"]:
        if arm == "full_pool_256":
            continue
        stage_result = {}
        for label, key in STAGES.items():
            ratios, exact = [], 0
            for report in reports.values():
                full = max(report["refinements"], key=lambda row: row[key])
                selected = {
                    repr(value) for value in report["arms"][arm]["selected_candidate_ids"]
                }
                chosen = _case_best(
                    report, report["arms"][arm]["selected_candidate_ids"], key,
                )
                exact += repr(full["candidate_id"]) in selected
                ratios.append(
                    chosen[key] / full[key] if full[key] > 0 else 1.0
                )
            stage_result[label] = {
                "mean_oracle_retention_ratio": statistics.fmean(ratios),
                "median_oracle_retention_ratio": statistics.median(ratios),
                "exact_oracle_candidate_retained": exact,
                "cases": len(ratios),
            }
        result[arm] = stage_result
    return result


def summarize_acceptability(reports, *, high_quality_cutoff=0.23):
    """Audit non-exclusive gate failures and their relationship to DockQ."""
    all_rows = [row for report in reports.values() for row in report["refinements"]]
    violations = Counter()
    patterns = Counter()
    for row in all_rows:
        failed = geometry_gate_violations(row["geometry"])
        violations.update(failed)
        patterns[" + ".join(failed) if failed else "acceptable"] += 1

    def group(rows):
        acceptable = [row for row in rows if row["acceptable"]]
        rejected = [row for row in rows if not row["acceptable"]]
        return {
            "total": len(rows),
            "acceptable": len(acceptable),
            "acceptable_rate": len(acceptable) / len(rows) if rows else None,
            "initial_dockq": {
                "acceptable": metric_summary(row["initial_dockq"] for row in acceptable),
                "rejected": metric_summary(row["initial_dockq"] for row in rejected),
            },
            "final_dockq": {
                "acceptable": metric_summary(row["refined_dockq"] for row in acceptable),
                "rejected": metric_summary(row["refined_dockq"] for row in rejected),
            },
        }

    oracle = {}
    for label, key in STAGES.items():
        winners = [
            max(report["refinements"], key=lambda row: row[key])
            for report in reports.values()
        ]
        oracle[label] = {
            "acceptable_winners": sum(row["acceptable"] for row in winners),
            "cases": len(winners),
        }
    high_quality = [row for row in all_rows if row["initial_dockq"] >= high_quality_cutoff]
    high_quality_violations = Counter(
        violation
        for row in high_quality
        for violation in geometry_gate_violations(row["geometry"])
    )
    by_source_violations = {}
    for source in ("bitseed", "sdpi"):
        source_rows = [row for row in all_rows if row["source"] == source]
        counts = Counter(
            violation
            for row in source_rows
            for violation in geometry_gate_violations(row["geometry"])
        )
        by_source_violations[source] = dict(counts.most_common())

    def hard_summary(rows):
        retained = [
            row for row in rows if not hard_safety_violations(row["geometry"])
        ]
        high = [row for row in rows if row["initial_dockq"] >= high_quality_cutoff]
        return {
            "total": len(rows),
            "retained": len(retained),
            "retained_rate": len(retained) / len(rows),
            "high_quality_total": len(high),
            "high_quality_retained": sum(
                not hard_safety_violations(row["geometry"]) for row in high
            ),
        }

    arm_audit = {}
    for arm in reports[next(iter(reports))]["arms"]:
        selected_count = acceptable_count = covered_cases = 0
        selected_sources = Counter()
        winner_acceptable = Counter()
        acceptable_best = {label: [] for label in STAGES}
        for report in reports.values():
            by_id = {
                repr(row["candidate_id"]): row for row in report["refinements"]
            }
            selected = [
                by_id[repr(candidate_id)]
                for candidate_id in report["arms"][arm]["selected_candidate_ids"]
            ]
            acceptable = [row for row in selected if row["acceptable"]]
            selected_count += len(selected)
            selected_sources.update(row["source"] for row in selected)
            acceptable_count += len(acceptable)
            covered_cases += bool(acceptable)
            for label, key in STAGES.items():
                winner_acceptable[label] += max(
                    selected, key=lambda row: row[key]
                )["acceptable"]
                if acceptable:
                    acceptable_best[label].append(max(row[key] for row in acceptable))
        arm_audit[arm] = {
            "selected_candidates": selected_count,
            "selected_source_counts": dict(sorted(selected_sources.items())),
            "acceptable_selected": acceptable_count,
            "acceptable_rate": acceptable_count / selected_count,
            "cases_with_acceptable_candidate": covered_cases,
            "best_candidate_acceptable": dict(winner_acceptable),
            "acceptable_constrained_best": {
                label: metric_summary(values)
                for label, values in acceptable_best.items()
            },
        }
    return {
        "gate": GEOMETRY_GATE,
        "overall": group(all_rows),
        "by_source": {
            source: group([row for row in all_rows if row["source"] == source])
            for source in ("bitseed", "sdpi")
        },
        "violation_counts_nonexclusive": dict(violations.most_common()),
        "violation_counts_by_source": by_source_violations,
        "violation_patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in patterns.most_common()
        ],
        "hard_safety_only_diagnostic": {
            "definition": sorted(HARD_SAFETY_CAUSES),
            "overall": hard_summary(all_rows),
            "by_source": {
                source: hard_summary([
                    row for row in all_rows if row["source"] == source
                ])
                for source in ("bitseed", "sdpi")
            },
            "production_selector_changed": False,
        },
        "high_quality_initial": {
            "cutoff": high_quality_cutoff,
            "total": len(high_quality),
            "acceptable": sum(row["acceptable"] for row in high_quality),
            "violation_counts_nonexclusive": dict(
                high_quality_violations.most_common()
            ),
        },
        "full_pool_oracle": oracle,
        "arms": arm_audit,
    }


def aggregate_reports(reports, *, integrity=None, failures=None):
    """Build the complete serializable audit payload."""
    timing = [float(report["timing_seconds"]["total"]) for report in reports.values()]
    return {
        "protocol": AUDIT_PROTOCOL,
        "integrity": integrity or {},
        "recorded_failures": failures or [],
        "timing_seconds": {
            "aggregate": sum(timing),
            "per_case": numeric_summary(timing),
        },
        "leaderboards": summarize_leaderboards(reports),
        "sources": summarize_sources(reports),
        "selector_retention": summarize_retention(reports),
        "acceptability": summarize_acceptability(reports),
    }
