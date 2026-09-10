#!/usr/bin/env python3
"""Inventory transferred root-cause results without loading model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path(
    "results/handoff_artifacts/results"
)
DEFAULT_OUTPUT = Path("results/handoff_summary")
CHECKPOINT_NAMES = {"best.pt", "latest.pt", "checkpoint.pt", "smoke_only.pt"}
IDENTIFIER_KEYS = (
    "case",
    "seed",
    "model",
    "variant",
    "policy",
    "chemistry",
    "torsion",
    "adaptation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_paths(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_paths(item, prefix + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scalar_paths(item, prefix + (str(index),))
    elif value is not None:
        yield prefix, value


def values_for_key(value: Any, wanted: str) -> list[Any]:
    return [item for path, item in scalar_paths(value) if path and path[-1] == wanted]


def unique_scalars(value: Any, wanted: str) -> list[Any]:
    values = values_for_key(value, wanted)
    return sorted(set(values), key=lambda item: (str(type(item)), str(item)))


def numeric_stats(values: Iterable[Any]) -> dict[str, Any] | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not numbers:
        return None
    return {
        "n": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def metric_records(value: Any) -> list[dict[str, Any]]:
    """Extract leaf observations carrying exact or approximate overall DockQ."""
    records: list[dict[str, Any]] = []

    def visit(node: Any, context: dict[str, Any]) -> None:
        if isinstance(node, dict):
            local = dict(context)
            for key in IDENTIFIER_KEYS:
                if key in node and isinstance(node[key], (str, int, float)):
                    local[key] = node[key]
            metrics = {
                key: node[key]
                for key in ("dockq", "approx_dockq")
                if isinstance(node.get(key), (int, float))
            }
            # A case or seed prevents aggregate summary rows from being counted as
            # observations. Context propagation handles nested walker reports.
            if metrics and ("case" in local or "seed" in local):
                records.append({**local, **metrics})
            for item in node.values():
                visit(item, local)
        elif isinstance(node, list):
            for item in node:
                visit(item, context)

    visit(value, {})
    # Parent dictionaries can contain a metric and then recurse into unrelated
    # children. Remove exact duplicate observations while preserving order.
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        marker = json.dumps(record, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            deduplicated.append(record)
    return deduplicated


def grouped_metric_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep intervention/model conditions separate within an evaluation file."""
    grouped: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        identifiers = tuple(
            (key, record[key])
            for key in ("model", "variant", "policy")
            if key in record
        )
        grouped[identifiers].append(record)
    results = []
    for identifiers, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        results.append(
            {
                "identifiers": dict(identifiers) or {"condition": "unspecified"},
                "observations": len(rows),
                "cases": sorted({row["case"] for row in rows if "case" in row}),
                "seeds": sorted({row["seed"] for row in rows if "seed" in row}),
                "exact_dockq": numeric_stats(row.get("dockq") for row in rows),
                "approximate_dockq": numeric_stats(row.get("approx_dockq") for row in rows),
            }
        )
    return results


def reported_summaries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("summary"), list):
        return []
    rows = []
    for item in value["summary"]:
        if not isinstance(item, dict):
            continue
        if not any(key in item for key in ("dockq", "approx_dockq")):
            continue
        rows.append(
            {
                key: val
                for key, val in item.items()
                if isinstance(val, (str, int, float, bool))
            }
        )
    return rows


def determinism_status(value: Any) -> dict[str, Any]:
    settings = unique_scalars(value, "deterministic_algorithms")
    if settings == [True]:
        status = "explicitly_enabled"
    elif settings == [False]:
        status = "explicitly_disabled"
    elif settings:
        status = "mixed"
    else:
        status = "unverified"
    return {
        "status": status,
        "deterministic_algorithms_values": settings,
        "limitation": (
            "Recorded seeds do not establish repeatability when deterministic "
            "algorithms are not explicitly recorded."
            if status == "unverified"
            else None
        ),
    }


def weight_view(path: Path, value: Any) -> dict[str, Any]:
    evidence = [str(path)]
    if isinstance(value, dict) and isinstance(value.get("scope"), str):
        evidence.append(value["scope"])
    text = " ".join(evidence).lower()
    if "raw_expert" in text or "raw expert" in text:
        return {"view": "raw_expert_proposals", "evidence": evidence}
    has_ema = re.search(r"\bema\b", text) is not None
    has_raw = re.search(r"\braw(?:-weight)?\b", text) is not None
    if has_ema and has_raw:
        return {"view": "mixed_raw_and_ema_mentions", "evidence": evidence}
    if has_ema:
        return {"view": "ema", "evidence": evidence}
    if has_raw:
        return {"view": "raw", "evidence": evidence}
    return {"view": "unspecified", "evidence": []}


def summarize_history(path: Path, value: Any, source: Path) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    rows = [row for row in value if isinstance(row, dict) and "train_loss" in row]
    if not rows:
        return None
    validation_keys = sorted(
        {key for row in rows for key in row if "val_loss" in key}
    )
    return {
        "experiment": path.parent.name,
        "path": str(path),
        "relative_path": str(path.relative_to(source)),
        "sha256": sha256(path),
        "epochs_recorded": len(rows),
        "epoch_first": rows[0].get("epoch"),
        "epoch_last": rows[-1].get("epoch"),
        "train_loss": {
            "first": rows[0].get("train_loss"),
            "last": rows[-1].get("train_loss"),
            "minimum": min(float(row["train_loss"]) for row in rows),
        },
        "validation_loss": {
            key: {
                "first": rows[0].get(key),
                "last": rows[-1].get(key),
                "minimum": min(float(row[key]) for row in rows if key in row),
            }
            for key in validation_keys
        },
        "note": "Loss values are training diagnostics and are not DockQ measurements.",
    }


def json_inventory(path: Path, value: Any, source: Path) -> dict[str, Any]:
    metric_keys = sorted(
        {
            parts[-1]
            for parts, _ in scalar_paths(value)
            if parts and any(token in parts[-1].lower() for token in ("dockq", "fnat", "lrms", "irms"))
        }
    )
    if isinstance(value, dict):
        top_level = sorted(value.keys())
        root_type = "object"
    elif isinstance(value, list):
        top_level = []
        root_type = "array"
    else:
        top_level = []
        root_type = type(value).__name__
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(source)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "root_type": root_type,
        "top_level_keys": top_level,
        "metric_keys": metric_keys,
        "scope": value.get("scope") if isinstance(value, dict) else None,
        "weight_view": weight_view(path, value),
        "determinism": determinism_status(value),
        "cases": unique_scalars(value, "case"),
        "seeds": unique_scalars(value, "seed"),
    }


def evaluation_summary(path: Path, value: Any, source: Path) -> dict[str, Any] | None:
    records = metric_records(value)
    metric_keys = {
        parts[-1]
        for parts, _ in scalar_paths(value)
        if parts and parts[-1] in {"dockq", "approx_dockq"}
    }
    if not metric_keys:
        return None
    if metric_keys == {"dockq", "approx_dockq"}:
        kind = "exact_and_approximate"
    elif "dockq" in metric_keys:
        kind = "exact"
    else:
        kind = "approximate"
    cases = unique_scalars(value, "case")
    scope = value.get("scope") if isinstance(value, dict) else None
    claimed_cases = None
    if isinstance(scope, str):
        match = re.search(r"\b(\d+)\s+(?:internal\s+)?(?:validation\s+)?cases\b", scope, re.I)
        claimed_cases = int(match.group(1)) if match else None
    return {
        "experiment": path.parent.name,
        "path": str(path),
        "relative_path": str(path.relative_to(source)),
        "sha256": sha256(path),
        "scope": scope,
        "scope_case_count_claim": claimed_cases,
        "scope_case_count_matches_observed": (
            claimed_cases == len(cases) if claimed_cases is not None else None
        ),
        "metric_kind": kind,
        "weight_view": weight_view(path, value),
        "determinism": determinism_status(value),
        "cases": cases,
        "seeds": unique_scalars(value, "seed"),
        "identifiers": {
            key: unique_scalars(value, key)
            for key in ("model", "variant", "policy")
            if unique_scalars(value, key)
        },
        "observation_count": len(records),
        "condition_groups": grouped_metric_stats(records),
        "exact_dockq": numeric_stats(row.get("dockq") for row in records),
        "approximate_dockq": numeric_stats(row.get("approx_dockq") for row in records),
        "reported_summaries": reported_summaries(value),
    }


def checkpoint_inventory(source: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for path in sorted(source.rglob("*.pt")):
        if path.name not in CHECKPOINT_NAMES:
            continue
        checkpoints.append(
            {
                "experiment": path.parent.name,
                "role": path.stem,
                "path": str(path),
                "relative_path": str(path.relative_to(source)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "contents_inspected": False,
            }
        )
    return checkpoints


def duplicate_checkpoint_groups(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in checkpoints:
        grouped[item["sha256"]].append(item)
    return [
        {
            "sha256": digest,
            "count": len(items),
            "paths": [item["path"] for item in items],
        }
        for digest, items in sorted(grouped.items())
        if len(items) > 1
    ]


def factorial_arms(json_values: dict[Path, Any], source: Path) -> dict[str, Any] | None:
    path = source / "overnight_root_cause_v3_run04" / "summary.json"
    value = json_values.get(path)
    if not isinstance(value, dict) or not isinstance(value.get("arms"), list):
        return None
    arms = []
    for arm in value["arms"]:
        history = arm.get("history", [])
        checkpoint = source / "overnight_root_cause_v3_run04" / (
            f"{arm['chemistry']}_{arm['torsion']}_{arm['adaptation']}_seed{arm['seed']}"
        ) / "best.pt"
        arms.append(
            {
                "chemistry": arm.get("chemistry"),
                "torsion": arm.get("torsion"),
                "adaptation": arm.get("adaptation"),
                "seed": arm.get("seed"),
                "completed_epochs": arm.get("completed_epochs"),
                "steps": arm.get("steps"),
                "best_val_loss": arm.get("best_val_loss"),
                "first_train_loss": history[0].get("train_loss") if history else None,
                "last_train_loss": history[-1].get("train_loss") if history else None,
                "transferred_checkpoint": str(checkpoint) if checkpoint.exists() else None,
                "transferred_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
                "recorded_original_checkpoint": arm.get("checkpoint"),
            }
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "scope": value.get("scope"),
        "arms": arms,
    }


def format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Handoff Results Summary",
        "",
        f"Source: `{summary['source_root']}`",
        "",
        "This report inventories transferred JSON records and model checkpoints. It does not rerun experiments or inspect pickle checkpoint contents.",
        "",
        "## Metric and comparison rules",
        "",
        "- `dockq` is catalogued as exact DockQ because the same records include DockQ Fnat/LRMS/iRMS fields; `approx_dockq` is catalogued separately.",
        "- Exact and approximate DockQ are not interchangeable and are never pooled here.",
        "- Training and validation losses are optimization diagnostics, not structural-quality metrics.",
        "- A recorded seed is not evidence of deterministic CUDA execution. Runs are marked unverified unless their own JSON records `deterministic_algorithms`.",
        "- Comparisons are valid only within matched cases, seeds, initialization, checkpoint view, graph/code version, and metric implementation.",
        "",
        "## DockQ-bearing evaluations",
        "",
        "| Experiment/file | Metric kind | View | Cases | Seeds | Exact mean | Approx. mean | Determinism |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for item in summary["evaluations"]:
        exact = item["exact_dockq"]
        approx = item["approximate_dockq"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                item["relative_path"],
                item["metric_kind"],
                item["weight_view"]["view"],
                len(item["cases"]),
                ", ".join(map(str, item["seeds"])) or "—",
                format_number(exact["mean"] if exact else None),
                format_number(approx["mean"] if approx else None),
                item["determinism"]["status"],
            )
        )
        if item["scope_case_count_matches_observed"] is False:
            lines.append(
                f"| ↳ scope discrepancy | claims {item['scope_case_count_claim']} cases; "
                f"records contain {len(item['cases'])} | — | — | — | — | — | source inconsistency |"
            )
    lines += [
        "",
        "The observation means above are recomputed from case/seed records. When a file supplies aggregate `summary` rows, those are retained verbatim in `summary.json`.",
        "",
        "### Comparable condition summaries",
        "",
        "Conditions within a file are kept separate below. Cross-file rows are comparable only when all comparison rules above are satisfied.",
        "",
        "| File | Condition | N | Cases | Seeds | Exact mean | Approx. mean |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for item in summary["evaluations"]:
        for group in item["condition_groups"]:
            condition = ", ".join(f"{key}={value}" for key, value in group["identifiers"].items())
            exact = group["exact_dockq"]
            approx = group["approximate_dockq"]
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    item["relative_path"],
                    condition,
                    group["observations"],
                    len(group["cases"]),
                    ", ".join(map(str, group["seeds"])) or "—",
                    format_number(exact["mean"] if exact else None),
                    format_number(approx["mean"] if approx else None),
                )
            )
    lines += [
        "",
        "## Raw/EMA evidence",
        "",
        "The view below is reported only when a JSON path or its `scope` explicitly says raw or EMA. Unspecified files are not assigned a view.",
        "",
        "| JSON | Recorded view | Scope |",
        "|---|---|---|",
    ]
    for item in summary["weight_view_records"]:
        lines.append(
            f"| `{item['relative_path']}` | {item['weight_view']['view']} | "
            f"{item['scope'] or '—'} |"
        )
    lines += [
        "",
        "A mixed label means the source text mentions both views (for example, EMA initialization followed by raw-weight validation); it does not assert that the resulting numbers are interchangeable.",
        "",
        "## Training histories",
        "",
        "| Experiment | Epochs | Train first → last (min) | Validation minimum | Source |",
        "|---|---:|---|---|---|",
    ]
    for item in summary["training_histories"]:
        validation = ", ".join(
            f"{key}={format_number(stats['minimum'])}"
            for key, stats in item["validation_loss"].items()
        ) or "—"
        train = item["train_loss"]
        lines.append(
            f"| {item['experiment']} | {item['epochs_recorded']} | "
            f"{format_number(train['first'])} → {format_number(train['last'])} "
            f"({format_number(train['minimum'])}) | {validation} | `{item['relative_path']}` |"
        )
    factorial = summary.get("factorial_training")
    if factorial:
        lines += [
            "",
            "## Factorial checkpoints",
            "",
            f"The factorial file records {len(factorial['arms'])} completed arms. Its validation scope is: {factorial['scope']}",
            "",
            "| Chemistry | Torsion | Adaptation | Seed | Epochs | Best val loss | Checkpoint SHA-256 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for arm in factorial["arms"]:
            digest = arm["transferred_checkpoint_sha256"]
            lines.append(
                f"| {arm['chemistry']} | {arm['torsion']} | {arm['adaptation']} | "
                f"{arm['seed']} | {arm['completed_epochs']} | {format_number(arm['best_val_loss'])} | "
                f"`{digest or 'missing'}` |"
            )
    lines += [
        "",
        "## Model checkpoint inventory",
        "",
        "Feature and graph cache tensors are excluded. Checkpoint contents were not deserialized.",
        "",
        "| Experiment | Role | Bytes | SHA-256 | Relative path |",
        "|---|---|---:|---|---|",
    ]
    for item in summary["checkpoints"]:
        lines.append(
            f"| {item['experiment']} | {item['role']} | {item['size_bytes']} | "
            f"`{item['sha256']}` | `{item['relative_path']}` |"
        )
    lines += [
        "",
        "## Inventory and limitations",
        "",
        f"- Parsed JSON files: {summary['counts']['json_files']} ({summary['counts']['json_parse_errors']} parse errors).",
        f"- DockQ-bearing JSON files: {summary['counts']['evaluation_files']}.",
        f"- Model checkpoint files: {summary['counts']['checkpoints']}; duplicate-content groups: {summary['counts']['duplicate_checkpoint_groups']}.",
        f"- Exact overall DockQ appears in {summary['counts']['exact_dockq_files']} file; approximate DockQ appears in {summary['counts']['approximate_dockq_files']} files.",
        "- Historical metric files without an explicit determinism flag remain useful records, but exact trajectory reproduction is unverified.",
        "- Paths and SHA-256 hashes for every JSON and selected model checkpoint are in `summary.json`; feature/cache `.pt` files are intentionally excluded from the checkpoint inventory.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.out.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")

    json_values: dict[Path, Any] = {}
    inventory = []
    parse_errors = []
    for path in sorted(source.rglob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_errors.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(source)),
                    "sha256": sha256(path),
                    "error": str(exc),
                }
            )
            continue
        json_values[path] = value
        inventory.append(json_inventory(path, value, source))

    evaluations = [
        result
        for path, value in json_values.items()
        if (result := evaluation_summary(path, value, source)) is not None
    ]
    histories = [
        result
        for path, value in json_values.items()
        if path.name == "history.json"
        and (result := summarize_history(path, value, source)) is not None
    ]
    checkpoints = checkpoint_inventory(source)
    duplicate_groups = duplicate_checkpoint_groups(checkpoints)
    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "output_root": str(output),
        "provenance": {
            "script_path": str(script_path),
            "script_sha256": sha256(script_path),
            "method": "Static JSON parsing and SHA-256 hashing; no checkpoint deserialization or experiment rerun.",
        },
        "metric_definitions": {
            "exact_dockq": "Fields named dockq; records also expose DockQ Fnat/LRMS/iRMS components.",
            "approximate_dockq": "Fields named approx_dockq; kept separate from exact DockQ.",
            "loss": "Recorded training/validation objective; not a DockQ proxy.",
        },
        "comparison_requirements": [
            "Use identical metric kind; never pool exact and approximate DockQ.",
            "Match cases and seeds, and verify initializer state when available.",
            "Match raw/EMA view; unspecified views are not assumed equivalent.",
            "Match code/graph implementation and evaluation protocol.",
        ],
        "default_nondeterministic_limitation": (
            "Seeded historical GPU results are marked unverified unless the source JSON "
            "explicitly records deterministic algorithms. A seed alone does not prove "
            "trajectory repeatability."
        ),
        "counts": {
            "json_files": len(inventory) + len(parse_errors),
            "json_parse_errors": len(parse_errors),
            "evaluation_files": len(evaluations),
            "exact_dockq_files": sum(item["exact_dockq"] is not None for item in evaluations),
            "approximate_dockq_files": sum(item["approximate_dockq"] is not None for item in evaluations),
            "training_histories": len(histories),
            "checkpoints": len(checkpoints),
            "duplicate_checkpoint_groups": len(duplicate_groups),
        },
        "evaluations": evaluations,
        "weight_view_records": [
            item for item in inventory if item["weight_view"]["view"] != "unspecified"
        ],
        "training_histories": histories,
        "factorial_training": factorial_arms(json_values, source),
        "checkpoints": checkpoints,
        "duplicate_checkpoint_groups": duplicate_groups,
        "json_inventory": inventory,
        "json_parse_errors": parse_errors,
    }

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    report_path = output / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path.write_text(report_markdown(summary))
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
