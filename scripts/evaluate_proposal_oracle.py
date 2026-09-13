#!/usr/bin/env python
"""Evaluate the native-reading oracle of online dual-expert proposals."""

import argparse
import contextlib
import io
import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation_common import resolve_entry_paths, sha256_file
from featurize_common import featurize_complex
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq
from tricoopdiff.evaluate import compute_deepternary_dockq
from tricoopdiff.expert_anchors import (
    DualExpertAnchorInitializer, generate_hybrid_candidates,
    generate_sdpi_candidates, select_hybrid_walkers,
)
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.reproducibility import configure_inference_reproducibility


ROOT = Path(__file__).resolve().parent.parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", required=True)
    result.add_argument("--out-dir", required=True)
    result.add_argument("--conformers", type=int, required=True)
    result.add_argument("--poses-per-side", type=int, required=True)
    result.add_argument("--candidate-budget", type=int, default=256)
    result.add_argument(
        "--sdpi-candidates", type=int, default=0,
        help="Mix this many independent SDPI proposals into the fixed budget",
    )
    result.add_argument("--binary-steps", type=int, default=20)
    result.add_argument(
        "--binary-retry-limit", type=int, default=2,
        help="Maximum draw multiplier while collecting up to M valid poses per side",
    )
    result.add_argument(
        "--binary-consecutive-failure-limit", type=int, default=3,
        help="Stop a binary side after this many consecutive NaN/Inf trajectories",
    )
    result.add_argument("--seed", type=int, default=31_000)
    result.add_argument("--device", default="cuda")
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--limit", type=int)
    result.add_argument(
        "--cases",
        help="Optional comma-separated case IDs, retained in index order",
    )
    result.add_argument("--cache-dir", default="results/featurize_cache_proposal_oracle")
    result.add_argument("--native-root", default="DeepTernary/data/TernaryDB/pdbs")
    result.add_argument("--deepternary-root", default="DeepTernary")
    result.add_argument("--model-dir", default="checkpoints/score_model")
    result.add_argument("--expert-checkpoint", default="best_ema_inference_epoch_model.pt")
    result.add_argument("--skip-exact-dockq", action="store_true")
    result.add_argument(
        "--select-walkers", type=int, nargs="*", default=[],
        help="Also audit native-free selection at these walker budgets",
    )
    result.add_argument(
        "--selector-suite", action="store_true",
        help="Run preregistered component, quota, threshold, and random audits",
    )
    result.add_argument(
        "--fallback-to-sdpi", action="store_true",
        help="Replace a failed BiTSeed case with 256 deterministic SDPI proposals",
    )
    result.add_argument(
        "--sdpi-only", action="store_true",
        help="Evaluate 256 default SDPI proposals without loading the binary expert",
    )
    result.add_argument("--no-resume", action="store_true")
    return result


def atomic_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def summarize(rows, key):
    values = finite(row.get(key) for row in rows)
    if not values:
        return {"count": 0, "best": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "best": max(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "hits_0.23": sum(value >= 0.23 for value in values),
        "hits_0.49": sum(value >= 0.49 for value in values),
    }


def selector_suite_specs(walker_count, seed):
    """Return the fixed selector audit matrix without consulting quality labels."""
    default_quota = 3 * walker_count // 4
    quota_values = ([20, 28] if walker_count == 32 else [40, 56])
    specs = {
        "geometry_top": {"use_diversity": False, "protect_sources": False},
        "geometry_quota": {"use_diversity": False, "protect_sources": True},
        "geometry_diversity": {"use_diversity": True, "protect_sources": False},
        "full_default": {"use_diversity": True, "protect_sources": True},
        "threshold_strict_1A_3A": {
            "use_diversity": True, "protect_sources": True,
            "ligand_rmsd_cutoff": 1.0, "poi_rmsd_cutoff": 3.0,
        },
        "threshold_wide_3A_8A": {
            "use_diversity": True, "protect_sources": True,
            "ligand_rmsd_cutoff": 3.0, "poi_rmsd_cutoff": 8.0,
        },
    }
    for quota in quota_values:
        specs[f"quota_bitseed_{quota}_sdpi_{walker_count - quota}"] = {
            "bitseed_quota": quota,
            "use_diversity": True,
            "protect_sources": True,
        }
    for repeat in range(10):
        specs[f"random_quota_seed_{repeat}"] = {
            "bitseed_quota": default_quota,
            "use_diversity": False,
            "protect_sources": True,
            "random_seed": int(seed) + 2_000_000 + repeat,
        }
    return specs


def audit_selection(selected, metadata, row_by_id, rows, oracle_index, oracle_key):
    selected_rows = [
        row_by_id[repr(candidate["candidate_id"])] for candidate in selected
    ]
    selected_best = max(selected_rows, key=lambda row: row[oracle_key])
    metadata.update({
        "oracle_metric": oracle_key,
        "selected_oracle_candidate": selected_best,
        "selected_summary": summarize(selected_rows, oracle_key),
        "full_oracle_retained": (
            selected_best[oracle_key] == rows[oracle_index][oracle_key]
        ),
        "oracle_retention_ratio": (
            selected_best[oracle_key] / max(rows[oracle_index][oracle_key], 1e-12)
        ),
    })
    return metadata


def main():
    args = parser().parse_args()
    if args.conformers < 1 or args.poses_per_side < 1:
        raise ValueError("conformers and poses-per-side must be positive")
    potential_count = args.conformers * args.poses_per_side ** 2
    bitseed_count = args.candidate_budget - args.sdpi_candidates
    if (args.candidate_budget != 256 or args.sdpi_candidates < 0
            or args.sdpi_candidates > args.candidate_budget):
        raise ValueError("this ablation requires a valid 256-candidate budget")
    if (not args.sdpi_only and (bitseed_count < 1
            or bitseed_count > potential_count
            or bitseed_count % args.conformers)):
        raise ValueError(
            "BiTSeed budget must divide across conformers and not exceed the Cartesian product"
        )
    configure_inference_reproducibility(True)
    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    if args.cases:
        requested = {case.strip() for case in args.cases.split(",") if case.strip()}
        entries = [entry for entry in entries if entry["case"] in requested]
        found = {entry["case"] for entry in entries}
        if found != requested:
            raise ValueError(f"case IDs absent from index: {sorted(requested - found)}")
    entries = entries[args.start:] if args.limit is None else entries[
        args.start:args.start + args.limit
    ]
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = (ROOT / args.model_dir).resolve()
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))
    expert = None
    if not args.sdpi_only:
        expert = build_diffdock_expert(
            model_dir=str(model_dir), device=args.device,
            ckpt=args.expert_checkpoint, num_conv_layers=model_args.num_conv_layers,
        )
    protocol = ("fixed_256_sdpi_proposal_oracle_v2" if args.sdpi_only else
                "fixed_256_bitseed_sdpi_hybrid_v2" if args.sdpi_candidates else
                "fixed_256_proposal_oracle_ablation_v5" if args.fallback_to_sdpi else
                "fixed_256_proposal_oracle_ablation_v4")
    manifest = {
        "protocol": protocol,
        "scope": "native-reading proposal oracle; no deployable ranking claim",
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "cases": [entry["case"] for entry in entries],
        "conformers": args.conformers,
        "poses_per_side": args.poses_per_side,
        "potential_pair_count": potential_count,
        "proposal_count": args.candidate_budget,
        "bitseed_proposal_count": 0 if args.sdpi_only else bitseed_count,
        "sdpi_proposal_count": args.candidate_budget if args.sdpi_only else args.sdpi_candidates,
        "binary_trajectories": 2 * args.conformers * args.poses_per_side,
        "binary_steps": args.binary_steps,
        "binary_retry_limit": args.binary_retry_limit,
        "binary_consecutive_failure_limit": args.binary_consecutive_failure_limit,
        "binary_recovery": (
            "retain_partial_pose_banks_then_fill_only_missing_pairs_with_sdpi"
        ),
        "fallback_policy": (
            "partial_pair_deficit_filled_with_deterministic_sdpi_v1"
            if args.sdpi_candidates else
            "256_deterministic_sdpi_v1_proposals" if args.fallback_to_sdpi else None
        ),
        "proposal_source": "sdpi_v1" if args.sdpi_only else "bitseed",
        "seed": args.seed,
        "sdpi_seed_schedule": "S_plus_1000000_plus_k",
        "conformer_policy": "etkdgv3_random",
        "pairing": "within_conformer_only",
        "pair_subsampling": (
            "balanced marginal degree for full banks; deterministic "
            "degree-balanced selection for partial banks"
        ),
        "pair_relaxation": "geometry-best of 13 outward offsets from 0A to 6A",
        "exact_dockq": not args.skip_exact_dockq,
        "selector_walker_counts": sorted(set(args.select_walkers)),
        "selector_suite": bool(args.selector_suite),
        "expert_checkpoint_sha256": (
            expert.provenance["score_checkpoint_sha256"] if expert is not None else None
        ),
    }
    atomic_json(manifest, output / "manifest.json")
    failures = []
    for position, entry in enumerate(entries, start=1):
        case = entry["case"]
        destination = output / f"{case}.json"
        if not args.no_resume and destination.is_file():
            try:
                previous = json.loads(destination.read_text(encoding="utf-8"))
                if previous.get("status") == "complete" and previous.get("config") == manifest:
                    print(f"[{position}/{len(entries)}] SKIP {case}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        print(f"[{position}/{len(entries)}] RUN {case}", flush=True)
        started = time.monotonic()
        try:
            paths = resolve_entry_paths(entry, ROOT)
            native_value = entry.get("complex")
            native = (Path(native_value) if native_value else
                      ROOT / args.native_root / case / "gt_complex.pdb")
            if not native.is_absolute():
                native = ROOT / native
            native = native.resolve()
            if not native.is_file():
                raise FileNotFoundError(native)
            metric_protein1 = entry.get("metric_protein1")
            metric_protein2 = entry.get("metric_protein2")
            if (metric_protein1 is None) != (metric_protein2 is None):
                raise ValueError(
                    "provide both metric_protein1 and metric_protein2 or neither"
                )
            metric_paths = {}
            if metric_protein1 is not None:
                metric_protein1 = Path(metric_protein1)
                metric_protein2 = Path(metric_protein2)
                if not metric_protein1.is_absolute():
                    metric_protein1 = ROOT / metric_protein1
                if not metric_protein2.is_absolute():
                    metric_protein2 = ROOT / metric_protein2
                metric_protein1 = metric_protein1.resolve()
                metric_protein2 = metric_protein2.resolve()
                if not metric_protein1.is_file():
                    raise FileNotFoundError(metric_protein1)
                if not metric_protein2.is_file():
                    raise FileNotFoundError(metric_protein2)
                metric_paths = {
                    "metric_protein1_pdb": str(metric_protein1),
                    "metric_protein2_pdb": str(metric_protein2),
                }
            e3_data, poi_data, ligand_data, _ = featurize_complex(
                case, str(paths["protein1"]), str(paths["protein2"]),
                str(paths["ligand"]), model_args, args.device,
                cache_dir=str((ROOT / args.cache_dir).resolve()),
            )
            generated_at = time.monotonic()
            fallback_reason = None
            if args.sdpi_only:
                candidates, conformers = generate_sdpi_candidates(
                    TernaryInitializer(conformer_policy="etkdgv3_random"),
                    e3_data["coords"][0], poi_data["coords"][0], ligand_data,
                    args.candidate_budget, int(args.seed) + 1_000_000,
                    seed_stride=1,
                )
                generation_mode = "sdpi_v1"
            else:
                initializer = DualExpertAnchorInitializer(
                    expert, e3_data, poi_data, binary_steps=args.binary_steps,
                    proposal_attempts=args.poses_per_side,
                    conformer_policy="etkdgv3_random",
                )
                try:
                    if args.sdpi_candidates:
                        candidates, hybrid = generate_hybrid_candidates(
                            initializer, e3_data["coords"][0],
                            poi_data["coords"][0], ligand_data, seed=args.seed,
                            conformer_count=args.conformers,
                            bitseed_candidate_budget=bitseed_count,
                            sdpi_candidate_budget=args.sdpi_candidates,
                            binary_retry_limit=args.binary_retry_limit,
                            binary_consecutive_failure_limit=(
                                args.binary_consecutive_failure_limit
                            ),
                        )
                        conformers = hybrid["successful_conformers"]
                        generation_mode = "bitseed_sdpi_hybrid"
                        fallback_reason = hybrid["fallback_events"] or None
                    else:
                        candidates, conformers, _ = initializer.generate_candidates(
                            e3_data["coords"][0], poi_data["coords"][0], ligand_data,
                            seed=args.seed, conformer_count=args.conformers,
                            candidate_budget=args.candidate_budget,
                            binary_retry_limit=args.binary_retry_limit,
                            binary_consecutive_failure_limit=(
                                args.binary_consecutive_failure_limit
                            ),
                        )
                        generation_mode = "bitseed"
                except FloatingPointError as error:
                    if not args.fallback_to_sdpi:
                        raise
                    fallback_reason = str(error)
                    candidates, conformers = generate_sdpi_candidates(
                        initializer.base, e3_data["coords"][0],
                        poi_data["coords"][0], ligand_data,
                        args.candidate_budget, args.seed,
                    )
                    generation_mode = "sdpi_v1_fallback"
            generation_seconds = time.monotonic() - generated_at
            native_ligand = ligand_data["coords"][0].detach().cpu().numpy()
            native_e3 = e3_data["coords"][0].detach().cpu().numpy()
            native_poi = poi_data["coords"][0].detach().cpu().numpy()
            rows = []
            scoring_at = time.monotonic()
            for candidate in candidates:
                pred_ligand = candidate["ligand"][0].detach().cpu().numpy()
                pred_poi = candidate["poi"][0].detach().cpu().numpy()
                approximate = compute_deepternary_dockq(
                    native_ligand, native_e3, native_poi,
                    pred_ligand, native_e3, pred_poi,
                )
                row = {
                    "candidate_id": candidate["candidate_id"],
                    "acceptable": candidate["acceptable"],
                    "fit_rmsd": candidate["fit_rmsd"],
                    "geometry": candidate["geometry"],
                    "approx_dockq": (approximate["dockq"] if approximate else None),
                }
                if not args.skip_exact_dockq:
                    with (contextlib.redirect_stdout(io.StringIO()),
                          contextlib.redirect_stderr(io.StringIO())):
                        exact = compute_fullatom_dockq(
                            pred_ligand=pred_ligand, pred_poi_ca=pred_poi,
                            featurized_e3_ca=native_e3,
                            protein1_pdb=str(paths["protein1"]),
                            protein2_pdb=str(paths["protein2"]),
                            ligand_pdb=str(paths["ligand"]),
                            native_complex_pdb=str(native),
                            deepternary_root=str((ROOT / args.deepternary_root).resolve()),
                            **metric_paths,
                        )
                    row.update({key: exact[key]
                                for key in ("dockq", "fnat", "lrms", "irms")})
                rows.append(row)
            scoring_seconds = time.monotonic() - scoring_at
            oracle_key = "dockq" if not args.skip_exact_dockq else "approx_dockq"
            oracle_index = max(range(len(rows)), key=lambda index: rows[index][oracle_key])
            selections = {}
            row_by_id = {
                repr(row["candidate_id"]): row for row in rows
            }
            for walker_count in sorted(set(args.select_walkers)):
                selected, selection_metadata = select_hybrid_walkers(
                    candidates, walker_count,
                )
                selections[str(walker_count)] = audit_selection(
                    selected, selection_metadata, row_by_id, rows,
                    oracle_index, oracle_key,
                )
            selector_audits = {}
            if args.selector_suite:
                for walker_count in (32, 64):
                    audits = {}
                    for name, options in selector_suite_specs(
                            walker_count, args.seed).items():
                        selected, selection_metadata = select_hybrid_walkers(
                            candidates, walker_count, **options,
                        )
                        audits[name] = audit_selection(
                            selected, selection_metadata, row_by_id, rows,
                            oracle_index, oracle_key,
                        )
                    selector_audits[str(walker_count)] = audits
            report = {
                "status": "complete",
                "case": case,
                "config": manifest,
                "n_candidates": len(rows),
                "n_acceptable": sum(row["acceptable"] for row in rows),
                "generation_mode": generation_mode,
                "fallback_reason": fallback_reason,
                "binary_retries": (0 if generation_mode not in {
                    "bitseed", "bitseed_sdpi_hybrid"} else sum(
                    sum(conformer.get(
                        "binary_failure_counts",
                        [sum(branch) for branch in conformer["binary_retry_counts"]],
                    ))
                    for conformer in conformers
                )),
                "binary_draws": (0 if generation_mode not in {
                    "bitseed", "bitseed_sdpi_hybrid"} else sum(
                    sum(conformer["binary_draw_counts"])
                    for conformer in conformers
                )),
                "oracle_metric": oracle_key,
                "oracle_index": oracle_index,
                "oracle_candidate": rows[oracle_index],
                "approximate_summary": summarize(rows, "approx_dockq"),
                "exact_summary": summarize(rows, "dockq"),
                "native_free_selections": selections,
                "selector_audits": selector_audits,
                "conformers": conformers,
                "candidates": rows,
                "timing_seconds": {
                    "generation": generation_seconds,
                    "scoring": scoring_seconds,
                    "total": time.monotonic() - started,
                },
            }
            atomic_json(report, destination)
            print(
                f"[{position}/{len(entries)}] DONE {case} "
                f"oracle={rows[oracle_index][oracle_key]:.6f} "
                f"generation={generation_seconds:.1f}s scoring={scoring_seconds:.1f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({"case": case, "error": f"{type(error).__name__}: {error}"})
            print(f"[{position}/{len(entries)}] FAIL {case}: {error}", flush=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    atomic_json({
        "requested_cases": len(entries),
        "failed_cases": failures,
        "complete": not failures,
    }, output / "batch_status.json")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
