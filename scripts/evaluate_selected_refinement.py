#!/usr/bin/env python
"""Measure whether CoopDiff refinement rescues native-free proposal selection."""

import argparse
import contextlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from evaluation_common import resolve_entry_paths, sha256_file
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq
from tricoopdiff.expert_anchors import (
    DualExpertAnchorInitializer, FixedCandidateInitializer,
    generate_hybrid_candidates, select_hybrid_walkers,
)
from tricoopdiff.sampler import sample_single


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "bitseed_hybrid_selected_refinement_v2"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", required=True)
    result.add_argument("--out-dir", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--cases", help="Comma-separated case IDs; default is the full index")
    result.add_argument("--device", default="cuda")
    result.add_argument("--cache-dir", default="results/featurize_cache_proposal_oracle")
    result.add_argument("--deepternary-root", default="DeepTernary")
    result.add_argument("--native-root", default="DeepTernary/data/TernaryDB/pdbs")
    result.add_argument("--seed", type=int, default=31_000)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--binary-steps", type=int, default=20)
    result.add_argument("--conformers", type=int, default=4)
    result.add_argument("--poses-per-side", type=int, default=8)
    result.add_argument("--bitseed-candidates", type=int, default=192)
    result.add_argument("--sdpi-candidates", type=int, default=64)
    result.add_argument("--binary-retry-limit", type=int, default=2)
    result.add_argument("--binary-consecutive-failure-limit", type=int, default=3)
    result.add_argument("--max-refinements", type=int, help="Smoke-only cap on the union")
    result.add_argument(
        "--refine-full-pool", action="store_true",
        help="Refine all 256 candidates so every selector has a counterfactual final label",
    )
    result.add_argument("--allow-experimental-checkpoint", action="store_true")
    result.add_argument("--no-resume", action="store_true")
    return result


def atomic_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def source(candidate):
    return "sdpi" if candidate["candidate_id"][0] == "sdpi" else "bitseed"


def exact_dockq(candidate_ligand, candidate_poi, *, e3_data, paths, native,
                deepternary_root, metric_paths):
    with (contextlib.redirect_stdout(io.StringIO()),
          contextlib.redirect_stderr(io.StringIO())):
        return compute_fullatom_dockq(
            pred_ligand=candidate_ligand,
            pred_poi_ca=candidate_poi,
            featurized_e3_ca=e3_data["coords"][0].detach().cpu().numpy(),
            protein1_pdb=str(paths["protein1"]),
            protein2_pdb=str(paths["protein2"]),
            ligand_pdb=str(paths["ligand"]),
            native_complex_pdb=str(native),
            deepternary_root=str(deepternary_root),
            **metric_paths,
        )


def summarize(rows, metric):
    values = [float(row[metric]) for row in rows
              if isinstance(row.get(metric), (int, float))
              and math.isfinite(row[metric])]
    return {
        "count": len(values),
        "best": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "hits_0.23": sum(value >= 0.23 for value in values),
        "hits_0.49": sum(value >= 0.49 for value in values),
    }


def main():
    args = parser().parse_args()
    if args.steps < 1 or args.max_refinements is not None and args.max_refinements < 1:
        raise ValueError("steps and max-refinements must be positive")
    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    if args.cases:
        requested = [case.strip() for case in args.cases.split(",") if case.strip()]
        by_case = {entry["case"]: entry for entry in entries}
        missing = set(requested) - set(by_case)
        if missing:
            raise ValueError(f"case IDs absent from index: {sorted(missing)}")
        entries = [by_case[case] for case in requested]

    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    deepternary_root = Path(args.deepternary_root)
    if not deepternary_root.is_absolute():
        deepternary_root = (ROOT / deepternary_root).resolve()
    manifest = {
        "protocol": PROTOCOL,
        "scope": "native-free selection followed by oracle-scored CoopDiff refinement",
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "cases": [entry["case"] for entry in entries],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "allow_experimental_checkpoint": args.allow_experimental_checkpoint,
        "candidate_pool": {
            "bitseed": args.bitseed_candidates, "sdpi": args.sdpi_candidates,
            "conformers": args.conformers, "poses_per_side": args.poses_per_side,
            "binary_steps": args.binary_steps,
            "binary_retry_limit": args.binary_retry_limit,
            "binary_consecutive_failure_limit": args.binary_consecutive_failure_limit,
        },
        "arms": {
            "sdpi_64": {"sdpi": 64},
            "bitseed_diversity_64": {"bitseed": 64},
            "geometry_64": {"source_quota": None, "diversity": False},
            "diversity_64": {"source_quota": None, "diversity": True},
            "hybrid_48_16": {"bitseed_quota": 48, "sdpi_quota": 16},
            "hybrid_56_8": {"bitseed_quota": 56, "sdpi_quota": 8},
        },
        "steps": args.steps,
        "proposal_seed": args.seed,
        "diffusion_seed_schedule": "S_plus_2000000_plus_pool_index",
        "max_refinements": args.max_refinements,
        "refine_full_pool": args.refine_full_pool,
        "shared_trajectory_policy": (
            "refine_every_pool_candidate_once" if args.refine_full_pool else
            "refine_each_unique_selected_candidate_once"
        ),
        "native_information_used_for_selection": False,
        "native_information_used_for_reported_metrics_only": True,
    }
    atomic_json(manifest, output / "manifest.json")

    runner = EvaluationRunner(
        ROOT, checkpoint, device=args.device, deepternary_root=deepternary_root,
        allow_experimental_checkpoint=args.allow_experimental_checkpoint,
    )
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
            metric_paths = {}
            for key in ("metric_protein1", "metric_protein2"):
                if key in entry:
                    value = Path(entry[key])
                    value = value if value.is_absolute() else ROOT / value
                    if not value.is_file():
                        raise FileNotFoundError(value)
                    metric_paths[f"{key}_pdb"] = str(value.resolve())
            if len(metric_paths) not in (0, 2):
                raise ValueError("provide both metric protein templates or neither")
            native_value = entry.get("complex")
            native = (Path(native_value) if native_value else
                      ROOT / args.native_root / case / "gt_complex.pdb")
            native = native if native.is_absolute() else ROOT / native
            if not native.is_file():
                raise FileNotFoundError(native)
            e3_data, poi_data, ligand_data, input_dim = featurize_complex(
                case, str(paths["protein1"]), str(paths["protein2"]),
                str(paths["ligand"]), runner.score_model_args, args.device,
                cache_dir=str((ROOT / args.cache_dir).resolve()),
            )
            runner._ensure_model(input_dim)
            proposal_initializer = DualExpertAnchorInitializer(
                runner.expert, e3_data, poi_data, binary_steps=args.binary_steps,
                proposal_attempts=args.poses_per_side,
                conformer_policy=runner.initializer.conformer_policy,
            )
            candidates, generation = generate_hybrid_candidates(
                proposal_initializer, e3_data["coords"][0], poi_data["coords"][0],
                ligand_data, seed=args.seed, conformer_count=args.conformers,
                bitseed_candidate_budget=args.bitseed_candidates,
                sdpi_candidate_budget=args.sdpi_candidates,
                binary_retry_limit=args.binary_retry_limit,
                binary_consecutive_failure_limit=args.binary_consecutive_failure_limit,
            )
            sdpi = [candidate for candidate in candidates if source(candidate) == "sdpi"]
            if len(sdpi) < 64:
                raise RuntimeError(f"only {len(sdpi)} SDPI candidates available")
            selected_48, meta_48 = select_hybrid_walkers(
                candidates, 64, bitseed_quota=48,
            )
            selected_56, meta_56 = select_hybrid_walkers(
                candidates, 64, bitseed_quota=56,
            )
            selected_geometry, meta_geometry = select_hybrid_walkers(
                candidates, 64, use_diversity=False, protect_sources=False,
            )
            selected_diversity, meta_diversity = select_hybrid_walkers(
                candidates, 64, use_diversity=True, protect_sources=False,
            )
            bitseed = [candidate for candidate in candidates
                       if source(candidate) == "bitseed"]
            if len(bitseed) >= 64:
                selected_bitseed, meta_bitseed = select_hybrid_walkers(
                    bitseed, 64, bitseed_quota=64,
                    use_diversity=True, protect_sources=False,
                )
            else:
                selected_bitseed = list(bitseed)
                meta_bitseed = {
                    "walker_count": 64,
                    "status": "insufficient_bitseed_candidates",
                    "available_count": len(bitseed),
                    "selected_candidate_ids": [
                        candidate["candidate_id"] for candidate in bitseed
                    ],
                }
            arms = {
                "sdpi_64": sdpi[:64],
                "bitseed_diversity_64": selected_bitseed,
                "geometry_64": selected_geometry,
                "diversity_64": selected_diversity,
                "hybrid_48_16": selected_48,
                "hybrid_56_8": selected_56,
            }
            pool_index = {repr(candidate["candidate_id"]): index
                          for index, candidate in enumerate(candidates)}
            selected_ids = {repr(candidate["candidate_id"])
                            for arm in arms.values() for candidate in arm}
            union = (list(candidates) if args.refine_full_pool else [
                candidate for candidate in candidates
                if repr(candidate["candidate_id"]) in selected_ids
            ])
            union_requested_count = len(union)
            if args.max_refinements is not None:
                union = union[:args.max_refinements]
            rows = []
            for union_index, candidate in enumerate(union, start=1):
                key = repr(candidate["candidate_id"])
                index = pool_index[key]
                diffusion_seed = (args.seed + 2_000_000 + index) % (2**31 - 1)
                print(
                    f"[{position}/{len(entries)}] {case} refinement "
                    f"{union_index}/{len(union)} id={candidate['candidate_id']}",
                    flush=True,
                )
                initial_ligand = candidate["ligand"][0].detach().cpu().numpy()
                initial_poi = candidate["poi"][0].detach().cpu().numpy()
                initial = exact_dockq(
                    initial_ligand, initial_poi, e3_data=e3_data, paths=paths,
                    native=native, deepternary_root=deepternary_root,
                    metric_paths=metric_paths,
                )
                fixed = FixedCandidateInitializer(
                    candidate, no_torsion=runner.initializer.no_torsion,
                )
                sampled = sample_single(
                    runner.model, runner.diffusion, fixed,
                    e3_data, poi_data, ligand_data, steps=args.steps,
                    device=args.device, seed=diffusion_seed,
                    init_mode="fixed_candidate", t_anchor=1.0,
                )
                row = {
                    "candidate_id": candidate["candidate_id"],
                    "pool_index": index,
                    "source": source(candidate),
                    "acceptable": bool(candidate.get("acceptable", False)),
                    "penalty": list(candidate.get("penalty", ())),
                    "geometry": candidate.get("geometry"),
                    "fit_rmsd": candidate.get("fit_rmsd"),
                    "diffusion_seed": diffusion_seed,
                    "initial_dockq": initial["dockq"],
                    "initial_fnat": initial["fnat"],
                    "initial_lrms": initial["lrms"],
                    "initial_irms": initial["irms"],
                    "valid": bool(sampled.get("valid", True)),
                    "termination_reason": sampled.get("termination_reason"),
                }
                if row["valid"]:
                    refined = exact_dockq(
                        sampled["ligand_coords"], sampled["poi_coords"],
                        e3_data=e3_data, paths=paths, native=native,
                        deepternary_root=deepternary_root,
                        metric_paths=metric_paths,
                    )
                    row.update({
                        "refined_dockq": refined["dockq"],
                        "refined_fnat": refined["fnat"],
                        "refined_lrms": refined["lrms"],
                        "refined_irms": refined["irms"],
                        "refinement_delta_dockq": (
                            refined["dockq"] - initial["dockq"]
                        ),
                        "safe_dockq": max(refined["dockq"], initial["dockq"]),
                        "safe_stage": (
                            "refined" if refined["dockq"] >= initial["dockq"]
                            else "initial"
                        ),
                    })
                rows.append(row)
            by_id = {repr(row["candidate_id"]): row for row in rows}
            arm_reports = {}
            for name, selected in arms.items():
                selected_rows = [by_id[repr(candidate["candidate_id"])]
                                 for candidate in selected
                                 if repr(candidate["candidate_id"]) in by_id]
                arm_reports[name] = {
                    "requested_count": len(selected),
                    "evaluated_count": len(selected_rows),
                    "selected_candidate_ids": [candidate["candidate_id"]
                                               for candidate in selected],
                    "initial": summarize(selected_rows, "initial_dockq"),
                    "refined": summarize(selected_rows, "refined_dockq"),
                    "safe_union": summarize(selected_rows, "safe_dockq"),
                }
            if args.refine_full_pool:
                arm_reports["full_pool_256"] = {
                    "requested_count": len(candidates),
                    "evaluated_count": len(rows),
                    "selected_candidate_ids": [candidate["candidate_id"]
                                               for candidate in candidates],
                    "initial": summarize(rows, "initial_dockq"),
                    "refined": summarize(rows, "refined_dockq"),
                    "safe_union": summarize(rows, "safe_dockq"),
                }
            source_reports = {}
            for candidate_source in ("bitseed", "sdpi"):
                source_rows = [row for row in rows
                               if row["source"] == candidate_source]
                source_reports[candidate_source] = {
                    "count": len(source_rows),
                    "initial": summarize(source_rows, "initial_dockq"),
                    "refined": summarize(source_rows, "refined_dockq"),
                    "safe_union": summarize(source_rows, "safe_dockq"),
                    "improved_count": sum(
                        row.get("refinement_delta_dockq", 0.0) > 0.0
                        for row in source_rows
                    ),
                    "damaged_count": sum(
                        row.get("refinement_delta_dockq", 0.0) < 0.0
                        for row in source_rows
                    ),
                }
            report = {
                "status": "complete",
                "case": case,
                "config": manifest,
                "generation": generation,
                "selector_metadata": {
                    "bitseed_diversity_64": meta_bitseed,
                    "geometry_64": meta_geometry,
                    "diversity_64": meta_diversity,
                    "hybrid_48_16": meta_48,
                    "hybrid_56_8": meta_56,
                },
                "union_requested_count": union_requested_count,
                "union_evaluated_count": len(rows),
                "arms": arm_reports,
                "sources": source_reports,
                "refinements": rows,
                "timing_seconds": {"total": time.monotonic() - started},
            }
            atomic_json(report, destination)
            print(f"[{position}/{len(entries)}] DONE {case}", flush=True)
        except Exception as error:
            failures.append({"case": case, "error": f"{type(error).__name__}: {error}"})
            print(f"[{position}/{len(entries)}] FAIL {case}: {error}", flush=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    atomic_json({"config": manifest, "failures": failures}, output / "failures.json")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
