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
from tricoopdiff.expert_anchors import DualExpertAnchorInitializer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility


ROOT = Path(__file__).resolve().parent.parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", required=True)
    result.add_argument("--out-dir", required=True)
    result.add_argument("--conformers", type=int, required=True)
    result.add_argument("--poses-per-side", type=int, required=True)
    result.add_argument("--candidate-budget", type=int, default=256)
    result.add_argument("--binary-steps", type=int, default=20)
    result.add_argument(
        "--binary-retry-limit", type=int, default=4,
        help="Maximum draw multiplier while collecting M valid poses per side",
    )
    result.add_argument("--seed", type=int, default=31_000)
    result.add_argument("--device", default="cuda")
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--limit", type=int)
    result.add_argument("--cache-dir", default="results/featurize_cache_proposal_oracle")
    result.add_argument("--native-root", default="DeepTernary/data/TernaryDB/pdbs")
    result.add_argument("--deepternary-root", default="DeepTernary")
    result.add_argument("--model-dir", default="checkpoints/score_model")
    result.add_argument("--expert-checkpoint", default="best_ema_inference_epoch_model.pt")
    result.add_argument("--skip-exact-dockq", action="store_true")
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


def main():
    args = parser().parse_args()
    if args.conformers < 1 or args.poses_per_side < 1:
        raise ValueError("conformers and poses-per-side must be positive")
    potential_count = args.conformers * args.poses_per_side ** 2
    if args.candidate_budget != 256 or potential_count < args.candidate_budget:
        raise ValueError(
            "this ablation requires a 256-candidate budget no larger than the Cartesian product"
        )
    configure_inference_reproducibility(True)
    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    entries = entries[args.start:] if args.limit is None else entries[
        args.start:args.start + args.limit
    ]
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = (ROOT / args.model_dir).resolve()
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))
    expert = build_diffdock_expert(
        model_dir=str(model_dir), device=args.device,
        ckpt=args.expert_checkpoint, num_conv_layers=model_args.num_conv_layers,
    )
    manifest = {
        "protocol": "fixed_256_proposal_oracle_ablation_v4",
        "scope": "native-reading proposal oracle; no deployable ranking claim",
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "cases": [entry["case"] for entry in entries],
        "conformers": args.conformers,
        "poses_per_side": args.poses_per_side,
        "potential_pair_count": potential_count,
        "proposal_count": args.candidate_budget,
        "binary_trajectories": 2 * args.conformers * args.poses_per_side,
        "binary_steps": args.binary_steps,
        "binary_retry_limit": args.binary_retry_limit,
        "binary_recovery": "collect_M_valid_poses_with_max_M_times_retry_limit_draws",
        "seed": args.seed,
        "conformer_policy": "etkdgv3_random",
        "pairing": "within_conformer_only",
        "pair_subsampling": "deterministic balanced marginal degree",
        "pair_relaxation": "geometry-best of 13 outward offsets from 0A to 6A",
        "exact_dockq": not args.skip_exact_dockq,
        "expert_checkpoint_sha256": expert.provenance["score_checkpoint_sha256"],
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
            native = (ROOT / args.native_root / case / "gt_complex.pdb").resolve()
            if not native.is_file():
                raise FileNotFoundError(native)
            e3_data, poi_data, ligand_data, _ = featurize_complex(
                case, str(paths["protein1"]), str(paths["protein2"]),
                str(paths["ligand"]), model_args, args.device,
                cache_dir=str((ROOT / args.cache_dir).resolve()),
            )
            initializer = DualExpertAnchorInitializer(
                expert, e3_data, poi_data, binary_steps=args.binary_steps,
                proposal_attempts=args.poses_per_side,
                conformer_policy="etkdgv3_random",
            )
            generated_at = time.monotonic()
            candidates, conformers, _ = initializer.generate_candidates(
                e3_data["coords"][0], poi_data["coords"][0], ligand_data,
                seed=args.seed, conformer_count=args.conformers,
                candidate_budget=args.candidate_budget,
                binary_retry_limit=args.binary_retry_limit,
            )
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
                        )
                    row.update({key: exact[key]
                                for key in ("dockq", "fnat", "lrms", "irms")})
                rows.append(row)
            scoring_seconds = time.monotonic() - scoring_at
            oracle_key = "dockq" if not args.skip_exact_dockq else "approx_dockq"
            oracle_index = max(range(len(rows)), key=lambda index: rows[index][oracle_key])
            report = {
                "status": "complete",
                "case": case,
                "config": manifest,
                "n_candidates": len(rows),
                "n_acceptable": sum(row["acceptable"] for row in rows),
                "binary_retries": sum(
                    sum(sum(branch) for branch in conformer["binary_retry_counts"])
                    for conformer in conformers
                ),
                "binary_draws": sum(
                    sum(conformer["binary_draw_counts"])
                    for conformer in conformers
                ),
                "oracle_metric": oracle_key,
                "oracle_index": oracle_index,
                "oracle_candidate": rows[oracle_index],
                "approximate_summary": summarize(rows, "approx_dockq"),
                "exact_summary": summarize(rows, "dockq"),
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
