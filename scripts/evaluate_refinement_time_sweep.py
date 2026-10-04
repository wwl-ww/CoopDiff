#!/usr/bin/env python
"""Evaluate truncated CoopDiff refinement clocks for one fixed proposal."""

import argparse
import json
import time
from pathlib import Path

from evaluation_common import resolve_entry_paths, sha256_file
from evaluation_runner import EvaluationRunner
from evaluate_selected_refinement import atomic_json, exact_dockq
from featurize_common import featurize_complex
from tricoopdiff.expert_anchors import (
    DualExpertAnchorInitializer, FixedCandidateInitializer,
    generate_hybrid_candidates,
)
from tricoopdiff.online_endpoints import endpoint_internal_geometry
from tricoopdiff.sampler import sample_single


ROOT = Path(__file__).resolve().parent.parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", required=True)
    result.add_argument("--case", required=True)
    result.add_argument("--candidate-id", required=True, help="JSON candidate ID")
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--cache-dir", default="results/featurize_cache_proposal_oracle")
    result.add_argument("--deepternary-root", default="DeepTernary")
    result.add_argument("--native-root", default="DeepTernary/data/TernaryDB/pdbs")
    result.add_argument("--seed", type=int, default=31_000)
    result.add_argument(
        "--schedules", default="0.10:2,0.25:5,0.50:10,1.00:20",
        help="Comma-separated t_anchor:steps pairs",
    )
    result.add_argument("--allow-experimental-checkpoint", action="store_true")
    result.add_argument("--expect-identity", type=float)
    result.add_argument("--expect-refined", type=float)
    result.add_argument("--expect-tolerance", type=float, default=1e-5)
    return result


def main():
    args = parser().parse_args()
    candidate_id = json.loads(args.candidate_id)
    schedules = []
    for item in args.schedules.split(","):
        anchor, steps = item.split(":", 1)
        schedules.append((float(anchor), int(steps)))
    if not schedules or any(not 0.0 < anchor <= 1.0 or steps < 1
                            for anchor, steps in schedules):
        raise ValueError("every schedule must have 0 < t_anchor <= 1 and steps >= 1")

    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    matches = [entry for entry in entries if entry["case"] == args.case]
    if len(matches) != 1:
        raise ValueError(f"expected one index entry for {args.case}, found {len(matches)}")
    entry = matches[0]
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
              ROOT / args.native_root / args.case / "gt_complex.pdb")
    native = native if native.is_absolute() else ROOT / native
    if not native.is_file():
        raise FileNotFoundError(native)
    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    deepternary_root = Path(args.deepternary_root)
    deepternary_root = (deepternary_root if deepternary_root.is_absolute()
                        else ROOT / deepternary_root)

    runner = EvaluationRunner(
        ROOT, checkpoint, device=args.device,
        deepternary_root=deepternary_root,
        allow_experimental_checkpoint=args.allow_experimental_checkpoint,
    )
    e3_data, poi_data, ligand_data, input_dim = featurize_complex(
        args.case, str(paths["protein1"]), str(paths["protein2"]),
        str(paths["ligand"]), runner.score_model_args, args.device,
        cache_dir=str((ROOT / args.cache_dir).resolve()),
    )
    runner._ensure_model(input_dim)
    proposal_initializer = DualExpertAnchorInitializer(
        runner.expert, e3_data, poi_data, binary_steps=20,
        proposal_attempts=8,
        conformer_policy=runner.initializer.conformer_policy,
    )
    started = time.monotonic()
    candidates, generation = generate_hybrid_candidates(
        proposal_initializer, e3_data["coords"][0], poi_data["coords"][0],
        ligand_data, seed=args.seed, conformer_count=4,
        bitseed_candidate_budget=192, sdpi_candidate_budget=64,
        binary_retry_limit=2, binary_consecutive_failure_limit=3,
    )
    matches = [(index, candidate) for index, candidate in enumerate(candidates)
               if candidate["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected one generated candidate {candidate_id}, found {len(matches)}")
    pool_index, candidate = matches[0]
    diffusion_seed = (args.seed + 2_000_000 + pool_index) % (2**31 - 1)
    identity = exact_dockq(
        candidate["ligand"][0].detach().cpu().numpy(),
        candidate["poi"][0].detach().cpu().numpy(),
        e3_data=e3_data, paths=paths, native=native,
        deepternary_root=deepternary_root, metric_paths=metric_paths,
    )
    endpoint_geometry = endpoint_internal_geometry(
        ligand_data["coords"][0].detach().cpu().numpy(),
        candidate["ligand"][0].detach().cpu().numpy(),
        ligand_data["edge_index"].detach().cpu().numpy(),
    )
    results = []
    for anchor, steps in schedules:
        print(f"[sweep] t_anchor={anchor:.2f}, steps={steps}", flush=True)
        fixed = FixedCandidateInitializer(
            candidate, no_torsion=runner.initializer.no_torsion,
        )
        sampled = sample_single(
            runner.model, runner.diffusion, fixed,
            e3_data, poi_data, ligand_data, steps=steps,
            device=args.device, seed=diffusion_seed,
            init_mode="fixed_candidate", t_anchor=anchor,
        )
        row = {
            "t_anchor": anchor, "steps": steps,
            "valid": bool(sampled.get("valid", True)),
            "termination_reason": sampled.get("termination_reason"),
        }
        if row["valid"]:
            metric = exact_dockq(
                sampled["ligand_coords"], sampled["poi_coords"],
                e3_data=e3_data, paths=paths, native=native,
                deepternary_root=deepternary_root, metric_paths=metric_paths,
            )
            row.update(metric)
            row["dockq_delta_from_identity"] = metric["dockq"] - identity["dockq"]
        results.append(row)
    if args.expect_identity is not None:
        if abs(identity["dockq"] - args.expect_identity) > args.expect_tolerance:
            raise AssertionError(
                f"identity DockQ {identity['dockq']} != expected {args.expect_identity}"
            )
    if args.expect_refined is not None:
        if len(results) != 1 or not results[0]["valid"]:
            raise AssertionError("expected refined DockQ requires one valid schedule")
        if abs(results[0]["dockq"] - args.expect_refined) > args.expect_tolerance:
            raise AssertionError(
                f"refined DockQ {results[0]['dockq']} != expected {args.expect_refined}"
            )
    report = {
        "status": "complete",
        "protocol": "fixed_candidate_truncated_clock_sweep_v1",
        "case": args.case,
        "candidate_id": candidate_id,
        "pool_index": pool_index,
        "diffusion_seed": diffusion_seed,
        "candidate_acceptable": bool(candidate.get("acceptable", False)),
        "candidate_geometry": candidate.get("geometry"),
        "identity": identity,
        "endpoint_internal_geometry": endpoint_geometry,
        "schedules": results,
        "generation": generation,
        "config": {
            "index": str(index_path), "index_sha256": sha256_file(index_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "proposal_seed": args.seed,
            "candidate_pool": "Hybrid-192/64 C4M8 binary20",
            "clock_policy": "constant_delta_t_approximately_0.05",
            "native_information_used_for_selection": False,
        },
        "timing_seconds": {"total": time.monotonic() - started},
    }
    atomic_json(report, args.out)


if __name__ == "__main__":
    main()
