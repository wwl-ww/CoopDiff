#!/usr/bin/env python
"""Generate native-free BiTSeed endpoints and project them into training flows."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation_common import resolve_entry_paths  # noqa: E402
from featurize_common import featurize_complex  # noqa: E402
from tricoopdiff.dual_init import (  # noqa: E402
    DUAL_INIT_CACHE_VERSION, validate_endpoint_cache, validate_endpoint_record,
)
from tricoopdiff.endpoint_kinematics import project_endpoint_coupling  # noqa: E402
from tricoopdiff.expert_anchors import (  # noqa: E402
    DualExpertAnchorInitializer, generate_hybrid_candidates,
)
from tricoopdiff.expert_factory import build_diffdock_expert  # noqa: E402
from tricoopdiff.sdpi_flow import SDPIFlow  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--index", required=True, nargs='+',
        help="One or more train/validation indexes; all selected cases share one cache",
    )
    result.add_argument("--out", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--model-dir", default="checkpoints/score_model")
    result.add_argument("--expert-checkpoint", default="best_ema_inference_epoch_model.pt")
    result.add_argument("--cache-dir", default="results/featurize_cache_dual_init")
    result.add_argument("--binary-steps", type=int, default=20)
    result.add_argument("--poses-per-side", type=int, default=4)
    result.add_argument("--conformers", type=int, default=1)
    result.add_argument("--endpoints-per-case", type=int, default=1)
    result.add_argument("--seed", type=int, default=41_000)
    result.add_argument("--limit", type=int)
    result.add_argument("--limit-per-index", type=int)
    result.add_argument("--case-shard-count", type=int, default=1,
                        help="Deterministically split cases by global index")
    result.add_argument("--case-shard-index", type=int, default=0,
                        help="Zero-based case shard assigned to this process")
    result.add_argument("--shard-dir",
                        help="Shared resumable per-case shard directory")
    result.add_argument("--exact-atol", type=float, default=1e-3)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main():
    args = parser().parse_args()
    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be positive")
    if not 0 <= args.case_shard_index < args.case_shard_count:
        raise ValueError("--case-shard-index must be in [0, case-shard-count)")
    index_paths = [Path(value).resolve() for value in args.index]
    indexed_entries = []
    for index_path in index_paths:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        values = index.get("entries", index)
        if args.limit_per_index is not None:
            values = values[:args.limit_per_index]
        indexed_entries.extend(
            (entry, index_path.parent.parent, str(index_path)) for entry in values
        )
    if args.limit is not None:
        indexed_entries = indexed_entries[:args.limit]
    if not indexed_entries:
        raise ValueError("index contains no entries")
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    checkpoint = model_dir / args.expert_checkpoint
    protocol = {
        "version": "online_bitseed_frozen_binary_v1",
        "binary_steps": args.binary_steps,
        "poses_per_side": args.poses_per_side,
        "conformers": args.conformers,
        "selection": "native_free_geometry_penalty_with_sdpi_ood_fallback",
        "frozen_expert_checkpoint": str(checkpoint.resolve()),
        "frozen_expert_checkpoint_sha256": sha256(checkpoint),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = (Path(args.shard_dir) if args.shard_dir
                 else Path(str(destination) + ".cases"))
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model_parameters.yml").open() as handle:
        from argparse import Namespace
        model_args = Namespace(**yaml.full_load(handle))
    expert = build_diffdock_expert(
        str(model_dir), args.device, ckpt=args.expert_checkpoint,
    )
    expert.eval().requires_grad_(False)
    flow = SDPIFlow()
    records = []
    for case_index, (entry, input_root, source_index) in enumerate(indexed_entries):
        if case_index % args.case_shard_count != args.case_shard_index:
            continue
        case = entry["case"]
        shard = shard_dir / f"{case_index:06d}_{case}.pt"
        if shard.is_file():
            cached = torch.load(shard, map_location="cpu", weights_only=False)
            if cached.get("generation_protocol") != protocol or cached.get("case") != case:
                raise ValueError(f"{case}: cached endpoint shard contract mismatch")
            for record in cached.get("entries", ()):
                validate_endpoint_record(record)
            if not cached.get("entries"):
                raise ValueError(f"{case}: cached endpoint shard is empty")
            records.extend(cached["entries"])
            print(f"[cache] SKIP {case}: {len(cached['entries'])} cached endpoints",
                  flush=True)
            continue
        paths = resolve_entry_paths(entry, input_root)
        e3, poi, ligand, _ = featurize_complex(
            case, str(paths["protein1"]), str(paths["protein2"]),
            str(paths["ligand"]), model_args, args.device,
            cache_dir=str((ROOT / args.cache_dir).resolve()),
        )
        initializer = DualExpertAnchorInitializer(
            expert, e3, poi, binary_steps=args.binary_steps,
            proposal_attempts=args.poses_per_side,
            conformer_policy="etkdgv3_random",
        )
        case_seed = args.seed + case_index * 1009
        budget = args.conformers * args.poses_per_side ** 2
        candidates, generation = generate_hybrid_candidates(
            initializer,
            e3["coords"][0], poi["coords"][0], ligand,
            seed=case_seed, conformer_count=args.conformers,
            bitseed_candidate_budget=budget, sdpi_candidate_budget=0,
        )
        bitseed = [candidate for candidate in candidates
                   if candidate["candidate_id"][0] == "bitseed"]
        selectable = bitseed or [candidate for candidate in candidates
                                 if candidate["candidate_id"][0] == "sdpi"]
        selected = sorted(
            selectable,
            key=lambda item: (tuple(item["penalty"]), repr(item["candidate_id"])),
        )[:args.endpoints_per_case]
        case_records = []
        for candidate in selected:
            actual_source = candidate["candidate_id"][0]
            coupling, projected, projection = project_endpoint_coupling(
                ligand, poi["coords"], candidate["ligand"], candidate["poi"],
                no_torsion=initializer.no_torsion, exact_atol=args.exact_atol,
            )
            ligand_one, poi_one, body_one, targets = flow.path(
                ligand, poi["coords"], coupling,
                torch.ones(1, device=ligand["coords"].device),
            )
            if not all(torch.isfinite(value).all() for branch in targets.values()
                       for value in (branch.values() if isinstance(branch, dict) else (branch,))):
                raise FloatingPointError(f"{case}: non-finite projected velocity target")
            if (not np.allclose(cpu_array(ligand_one[0]), projected["ligand"], atol=1e-5)
                    or not np.allclose(cpu_array(poi_one[0]), projected["poi"], atol=1e-5)):
                raise AssertionError(f"{case}: projected endpoint failed t=1 round-trip")
            conformer_index = candidate["conformer_index"]
            if conformer_index is not None:
                conformer_index = int(conformer_index)
            case_records.append({
                "case": case,
                "source_index": source_index,
                "actual_source": actual_source,
                "candidate_id": candidate["candidate_id"],
                "conformer_seed": (
                    initializer._conformer_seed(case_seed, conformer_index)
                    if conformer_index is not None else None
                ),
                "binary_pose_ids": list(candidate["attempt"]),
                "fallback": candidate.get("fallback"),
                "native_used_in_candidate_generation": False,
                "native_used_in_candidate_selection": False,
                "native_used_for_training_projection": True,
                "endpoint": {
                    "ligand": cpu_array(candidate["ligand"][0]),
                    "poi": cpu_array(candidate["poi"][0]),
                    "body_frame": cpu_array(candidate["rotation"][0]),
                },
                "training_endpoint": {
                    "ligand": cpu_array(ligand_one[0]),
                    "poi": cpu_array(poi_one[0]),
                    "body_frame": cpu_array(body_one[0]),
                },
                "coupling": {name: cpu_array(value) for name, value in coupling.items()},
                "projection": projection,
                "generation": {
                    "case_seed": case_seed,
                    "conformer": next((item for item in generation["successful_conformers"]
                                       if item["conformer_index"] == conformer_index), None),
                    "fallback_events": generation["fallback_events"],
                    "geometry": candidate["geometry"],
                    "fit_rmsd": candidate["fit_rmsd"],
                },
            })
        for record in case_records:
            validate_endpoint_record(record)
        atomic_torch_save({
            "case": case, "generation_protocol": protocol,
            "entries": case_records,
        }, shard)
        records.extend(case_records)
        print(f"[cache] {case}: {len(case_records)} endpoints", flush=True)
    payload = {
        "cache_version": DUAL_INIT_CACHE_VERSION,
        "indexes": [str(path) for path in index_paths],
        "generation_protocol": protocol,
        "case_partition": {
            "count": args.case_shard_count,
            "index": args.case_shard_index,
            "total_cases": len(indexed_entries),
        },
        "entries": records,
    }
    validate_endpoint_cache(payload)
    atomic_torch_save(payload, destination)
    print(f"[cache] wrote {len(records)} endpoints to {destination}", flush=True)


if __name__ == "__main__":
    main()
