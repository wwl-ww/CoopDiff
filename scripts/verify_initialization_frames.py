#!/usr/bin/env python
"""Validate shared-frame featurization and ternary initialization on one case."""

import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]

from featurize_common import (  # noqa: E402
    _FEATURIZE_CACHE_VERSION,
    _true_relative_offsets,
    featurize_complex,
)
from tricoopdiff.diffusion import MultiRateDiffusion, TernaryInitializer  # noqa: E402
from tricoopdiff.graphs import transform_ligand_to_poi_frame  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="5T35_D_A_759")
    parser.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    parser.add_argument("--model-dir", default="checkpoints/score_model")
    parser.add_argument("--cache-dir", default="results/featurize_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--out", default="results/initialization_frame_check.json")
    return parser


def _paths(args):
    case_dir = (PROJECT_ROOT / args.protac_dir / args.case).resolve()
    ligand = case_dir / "ligand.sdf"
    if not ligand.exists():
        ligand = case_dir / "ligand.pdb"
    paths = (case_dir / "protein1.pdb", case_dir / "protein2.pdb", ligand)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing case inputs: {missing}")
    return tuple(str(path) for path in paths)


def _error(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected).item())


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protein1, protein2, ligand_path = _paths(args)
    model_config = PROJECT_ROOT / args.model_dir / "model_parameters.yml"
    with model_config.open() as handle:
        model_args = Namespace(**yaml.full_load(handle))

    cache_dir = (PROJECT_ROOT / args.cache_dir).resolve()
    e3, poi, ligand, _ = featurize_complex(
        args.case,
        protein1,
        protein2,
        ligand_path,
        model_args,
        args.device,
        cache_dir=str(cache_dir),
    )
    true_poi_offset, true_ligand_offset = _true_relative_offsets(
        protein1, protein2, ligand_path
    )
    true_poi_offset = torch.as_tensor(true_poi_offset, device=args.device)
    true_ligand_offset = torch.as_tensor(true_ligand_offset, device=args.device)

    e3_center = e3["coords"][0].mean(0)
    poi_center = poi["coords"][0].mean(0)
    ligand_center = ligand["coords"][0].mean(0)
    poi_offset_error = _error(poi_center - e3_center, true_poi_offset)
    ligand_offset_error = _error(ligand_center - e3_center, true_ligand_offset)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=args.device))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    ligand_poi = transform_ligand_to_poi_frame(
        ligand["coords"][0], poi_center - e3_center, rotation
    )
    ligand_roundtrip = ligand_poi @ rotation.T + (poi_center - e3_center)
    roundtrip_error = float((ligand_roundtrip - ligand["coords"][0]).abs().max().item())

    initializer = TernaryInitializer()
    poi_initialized, poi_rotation = initializer.initialize_poi(
        e3_center[None], poi["coords"][0], return_rotation=True
    )
    poi_shell_radius = float(
        torch.linalg.vector_norm(poi_initialized.mean(1)[0] - e3_center).item()
    )
    orthogonality_error = float(
        (poi_rotation @ poi_rotation.transpose(-1, -2) - torch.eye(3, device=args.device))
        .abs()
        .max()
        .item()
    )
    rotation_det = float(torch.linalg.det(poi_rotation[0]).item())

    diffusion = MultiRateDiffusion()
    ligand_initialized = initializer.initialize_protac(
        e3_center[None],
        ligand["coords"][0],
        lig_edge_index=ligand.get("edge_index"),
        lig_edge_mask=ligand.get("edge_mask"),
        lig_mask_rotate=ligand.get("mask_rotate"),
        tr_sigma_max=diffusion.tr_sigma_max,
    )
    ligand_init_finite = bool(torch.isfinite(ligand_initialized).all().item())

    cache_path = cache_dir / f"{args.case.replace('/', '_')}.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_version = cached.get("cache_version")
    checks = {
        "cache_version": cache_version == _FEATURIZE_CACHE_VERSION,
        "e3_centered": float(e3_center.norm().item()) <= args.atol,
        "poi_offset": poi_offset_error <= args.atol,
        "ligand_offset": ligand_offset_error <= args.atol,
        "frame_roundtrip": roundtrip_error <= args.atol,
        "poi_shell": initializer.poi_shell_min <= poi_shell_radius <= initializer.poi_shell_max,
        "poi_rotation": orthogonality_error <= args.atol and abs(rotation_det - 1.0) <= args.atol,
        "ligand_init_finite": ligand_init_finite,
    }
    report = {
        "case": args.case,
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "cache_version": cache_version,
            "expected_cache_version": _FEATURIZE_CACHE_VERSION,
            "e3_center_norm": float(e3_center.norm().item()),
            "poi_offset_error": poi_offset_error,
            "ligand_offset_error": ligand_offset_error,
            "frame_roundtrip_max_error": roundtrip_error,
            "poi_shell_radius": poi_shell_radius,
            "poi_rotation_orthogonality_error": orthogonality_error,
            "poi_rotation_determinant": rotation_det,
        },
    }
    output_path = PROJECT_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
