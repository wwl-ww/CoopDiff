#!/usr/bin/env python
"""
Compare vendored native DiffDock forward vs TriCoopDiff's inline
`_expert_forward_with_film` on the exact same zero-cross step-0 graph.

This is a narrow parity probe: it builds the same step-0 subgraph that
`evaluate_entry.py --debug-trace` would feed into the expert, then runs:
  1. native `expert.model(...)`
  2. inline `_expert_forward_with_film(..., zero film)`
under the same `NAN_TRACE` instrumentation.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tricoopdiff.diffdock_compat  # noqa: F401

import numpy as np
import torch
import yaml
from argparse import Namespace

from featurize_common import featurize_complex
from tricoopdiff.diffusion import MultiRateDiffusion, TernaryInitializer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.graphs import crop_receptor_to_ligand, build_subgraph_e3l, build_subgraph_pol, transform_ligand_to_poi_frame
from tricoopdiff.model import _expert_forward_with_film, _set_time_embeddings

os.chdir(_PROJECT_ROOT)


def build_parser():
    p = argparse.ArgumentParser(description="Probe native vs inline expert forward on step-0 graph")
    p.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    p.add_argument("--model-dir", default="checkpoints/score_model")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--case", default="5T35_D_A_759")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--cache-dir", default="results/featurize_cache")
    p.add_argument("--branch", choices=["sub1", "sub2"], default="sub1")
    p.add_argument("--init-mode", choices=["native", "coupled5a"], default="coupled5a")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--nan-trace", action="store_true",
                   help="Print native and inline per-layer finite/max diagnostics")
    p.add_argument("--nan-clamp", default="10000",
                   help="Per-layer node activation clamp (default: runtime 10000)")
    p.add_argument("--out", default=None)
    return p


def _zero_film_params(num_layers, ns, device):
    zeros = []
    for _ in range(num_layers):
        z = torch.zeros((1, ns), device=device)
        zeros.append((z, z.clone()))
    return zeros


def _tensor_stats(x):
    if x.numel() == 0:
        return {"shape": list(x.shape), "finite": True, "absmax": 0.0}
    finite = bool(torch.isfinite(x).all().item())
    absmax = float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
    return {"shape": list(x.shape), "finite": finite, "absmax": absmax}


def _compare_tensors(native, inline, atol, rtol):
    """Return elementwise parity diagnostics without hiding NaN mismatches."""
    same_shape = native.shape == inline.shape
    if not same_shape:
        return {
            "pass": False,
            "same_shape": False,
            "native_shape": list(native.shape),
            "inline_shape": list(inline.shape),
        }

    finite = bool(torch.isfinite(native).all() and torch.isfinite(inline).all())
    delta = (native - inline).abs()
    scale = torch.maximum(native.abs(), inline.abs()).clamp_min(atol)
    return {
        "pass": finite and bool(torch.allclose(native, inline, atol=atol, rtol=rtol)),
        "same_shape": True,
        "finite": finite,
        "max_abs_error": float(torch.nan_to_num(delta, nan=float("inf")).max().item()),
        "max_rel_error": float(torch.nan_to_num(delta / scale, nan=float("inf")).max().item()),
    }


def _build_step0_subgraph(args, expert):
    protac_dir = _PROJECT_ROOT / args.protac_dir
    case_dir = protac_dir / args.case
    protein1 = str(case_dir / "protein1.pdb")
    protein2 = str(case_dir / "protein2.pdb")
    _sdf = case_dir / "ligand.sdf"
    _pdb = case_dir / "ligand.pdb"
    ligand_path = str(_sdf if _sdf.exists() else _pdb)

    with open(_PROJECT_ROOT / args.model_dir / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    e3_data, poi_data, ligand_data, _ = featurize_complex(
        args.case, protein1, protein2, ligand_path, score_model_args, args.device,
        cache_dir=str(_PROJECT_ROOT / args.cache_dir),
    )

    diffusion = MultiRateDiffusion()
    initializer = TernaryInitializer()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ligand_template = ligand_data["coords"][0].detach()
    poi_template = poi_data["coords"][0].detach()
    e3_coords = e3_data["coords"][0].detach()
    e3_center = e3_coords.mean(dim=0, keepdim=True)

    ligand_pos = initializer.initialize_protac(
        e3_center,
        ligand_template,
        lig_edge_index=ligand_data.get("edge_index"),
        lig_edge_mask=ligand_data.get("edge_mask"),
        lig_mask_rotate=ligand_data.get("mask_rotate"),
        tr_sigma_max=diffusion.tr_sigma_max,
        init_mode=args.init_mode,
    )
    poi_pos, rot_init = initializer.initialize_poi(
        e3_center, poi_template, return_rotation=True, init_mode=args.init_mode
    )

    schedule = diffusion.get_inference_schedule(args.steps)
    t_tr = torch.tensor([schedule["tr_schedule"][0]], device=args.device, dtype=torch.float32)
    tr_sigma, _, _ = diffusion.t_to_sigma(t_tr, t_tr, t_tr)
    receptor_crop_cutoff = tr_sigma * 3 + 20 if bool(getattr(expert.model, "dynamic_max_cross", False)) else None

    ligand_coords_e3 = ligand_pos[0]
    e3_center0 = e3_data["coords"][0].mean(dim=0)
    poi_center0 = poi_pos[0].mean(dim=0)
    t_e3_to_poi = poi_center0 - e3_center0
    # initialize_poi applies ``template @ rot_init.T``; rot_init is therefore
    # already the inverse/body-frame map expected by model.forward.
    r_e3_to_poi = rot_init[0]
    ligand_coords_poi = transform_ligand_to_poi_frame(
        ligand_coords_e3, t_e3_to_poi, r_e3_to_poi
    )
    # Use the current initialized POI pose, exactly as sample_single() passes it
    # into model.forward(). Using the featurized template here while deriving the
    # translation from `poi_pos` creates a hybrid graph and leaves the reported
    # POI local COM far from the origin.
    poi_coords_poi = transform_ligand_to_poi_frame(
        poi_pos[0],
        t_e3_to_poi,
        r_e3_to_poi,
    )

    use_native_step_crop = bool(getattr(expert.model, "dynamic_max_cross", False))
    crop_cutoff_b = receptor_crop_cutoff[0] if receptor_crop_cutoff is not None and use_native_step_crop else None
    crop_e3 = crop_cutoff_b if use_native_step_crop else None
    crop_poi = crop_cutoff_b if use_native_step_crop else None

    e3_crop = crop_receptor_to_ligand(
        receptor_coords=e3_data["coords"][0],
        receptor_features=e3_data["features"][0],
        ligand_coords=ligand_coords_e3,
        rec_edge_index=e3_data.get("edge_index"),
        cutoff=crop_e3,
        min_residues=1,
        min_residues_with_edges=8,
        require_internal_edges=True,
    )
    poi_crop = crop_receptor_to_ligand(
        receptor_coords=poi_coords_poi,
        receptor_features=poi_data["features"][0],
        ligand_coords=ligand_coords_poi,
        rec_edge_index=poi_data.get("edge_index"),
        cutoff=crop_poi,
        min_residues=1,
        min_residues_with_edges=8,
        require_internal_edges=True,
    )

    sub1 = build_subgraph_e3l(
        e3_coords=e3_crop["coords"],
        e3_features=e3_crop["features"],
        ligand_coords=ligand_coords_e3,
        ligand_features=ligand_data["features"][0],
        rec_edge_index=e3_crop["edge_index"],
        lig_edge_index=ligand_data.get("edge_index"),
        lig_edge_attr=ligand_data.get("edge_attr"),
        lig_edge_mask=ligand_data.get("edge_mask"),
        lig_mask_rotate=ligand_data.get("mask_rotate"),
    )
    sub2 = build_subgraph_pol(
        poi_coords=poi_crop["coords"],
        poi_features=poi_crop["features"],
        ligand_coords_poi_frame=ligand_coords_poi,
        ligand_features=ligand_data["features"][0],
        rec_edge_index=poi_crop["edge_index"],
        lig_edge_index=ligand_data.get("edge_index"),
        lig_edge_attr=ligand_data.get("edge_attr"),
        lig_edge_mask=ligand_data.get("edge_mask"),
        lig_mask_rotate=ligand_data.get("mask_rotate"),
    )

    return {
        "t": t_tr,
        "sub1": sub1,
        "sub2": sub2,
        "meta": {
            "receptor_crop_cutoff": float(receptor_crop_cutoff[0].item()) if receptor_crop_cutoff is not None else None,
            "e3_crop_residues": int(e3_crop["coords"].shape[0]),
            "poi_crop_residues": int(poi_crop["coords"].shape[0]),
            "ligand_com_e3": ligand_coords_e3.mean(dim=0).detach().cpu().tolist(),
            "ligand_com_poi": ligand_coords_poi.mean(dim=0).detach().cpu().tolist(),
            "poi_com_poi": poi_coords_poi.mean(dim=0).detach().cpu().tolist(),
            "poi_local_com_norm": float(poi_coords_poi.mean(dim=0).norm().item()),
        }
    }


def main():
    args = build_parser().parse_args()
    # The canonical SLURM evaluation entry sets this same guard. Keep the
    # standalone parity probe on the production numerical path by default.
    if args.nan_clamp.lower() not in {"none", "off", "0"}:
        os.environ["NAN_CLAMP"] = args.nan_clamp
    else:
        os.environ.pop("NAN_CLAMP", None)
    if args.nan_trace:
        os.environ["NAN_TRACE"] = "1"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda requested but CUDA is unavailable", flush=True)
        sys.exit(1)

    expert = build_diffdock_expert(
        model_dir=str(_PROJECT_ROOT / args.model_dir),
        device=args.device,
        ckpt=args.ckpt,
    )
    bundle = _build_step0_subgraph(args, expert)
    data = copy.deepcopy(bundle[args.branch])
    t = bundle["t"]
    meta = bundle["meta"]

    # Native forward on the exact same graph.
    native_data = copy.deepcopy(data)
    _set_time_embeddings(native_data, t, expert.model)
    with torch.no_grad():
        native_out = expert.model(native_data)

    # Inline forward with explicit zero FiLM.
    inline_data = copy.deepcopy(data)
    debug_probe = {}
    with torch.no_grad():
        inline_scores = _expert_forward_with_film(
            expert,
            inline_data,
            t,
            _zero_film_params(expert.num_conv_layers, expert.ns, t.device),
            trace_prefix=f"inline/{args.branch}",
            debug_probe=debug_probe,
        )

    native_scores = {
        "tr": native_out[0],
        "rot": native_out[1],
        "tor": native_out[2],
    }
    inline_outputs = {
        "tr": inline_scores["tr_score"],
        "rot": inline_scores["rot_score"],
        "tor": inline_scores["tor_score"],
    }
    comparisons = {
        key: _compare_tensors(native_scores[key], inline_outputs[key], args.atol, args.rtol)
        for key in native_scores
    }
    parity_pass = all(item["pass"] for item in comparisons.values())

    result = {
        "case": args.case,
        "branch": args.branch,
        "seed": args.seed,
        "steps": args.steps,
        "init_mode": args.init_mode,
        "meta": meta,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "nan_clamp": os.environ.get("NAN_CLAMP"),
        "parity_pass": parity_pass,
        "comparison": comparisons,
        "native": {key: _tensor_stats(value) for key, value in native_scores.items()},
        "inline": {key: _tensor_stats(value) for key, value in inline_outputs.items()},
        "debug_probe": debug_probe,
    }

    print(json.dumps(result, indent=2), flush=True)
    if args.out is not None:
        out_path = _PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    if not parity_pass:
        sys.exit(2)


if __name__ == "__main__":
    main()
