#!/usr/bin/env python
"""
TriCoopDiff MVP evaluation entry point.

Samples a conformational ensemble for a single ternary complex and scores the
PROTAC (E3-L) pose against the native structure embedded in the featurized data.

Pipeline:
  1. Featurize the E3-L and POI-L subgraphs (``featurize_common``, on-the-fly
     ESM2-650M embeddings).
  2. Build the frozen DiffDock-L expert + TriFuse + TriCoopDiff +
     InterfaceDiffusion + TernaryInitializer.
  3. Optionally load a trained TriFuse checkpoint (``--trifuse-ckpt``).
  4. ``evaluate_model``: run ``--n-walkers`` reverse-diffusion trajectories and
     report PROTAC RMSD / pocket distance / clash rate + ensemble diversity.

The zero-FiLM baseline (no ``--trifuse-ckpt``) is the Go/No-Go reference:
TriFuse starts at FiLM-zero-init, so the sampled ensemble should reproduce the
native DiffDock-L two-body PROTAC distribution and its RMSD.

Output: a JSON report printed to stdout and dumped to ``--out``.

Example:
    python scripts/evaluate_entry.py --case 5T35_D_A_759 --n-walkers 32 \
        --out results/eval_5T35.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Make the project root + this script's dir importable (not pip-installed).
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import diffdock_compat FIRST — patches torch_cluster/torch_scatter before
# the vendored cg_model and the vendored process_mols bind those symbols at
# import time.
import tricoopdiff.diffdock_compat  # noqa: F401

import torch
import yaml
from argparse import Namespace

from featurize_common import featurize_complex
from evaluation_common import (
    load_index_entry,
    resolve_entry_paths,
    validate_checkpoint_payload,
)

from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.trifuse import TriFuse
from tricoopdiff.model import TriCoopDiff
from tricoopdiff.diffusion import InterfaceDiffusion, TernaryInitializer
from tricoopdiff.evaluate import evaluate_model

# DiffDock's SO(3)/torus cache lookup resolves relative to CWD; cache files live
# in the project root. Chdir there so score lookups find them.
os.chdir(_PROJECT_ROOT)


def build_parser():
    p = argparse.ArgumentParser(description="TriCoopDiff MVP evaluation entry point")
    p.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    p.add_argument("--index", default=None,
                   help="Official ternary JSON index containing --case")
    p.add_argument("--native-root", default="data/TernaryDB_release/pdbs",
                   help="Root containing <case>/gt_complex.pdb for indexed data")
    p.add_argument("--model-dir", default="checkpoints/score_model")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--case", default="5T35_D_A_759",
                   help="Ternary complex case name (directory under --protac-dir)")
    p.add_argument("--trifuse-ckpt", default=None,
                   help="Version-16 TriFuse checkpoint")
    p.add_argument(
        "--nan-clamp", type=float, default=100.0,
        help="Activation clamp for an explicit zero-init baseline",
    )
    p.add_argument("--allow-zero-init", action="store_true",
                   help="Explicitly allow evaluation without trained weights")
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-start-layer", type=int, default=0)
    p.add_argument("--lora-early-layers", type=int, default=4)
    p.add_argument("--lora-early-rank", type=int, default=2)
    p.add_argument("--lora-early-alpha", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-walkers", type=int, default=32,
                   help="Number of independent reverse-diffusion trajectories")
    p.add_argument("--steps", type=int, default=None,
                   help="Reverse-diffusion steps (default: diffusion.inference_steps)")
    p.add_argument("--cache-dir", default="results/featurize_cache",
                   help="Dir caching featurized complexes across runs "
                        "(shared with train_entry.py)")
    p.add_argument("--out", default=None,
                   help="Path to dump the JSON report (default: stdout only)")
    p.add_argument("--debug-trace", action="store_true",
                   help="Also emit a single-walker per-step trajectory trace for parity debugging")
    p.add_argument("--debug-seed", type=int, default=0,
                   help="Seed used by --debug-trace single-walker trajectory")
    p.add_argument(
        "--freeze-poi",
        action="store_true",
        help="Keep the POI fixed and disable SE3 reverse updates for strict E3-L parity evaluation.",
    )
    p.add_argument("--init-mode", choices=["dual_native"], default="dual_native",
                   help="Shared full-ligand dual binary receptor-size-adaptive prior")
    p.add_argument("--t-anchor", type=float, choices=[1.0], default=None,
                   help="dual_native starts at t=1; other values are rejected")
    return p


def main():
    args = build_parser().parse_args()
    if args.nan_clamp <= 0:
        raise SystemExit("FATAL: --nan-clamp must be positive")
    if args.trifuse_ckpt is None:
        os.environ["NAN_CLAMP"] = str(args.nan_clamp)

    if args.trifuse_ckpt is None and not args.allow_zero_init:
        print("FATAL: --trifuse-ckpt is required; use --allow-zero-init only "
              "for an explicit baseline", flush=True)
        sys.exit(2)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda requested but CUDA is unavailable", flush=True)
        sys.exit(1)
    device = args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Resolve either an official index entry or a legacy case directory.
    if args.index:
        index_path = Path(args.index)
        if not index_path.is_absolute():
            index_path = _PROJECT_ROOT / index_path
        try:
            entry = load_index_entry(index_path, args.case)
            paths = resolve_entry_paths(entry, _PROJECT_ROOT)
        except (OSError, KeyError, ValueError) as error:
            print(f"FATAL: invalid indexed case: {error}", flush=True)
            sys.exit(2)
        protein1 = str(paths["protein1"])
        protein2 = str(paths["protein2"])
        ligand_path = str(paths["ligand"])
        case_dir = (_PROJECT_ROOT / args.native_root / args.case).resolve()
    else:
        case_dir = _PROJECT_ROOT / args.protac_dir / args.case
        if not case_dir.exists():
            print(f"FATAL: case directory not found: {case_dir}", flush=True)
            sys.exit(1)
        protein1 = str(case_dir / "protein1.pdb")
        protein2 = str(case_dir / "protein2.pdb")
        _sdf = case_dir / "ligand.sdf"
        _pdb = case_dir / "ligand.pdb"
        ligand_path = str(_sdf if _sdf.exists() else _pdb)
        paths = {
            "protein1": Path(protein1),
            "protein2": Path(protein2),
            "ligand": Path(ligand_path),
        }

    print("=" * 72, flush=True)
    print(f"TriCoopDiff MVP evaluation — case {args.case}", flush=True)
    print(f"  device={device}  model-dir={args.model_dir}", flush=True)
    print(f"  n_walkers={args.n_walkers}  steps={args.steps}", flush=True)
    print(f"  trifuse_ckpt={args.trifuse_ckpt or '(zero-init baseline)'}", flush=True)
    print(f"  activation_clamp={os.environ.get('NAN_CLAMP', 'from checkpoint')}", flush=True)
    print("=" * 72, flush=True)

    # Production single-case and indexed evaluation intentionally share one
    # runtime. Keep the code below only for the explicit zero-init diagnostic.
    if args.trifuse_ckpt is not None:
        from evaluation_runner import EvaluationRunner, write_report_atomic

        native_pdb = case_dir / "gt_complex.pdb"
        if not native_pdb.exists():
            native_pdb = case_dir / "complex.pdb"
        try:
            runner = EvaluationRunner(
                _PROJECT_ROOT, checkpoint=args.trifuse_ckpt,
                device=args.device, model_dir=args.model_dir,
                expert_checkpoint=args.ckpt,
            )
            report = runner.evaluate_case(
                args.case, paths, cache_dir=_PROJECT_ROOT / args.cache_dir,
                native_complex_pdb=native_pdb if native_pdb.is_file() else None,
                n_walkers=args.n_walkers, steps=args.steps, seed=args.seed,
                freeze_poi=args.freeze_poi, init_mode=args.init_mode,
                t_anchor=args.t_anchor, debug_trace=args.debug_trace,
                debug_seed=args.debug_seed,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FATAL: trained evaluation failed: {error}", flush=True)
            sys.exit(2)
        print(json.dumps(report, indent=2), flush=True)
        if args.out is not None:
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = _PROJECT_ROOT / out_path
            write_report_atomic(report, out_path)
            print(f"\n[saved] {out_path}", flush=True)
        return

    # --- Load score-model args (featurization hyperparams) --------------------
    with open(_PROJECT_ROOT / args.model_dir / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    print(f"[cfg] ns={score_model_args.ns} nv={score_model_args.nv} "
          f"num_conv_layers={score_model_args.num_conv_layers}", flush=True)

    trained_ckpt = None
    if args.trifuse_ckpt is not None:
        ckpt_path = _PROJECT_ROOT / args.trifuse_ckpt
        if not ckpt_path.exists():
            print(f"FATAL: --trifuse-ckpt not found: {ckpt_path}", flush=True)
            sys.exit(1)
        trained_ckpt = torch.load(ckpt_path, map_location=device)
        try:
            validate_checkpoint_payload(trained_ckpt)
        except ValueError as error:
            print(f"FATAL: invalid trained checkpoint: {error}", flush=True)
            sys.exit(2)

    # --- Build the frozen expert ---------------------------------------------
    print("[expert] building frozen DiffDock-L expert...", flush=True)
    expert = build_diffdock_expert(
        model_dir=str(_PROJECT_ROOT / args.model_dir),
        device=device,
        ckpt=args.ckpt,
        num_conv_layers=score_model_args.num_conv_layers,
    )
    if isinstance(trained_ckpt, dict) and trained_ckpt.get("expert_lora_state_dict"):
        from tricoopdiff.lora import inject_expert_lora
        config = trained_ckpt.get("expert_lora_config") or {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "start_layer": args.lora_start_layer,
            "early_layers": args.lora_early_layers,
            "early_rank": args.lora_early_rank,
            "early_alpha": args.lora_early_alpha,
        }
        count = inject_expert_lora(expert, **config)
        print(f"[expert] injected {count} LoRA linear layers", flush=True)
    num_conv_layers = expert.num_conv_layers
    print(f"[expert] loaded (num_conv_layers={num_conv_layers}, ns={expert.ns})",
          flush=True)

    # --- Featurize the ternary complex (cached) ------------------------------
    print("[data] featurizing ternary complex (ESM2 on the fly)...", flush=True)
    cache_dir = str(_PROJECT_ROOT / args.cache_dir)
    e3_data, poi_data, ligand_data, d_rec = featurize_complex(
        args.case, protein1, protein2, ligand_path, score_model_args, device,
        cache_dir=cache_dir,
    )
    n_lig = ligand_data['features'].shape[1]
    n_rot = int(ligand_data['edge_mask'].sum().item())
    print(f"[data] receptor feat dim={d_rec}  PROTAC atoms={n_lig}  "
          f"rotatable bonds={n_rot}", flush=True)

    # --- Build TriFuse + TriCoopDiff + diffusion + initializer --------------
    trifuse = TriFuse(
        node_dim=128,
        input_dim=d_rec,
        num_conv_layers_per_expert=num_conv_layers,
        film_dim=expert.ns,
    )
    model = TriCoopDiff(
        frozen_expert=expert,
        trifuse=trifuse,
        num_conv_layers=num_conv_layers,
        use_film=True,
    ).to(device)
    diffusion = InterfaceDiffusion()
    initializer = TernaryInitializer(
        no_torsion=getattr(score_model_args, "no_torsion", False),
    )
    n_params = sum(p.numel() for p in trifuse.parameters())
    print(f"[model] TriFuse trainable params = {n_params:,}", flush=True)

    # --- Optional: load a trained TriFuse checkpoint ------------------------
    if args.trifuse_ckpt is not None:
        if "trifuse_state_dict" in trained_ckpt:
            model.trifuse.load_state_dict(trained_ckpt["trifuse_state_dict"])
            if trained_ckpt.get("expert_lora_state_dict"):
                from tricoopdiff.lora import load_expert_lora_state_dict
                load_expert_lora_state_dict(
                    model.expert, trained_ckpt["expert_lora_state_dict"],
                )
        else:
            model.trifuse.load_state_dict(trained_ckpt)
        print(f"[model] loaded TriFuse weights from {ckpt_path}", flush=True)

    # --- Evaluate ------------------------------------------------------------
    print("[eval] sampling ensemble...", flush=True)
    # Native ternary complex PDB (gt_complex.pdb, else complex.pdb) supplies the
    # ground-truth E3/POI/PROTAC in one frame for the frame-invariant
    # PROTAC-POI interface metric (compute_poi_dockq). Optional; POI keys are
    # None when the file is absent.
    _native_pdb = case_dir / "gt_complex.pdb"
    if not _native_pdb.exists():
        _native_pdb = case_dir / "complex.pdb"
    native_complex_pdb = str(_native_pdb) if _native_pdb.exists() else None

    report = evaluate_model(
        model, e3_data, poi_data, ligand_data, diffusion,
        initializer=initializer, device=device,
        n_walkers=args.n_walkers, steps=args.steps,
        seed=args.seed,
        native_complex_pdb=native_complex_pdb,
        debug_trace=args.debug_trace, debug_seed=args.debug_seed,
        freeze_poi=args.freeze_poi,
        init_mode=args.init_mode,
        t_anchor=args.t_anchor,
        deepternary_dockq_paths={
            "protein1_pdb": protein1,
            "protein2_pdb": protein2,
            "ligand_pdb": ligand_path,
            "deepternary_root": str(_PROJECT_ROOT / "DeepTernary"),
        },
    )
    report["case"] = args.case
    report["run_config"] = {
        "seed": args.seed,
        "n_walkers": args.n_walkers,
        "steps": args.steps if args.steps is not None else diffusion.inference_steps,
        "freeze_poi": args.freeze_poi,
        "init_mode": args.init_mode,
        "initialization_version": "dual_native_v1",
        "t_anchor": args.t_anchor if args.t_anchor is not None else 1.0,
        "trifuse_checkpoint": args.trifuse_ckpt,
        "activation_clamp": args.nan_clamp,
        "expert_model_dir": args.model_dir,
        "expert_checkpoint": args.ckpt,
    }

    print("\n" + "=" * 72, flush=True)
    print("REPORT", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(report, indent=2), flush=True)

    if args.out is not None:
        out_path = _PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
