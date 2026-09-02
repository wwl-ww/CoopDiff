#!/usr/bin/env python
"""
TriCoopDiff MVP evaluation entry point.

Samples a conformational ensemble for a single ternary complex and scores the
PROTAC (E3-L) pose against the native structure embedded in the featurized data.

Pipeline:
  1. Featurize the E3-L and POI-L subgraphs (``featurize_common``, on-the-fly
     ESM2-650M embeddings).
  2. Build the frozen DiffDock-L expert + TriBrain + TriCoopDiff +
     MultiRateDiffusion + TernaryInitializer.
  3. Optionally load a trained TriBrain checkpoint (``--tribrain-ckpt``).
  4. ``evaluate_model``: run ``--n-walkers`` reverse-diffusion trajectories and
     report PROTAC RMSD / pocket distance / clash rate + ensemble diversity.

The zero-FiLM baseline (no ``--tribrain-ckpt``) is the Go/No-Go reference:
TriBrain starts at FiLM-zero-init, so the sampled ensemble should reproduce the
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

from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.tribrain import TriBrain
from tricoopdiff.model import TriCoopDiff
from tricoopdiff.diffusion import MultiRateDiffusion, TernaryInitializer
from tricoopdiff.evaluate import evaluate_model

# DiffDock's SO(3)/torus cache lookup resolves relative to CWD; cache files live
# in the project root. Chdir there so score lookups find them.
os.chdir(_PROJECT_ROOT)


def build_parser():
    p = argparse.ArgumentParser(description="TriCoopDiff MVP evaluation entry point")
    p.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    p.add_argument("--model-dir", default="checkpoints/score_model")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--case", default="5T35_D_A_759",
                   help="Ternary complex case name (directory under --protac-dir)")
    p.add_argument("--tribrain-ckpt", default=None,
                   help="Optional trained TriBrain checkpoint (default: zero-init "
                        "baseline = no loading)")
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
    p.add_argument("--init-mode", choices=["native", "coupled5a"], default="native",
                   help="Ternary initialization prior")
    return p


def main():
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda requested but CUDA is unavailable", flush=True)
        sys.exit(1)
    device = args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Resolve the case directory + input files.
    protac_dir = _PROJECT_ROOT / args.protac_dir
    case_dir = protac_dir / args.case
    if not case_dir.exists():
        print(f"FATAL: case directory not found: {case_dir}", flush=True)
        sys.exit(1)

    protein1 = str(case_dir / "protein1.pdb")   # E3
    protein2 = str(case_dir / "protein2.pdb")   # POI
    _sdf = case_dir / "ligand.sdf"
    _pdb = case_dir / "ligand.pdb"
    ligand_path = str(_sdf if _sdf.exists() else _pdb)

    print("=" * 72, flush=True)
    print(f"TriCoopDiff MVP evaluation — case {args.case}", flush=True)
    print(f"  device={device}  model-dir={args.model_dir}", flush=True)
    print(f"  n_walkers={args.n_walkers}  steps={args.steps}", flush=True)
    print(f"  tribrain_ckpt={args.tribrain_ckpt or '(zero-init baseline)'}", flush=True)
    print("=" * 72, flush=True)

    # --- Load score-model args (featurization hyperparams) --------------------
    with open(_PROJECT_ROOT / args.model_dir / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    print(f"[cfg] ns={score_model_args.ns} nv={score_model_args.nv} "
          f"num_conv_layers={score_model_args.num_conv_layers}", flush=True)

    # --- Build the frozen expert ---------------------------------------------
    print("[expert] building frozen DiffDock-L expert...", flush=True)
    expert = build_diffdock_expert(
        model_dir=str(_PROJECT_ROOT / args.model_dir),
        device=device,
        ckpt=args.ckpt,
        num_conv_layers=score_model_args.num_conv_layers,
    )
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

    # --- Build TriBrain + TriCoopDiff + diffusion + initializer --------------
    tribrain = TriBrain(
        node_dim=128,
        input_dim=d_rec,
        num_conv_layers_per_expert=num_conv_layers,
        film_dim=expert.ns,
    )
    model = TriCoopDiff(
        frozen_expert=expert,
        tribrain=tribrain,
        num_conv_layers=num_conv_layers,
        use_film=True,
        # use_learned_merge defaults True: residual-gated merge (zero-init gate →
        # s_q = s_q1), so the zero-FiLM baseline reproduces pure E3-L, not the
        # double-counted s_q1 + s_q2.
    ).to(device)
    diffusion = MultiRateDiffusion()
    initializer = TernaryInitializer()
    n_params = sum(p.numel() for p in tribrain.parameters())
    print(f"[model] TriBrain trainable params = {n_params:,}", flush=True)

    # --- Optional: load a trained TriBrain checkpoint ------------------------
    if args.tribrain_ckpt is not None:
        ckpt_path = _PROJECT_ROOT / args.tribrain_ckpt
        if not ckpt_path.exists():
            print(f"FATAL: --tribrain-ckpt not found: {ckpt_path}", flush=True)
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        if "tribrain_state_dict" in ckpt:
            model.tribrain.load_state_dict(ckpt["tribrain_state_dict"])
        else:
            model.tribrain.load_state_dict(ckpt)
        print(f"[model] loaded TriBrain weights from {ckpt_path}", flush=True)

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
        native_complex_pdb=native_complex_pdb,
        debug_trace=args.debug_trace, debug_seed=args.debug_seed,
        freeze_poi=args.freeze_poi,
        init_mode=args.init_mode,
        deepternary_dockq_paths={
            "protein1_pdb": protein1,
            "protein2_pdb": protein2,
            "ligand_pdb": str(case_dir / "ligand.pdb"),
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
        "tribrain_checkpoint": args.tribrain_ckpt,
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
