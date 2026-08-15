#!/usr/bin/env python
"""
Diagnose the SE3-head geometry-awareness fix.

Before this fix, TriBrain was "coordinate-blind": MultiScalePooling and
EquivariantTransformer both ignored ``node_coords``, so the SE3 head (which has
no frozen expert underneath, unlike the FiLM path) could only see ESM features.
It therefore could not regress the coordinate-noise-dependent SE3 score target,
and loss_se3 plateaued ~1.1 instead of converging to 0.

The fix adds ``SE3GeometryEncoder``: a coordinate set-encoder whose output
``h_geo`` is concatenated into the SE3 head input alongside ``h_cond`` and the
time embedding. This script proves the head now *sees* coordinates and that the
encoder is trainable:

  PART A (coordinate sensitivity) — fix t/t_se3, run the forward pass on the
      original POI vs. a +10 Å translated POI (identical features). s_se3 must
      change. Also sweeps t_se3 with fixed coords to confirm time response.

  PART B (gradient flow) — backward() through s_se3 and confirm finite grads
      reach geo_encoder (the new module), not just the SE3 head's last layer.

Run via ``scripts/diagnose_se3_geo.slurm`` (SLURM — do not run GPU work
interactively).
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DIFFDOCK_DIR = _PROJECT_ROOT.parent / "DiffDock"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tricoopdiff.diffdock_compat  # noqa: F401  (patches torch_cluster/scatter first)

if not _DIFFDOCK_DIR.exists():
    print(f"FATAL: DiffDock checkout not found at {_DIFFDOCK_DIR}", flush=True)
    sys.exit(1)
sys.path.insert(0, str(_DIFFDOCK_DIR))

import torch
import yaml
from argparse import Namespace

from featurize_common import featurize_complex

from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.tribrain import TriBrain
from tricoopdiff.model import TriCoopDiff
from tricoopdiff.diffusion import MultiRateDiffusion

# SO(3)/torus cache resolves relative to CWD; chdir to project root.
os.chdir(_PROJECT_ROOT)


def main():
    case = os.environ.get("CASE", "5T35_D_A_759")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    protac_dir = _PROJECT_ROOT / "DeepTernary/output/protac22"
    case_dir = protac_dir / case
    protein1 = str(case_dir / "protein1.pdb")
    protein2 = str(case_dir / "protein2.pdb")
    _sdf = case_dir / "ligand.sdf"
    _pdb = case_dir / "ligand.pdb"
    ligand_path = str(_sdf if _sdf.exists() else _pdb)

    model_dir = _PROJECT_ROOT / "checkpoints/score_model"
    with open(model_dir / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    print(f"[setup] case={case} device={device}", flush=True)

    expert = build_diffdock_expert(model_dir=str(model_dir), device=str(device))
    num_conv_layers = expert.num_conv_layers

    e3_data, poi_data, ligand_data, d_rec = featurize_complex(
        case, protein1, protein2, ligand_path, score_model_args, str(device),
        cache_dir=str(_PROJECT_ROOT / "results/featurize_cache"),
    )
    print(f"[data] receptor feat dim={d_rec}", flush=True)

    tribrain = TriBrain(
        node_dim=128, input_dim=d_rec, ns=32, nv=4,
        num_transformer_layers=3, num_conv_layers_per_expert=num_conv_layers,
        film_dim=expert.ns,
    ).to(device)
    model = TriCoopDiff(expert, tribrain, num_conv_layers=num_conv_layers).to(device)
    model.eval()

    diffusion = MultiRateDiffusion(inference_steps=20)

    e3_d = {**e3_data, 'coords': e3_data['coords'].to(device)}
    lig_d = {**ligand_data, 'coords': ligand_data['coords'].to(device)}
    poi0 = poi_data['coords'].to(device)

    def run(poi_coords, t_se3):
        with torch.no_grad():
            out = model(
                e3_d, {**poi_data, 'coords': poi_coords}, lig_d,
                t=torch.tensor([0.5], device=device), t_se3=t_se3,
            )
        return out['s_se3'].detach()

    # ======================================================================
    # PART A — coordinate sensitivity
    # ======================================================================
    print("\n" + "=" * 92, flush=True)
    print("PART A — coordinate sensitivity: s_se3 must change with POI pose",
          flush=True)
    print("=" * 92, flush=True)
    t_se3 = torch.tensor([0.5], device=device)

    s0 = run(poi0, t_se3)                      # original POI
    s1 = run(poi0 + 10.0, t_se3)               # +10 Å translation (all axes)
    s2 = run(poi0 + 10.0, t_se3)               # repeat → deterministic check

    print(f"  s_se3(original)      = {s0.squeeze().tolist()}", flush=True)
    print(f"  s_se3(poi +10A)      = {s1.squeeze().tolist()}", flush=True)
    d_coord = (s0 - s1).abs().max().item()
    print(f"  max|Δs_se3| from +10A translation = {d_coord:.5f}", flush=True)
    print(f"  deterministic (repeat) max|s1-s2| = {(s1-s2).abs().max().item():.3e}",
          flush=True)

    # Time response (fixed coords, vary t_se3) — already worked after the
    # time-conditioning fix; confirm it still holds with the encoder present.
    print("\n  time sweep (fixed coords):", flush=True)
    prev = None
    max_dt = 0.0
    for q in (0.2, 0.5, 0.8):
        sq = run(poi0, torch.tensor([q], device=device))
        if prev is not None:
            max_dt = max(max_dt, (sq - prev).abs().max().item())
        prev = sq
    print(f"  max|Δs_se3| across t_se3 sweep      = {max_dt:.5f}", flush=True)

    coord_ok = d_coord > 1e-3 and (s1 - s2).abs().max().item() < 1e-6
    print(f"\n  COORDINATE-SENSITIVITY: {'PASS' if coord_ok else 'FAIL'}", flush=True)

    # ======================================================================
    # PART B — gradient flow to geo_encoder
    # ======================================================================
    print("\n" + "=" * 92, flush=True)
    print("PART B — gradient flow to geo_encoder", flush=True)
    print("=" * 92, flush=True)
    model.train()
    tribrain.zero_grad()
    out = model(
        e3_d, {**poi_data, 'coords': poi0}, lig_d,
        t=torch.tensor([0.5], device=device), t_se3=torch.tensor([0.5], device=device),
    )
    out['s_se3'].norm().backward()

    geo_grad_ok = False
    geo_norm = 0.0
    for name, p in tribrain.geo_encoder.named_parameters():
        if p.grad is not None:
            geo_norm += p.grad.norm().item() ** 2
            if torch.isfinite(p.grad).all():
                geo_grad_ok = True
    print(f"  geo_encoder grad L2 = {geo_norm ** 0.5:.5f}  finite={geo_grad_ok}",
          flush=True)
    print(f"\n  GEO-ENCODER GRADIENT: {'PASS' if geo_grad_ok else 'FAIL'}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
