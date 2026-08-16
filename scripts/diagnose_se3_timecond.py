#!/usr/bin/env python
"""
Diagnose *why* the SE3 head does not learn (flat loss_se3 ≈ 1.2 over training).

Root-cause hypothesis (verified by code reading): the SE3 score target is a
function of the diffusion time ``t_se3`` (via σ_tr ∈ [0.5, 20], σ_rot ∈
[0.05, 1.0]), but ``TriBrain.forward`` takes NO time argument and
``model.forward`` never passes ``t_se3`` to it. The SE3 head therefore has no
way to condition its output on the noise level — it must emit a *single fixed
6-vector* while the target varies over orders of magnitude.

This script demonstrates it directly on the trained checkpoint:

  PART A (time-sweep) — hold the noisy POI coords fixed, sweep the *query*
  t_se3 used to build the score target, and show the target magnitude changes
  dramatically while the head output (which never sees t_se3) stays constant.

  PART B (trained vs zero-init) — compare |s_rot| / |s_tr| of the trained
  checkpoint against the zero-init head, and against the target magnitude.

Run via ``scripts/diagnose_se3_timecond.slurm`` (SLURM — do not run GPU work
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
from tricoopdiff.diffusion import MultiRateDiffusion, so3_score_norm, so3_score_vec

# SO(3)/torus cache resolves relative to CWD; chdir to project root.
os.chdir(_PROJECT_ROOT)


def main():
    case = os.environ.get("CASE", "5T35_D_A_759")
    ckpt_rel = os.environ.get("CKPT", "results/train_phase1/best_model.pt")
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

    print(f"[setup] case={case} device={device} ckpt={ckpt_rel}", flush=True)

    # --- Frozen expert ---
    expert = build_diffdock_expert(model_dir=str(model_dir), device=str(device))
    num_conv_layers = expert.num_conv_layers

    # --- Featurize ---
    e3_data, poi_data, ligand_data, d_rec = featurize_complex(
        case, protein1, protein2, ligand_path, score_model_args, str(device),
        cache_dir=str(_PROJECT_ROOT / "results/featurize_cache"),
    )
    print(f"[data] receptor feat dim={d_rec}", flush=True)

    def make_model():
        tribrain = TriBrain(
            node_dim=128, input_dim=d_rec, ns=32, nv=4,
            num_transformer_layers=3, num_conv_layers_per_expert=num_conv_layers,
            film_dim=expert.ns,
        )
        m = TriCoopDiff(expert, tribrain, num_conv_layers=num_conv_layers).to(device)
        return m, tribrain

    # ======================================================================
    # PART A — time-sweep: fixed noisy POI, vary the target's t_se3
    # ======================================================================
    model, tribrain = make_model()
    if ckpt_rel and ckpt_rel.lower() != "none":
        ckpt = torch.load(_PROJECT_ROOT / ckpt_rel, map_location=device)
        tribrain.load_state_dict(ckpt["tribrain_state_dict"])
        print(f"[ckpt] loaded tribrain_state_dict (global_step={ckpt.get('global_step')})",
              flush=True)
    else:
        print("[ckpt] skipped — using fresh (random-init) TriBrain", flush=True)
    model.eval()  # disable dropout so t_se3 is the only source of output variance

    diffusion = MultiRateDiffusion(inference_steps=20)

    # Generate the noisy POI ONCE (fixed t_se3_gen). The head sees only these
    # coords; the query t_se3 below changes only the *target* σ.
    t_se3_gen = torch.tensor([0.7], device=device)
    noisy_poi, se3_tr_noise, se3_rot_noise, se3_rot_omega = diffusion.add_noise_se3(
        poi_data['coords'].to(device), t_se3_gen, t_se3_gen,
    )
    print(f"[partA] fixed noisy POI generated at t_se3_gen={t_se3_gen.item():.2f}",
          flush=True)

    print("\n" + "=" * 92, flush=True)
    print("PART A — sweep query t_se3, head input held FIXED (no time conditioning)",
          flush=True)
    print("=" * 92, flush=True)
    hdr = (f"{'q_t_se3':>7} {'σ_tr':>7} {'σ_rot':>7} | "
           f"{'|tr_target|':>11} {'|rot_target|/norm':>16} {'|s_tr|':>9} {'|s_rot|':>9} | "
           f"{'L_tr':>8} {'L_rot':>8}")
    print(hdr, flush=True)

    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        t_se3 = torch.tensor([q], device=device)
        sigma_tr, sigma_rot = diffusion.se3_t_to_sigma(t_se3, t_se3)

        # Targets for THIS t_se3 (same noise realization, different σ)
        tr_target = -se3_tr_noise / sigma_tr.unsqueeze(-1) ** 2
        rot_target = so3_score_vec(sigma_rot, se3_rot_omega)
        rot_norm = so3_score_norm(sigma_rot).unsqueeze(-1)

        # Head output — model.forward ignores t_se3 for TriBrain, so it is
        # IDENTICAL across the sweep.
        output = model(
            {**e3_data, 'coords': e3_data['coords'].to(device)},
            {**poi_data, 'coords': noisy_poi},
            {**ligand_data, 'coords': ligand_data['coords'].to(device)},
            t=torch.tensor([0.5], device=device), t_se3=t_se3,
            r_e3_to_poi=se3_rot_noise.transpose(-1, -2),
        )
        s_tr = output['s_se3_components']['tr']
        s_rot = output['s_se3_components']['rot']

        loss_tr = ((s_tr - tr_target) ** 2 * sigma_tr.unsqueeze(-1) ** 2).mean()
        loss_rot = ((s_rot - rot_target) ** 2 / rot_norm ** 2).mean()

        print(f"{q:>7.1f} {sigma_tr.item():>7.3f} {sigma_rot.item():>7.3f} | "
              f"{tr_target.norm():>11.4f} {rot_target.norm()/rot_norm.squeeze():>16.4f} "
              f"{s_tr.norm():>9.4f} {s_rot.norm():>9.4f} | "
              f"{loss_tr.item():>8.4f} {loss_rot.item():>8.4f}",
              flush=True)

    # ======================================================================
    # PART B — trained vs zero-init head magnitude
    # ======================================================================
    print("\n" + "=" * 92, flush=True)
    print("PART B — head output magnitude: trained vs zero-init", flush=True)
    print("=" * 92, flush=True)

    def head_stats(model, tribrain, label, n=24):
        norms_tr, norms_rot = [], []
        tgt_tr, tgt_rot = [], []
        ltr, lrot = [], []
        for _ in range(n):
            t = torch.rand(1, device=device)
            t_se3 = torch.rand(1, device=device)
            noisy_poi, se3_tr_noise, se3_rot_noise, se3_rot_omega = diffusion.add_noise_se3(
                poi_data['coords'].to(device), t_se3, t_se3)
            sigma_tr, sigma_rot = diffusion.se3_t_to_sigma(t_se3, t_se3)
            tr_target = -se3_tr_noise / sigma_tr.unsqueeze(-1) ** 2
            rot_target = so3_score_vec(sigma_rot, se3_rot_omega)
            rot_norm = so3_score_norm(sigma_rot).unsqueeze(-1)
            output = model(
                {**e3_data, 'coords': e3_data['coords'].to(device)},
                {**poi_data, 'coords': noisy_poi},
                {**ligand_data, 'coords': ligand_data['coords'].to(device)},
                t=t, t_se3=t_se3,
                r_e3_to_poi=se3_rot_noise.transpose(-1, -2),
            )
            s_tr = output['s_se3_components']['tr']
            s_rot = output['s_se3_components']['rot']
            norms_tr.append(s_tr.norm().item())
            norms_rot.append(s_rot.norm().item())
            tgt_tr.append(tr_target.norm().item())
            tgt_rot.append((rot_target.norm() / rot_norm.squeeze()).item())
            ltr.append(((s_tr - tr_target) ** 2 * sigma_tr.unsqueeze(-1) ** 2).mean().item())
            lrot.append(((s_rot - rot_target) ** 2 / rot_norm ** 2).mean().item())
        def agg(x):
            return (sum(x) / len(x), sorted(x)[len(x) // 2])
        mt, md = agg(norms_tr)
        mr, mdr = agg(norms_rot)
        tt, td = agg(tgt_tr)
        trt, trd = agg(tgt_rot)
        lt, ld = agg(ltr)
        lr, lrd = agg(lrot)
        print(f"  [{label}] |s_tr| mean={mt:.4f} med={md:.4f} | "
              f"|s_rot| mean={mr:.4f} med={mdr:.4f}", flush=True)
        print(f"      target |tr| mean={tt:.4f} med={td:.4f} | "
              f"|rot|/norm mean={trt:.4f} med={trd:.4f}", flush=True)
        print(f"      loss_tr mean={lt:.4f} med={ld:.4f} | "
              f"loss_rot mean={lr:.4f} med={lrd:.4f}", flush=True)

    # Trained
    head_stats(model, tribrain, "trained", n=24)

    # Zero-init (fresh TriBrain, same weights as training start)
    torch.manual_seed(1234)
    model0, tribrain0 = make_model()
    torch.manual_seed(0)  # restore so noise sampling below matches the trained pass
    head_stats(model0, tribrain0, "zero-init", n=24)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
