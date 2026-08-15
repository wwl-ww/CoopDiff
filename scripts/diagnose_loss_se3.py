#!/usr/bin/env python
"""
Diagnose the ``loss_se3 ≈ 0`` observation from the Phase-1 smoke run.

Two parts:
  1. Scale inspection — for N random (t, t_se3), print the SE3 score-matching
     intermediates: λ_se3 mask, σ_tr/σ_rot, |noise|, |target|, |head output|,
     and the loss_se3_tr / loss_se3_rot split. This reveals whether the loss is
     genuinely ~0 (target/head scale off) or just masked by λ_se3(t<0.5).
  2. Gradient inspection — one backward pass through the real ``compute_loss``,
     then report per-head gradient norms (SE3 head, FiLM heads, FiLM-shared,
     transformer) and confirm the frozen expert carries no grads.

Run via ``scripts/diagnose_loss_se3.slurm`` (SLURM — do not run GPU work
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
from tricoopdiff.train import TriCoopDiffTrainer

# SO(3)/torus cache resolves relative to CWD; chdir to project root.
os.chdir(_PROJECT_ROOT)


def grad_norm(mod):
    g = [p.grad for p in mod.parameters() if p.grad is not None]
    if not g:
        return 0.0
    return sum(x.norm().item() ** 2 for x in g) ** 0.5


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

    # --- Frozen expert ---
    expert = build_diffdock_expert(model_dir=str(model_dir), device=str(device))
    num_conv_layers = expert.num_conv_layers

    # --- Featurize ---
    e3_data, poi_data, ligand_data, d_rec = featurize_complex(
        case, protein1, protein2, ligand_path, score_model_args, str(device),
        cache_dir=str(_PROJECT_ROOT / "results/featurize_cache"),
    )
    print(f"[data] receptor feat dim={d_rec}", flush=True)

    # --- TriBrain + model + diffusion + trainer (mirrors build_trainer) ---
    tribrain = TriBrain(
        node_dim=128, input_dim=d_rec, ns=32, nv=4,
        num_transformer_layers=3, num_conv_layers_per_expert=num_conv_layers,
        film_dim=expert.ns,
    )
    model = TriCoopDiff(expert, tribrain, num_conv_layers=num_conv_layers).to(device)
    diffusion = MultiRateDiffusion(inference_steps=20)
    trainer = TriCoopDiffTrainer(model=model, diffusion=diffusion, lr=1e-4, device=str(device))
    trainer.set_phase(1)

    # ======================================================================
    # Part 1 — SE3 scale inspection (no backward)
    # ======================================================================
    print("\n" + "=" * 78, flush=True)
    print("PART 1 — SE3 score-matching scales (per random t, t_se3)", flush=True)
    print("=" * 78, flush=True)
    hdr = (f"{'step':>4} {'t_se3':>7} {'λ_se3':>6} {'σ_tr':>7} {'σ_rot':>7} | "
           f"{'|tr_noise|':>10} {'|tr_target|':>11} {'|s_tr|':>9} {'|s_rot|':>9} | "
           f"{'L_se3_tr':>9} {'L_se3_rot':>9}")
    print(hdr, flush=True)

    n_steps = int(os.environ.get("N_STEPS", "12"))
    for step in range(n_steps):
        t = torch.rand(1, device=device)
        t_se3 = torch.rand(1, device=device)

        noisy_ligand, tr_noise, rot_noise, rot_omega = diffusion.add_noise_protac(
            ligand_data['coords'].to(device), t, t, t)
        if ligand_data.get('edge_mask') is not None and ligand_data.get('mask_rotate') is not None:
            noisy_ligand, _ = diffusion.add_torsion_noise(
                noisy_ligand, t, ligand_data['edge_index'],
                ligand_data['edge_mask'], ligand_data['mask_rotate'])

        noisy_poi, se3_tr_noise, se3_rot_noise, se3_rot_omega = diffusion.add_noise_se3(
            poi_data['coords'].to(device), t_se3, t_se3)

        output = model(
            {**e3_data, 'coords': e3_data['coords'].to(device)},
            {**poi_data, 'coords': noisy_poi},
            {**ligand_data, 'coords': noisy_ligand},
            t=t, t_se3=t_se3,
            r_e3_to_poi=se3_rot_noise.transpose(-1, -2),
        )

        se3_tr_sigma, se3_rot_sigma = diffusion.se3_t_to_sigma(t_se3, t_se3)
        se3_tr_target = -se3_tr_noise / se3_tr_sigma.unsqueeze(-1) ** 2
        se3_rot_target = so3_score_vec(se3_rot_sigma, se3_rot_omega)
        se3_rot_norm = so3_score_norm(se3_rot_sigma).unsqueeze(-1)

        s_tr = output['s_se3_components']['tr']
        s_rot = output['s_se3_components']['rot']
        loss_se3_tr = ((s_tr - se3_tr_target) ** 2 * se3_tr_sigma.unsqueeze(-1) ** 2).mean()
        loss_se3_rot = ((s_rot - se3_rot_target) ** 2 / se3_rot_norm ** 2).mean()

        lam = diffusion.get_lambda_se3(t_se3)
        print(
            f"{step:>4} {t_se3.item():>7.3f} {lam.item():>6.1f} "
            f"{se3_tr_sigma.item():>7.3f} {se3_rot_sigma.item():>7.3f} | "
            f"{se3_tr_noise.norm():>10.3f} {se3_tr_target.norm():>11.4f} "
            f"{s_tr.norm():>9.4f} {s_rot.norm():>9.4f} | "
            f"{loss_se3_tr.item():>9.4f} {loss_se3_rot.item():>9.4f}",
            flush=True,
        )

    # ======================================================================
    # Part 2 — gradient inspection (one backward through real compute_loss)
    # ======================================================================
    print("\n" + "=" * 78, flush=True)
    print("PART 2 — per-head gradient norms (one backward pass)", flush=True)
    print("=" * 78, flush=True)

    batch = {
        'e3_data': e3_data, 'poi_data': poi_data,
        'ligand_data': ligand_data, 't_se3_data': None,
    }
    loss_dict = trainer.compute_loss(**batch)
    trainer.optimizer.zero_grad()
    loss_dict['loss_total'].backward()

    # Frozen expert must be grad-free.
    expert_grad = 0.0
    for p in expert.parameters():
        if p.grad is not None:
            expert_grad += p.grad.norm().item() ** 2
    expert_grad = expert_grad ** 0.5

    print(f"loss_total   = {loss_dict['loss_total'].item():.4f}", flush=True)
    print(f"  loss_score = {loss_dict['loss_score']:.4f} "
          f"(tr {loss_dict['loss_score_tr']:.4f} / rot {loss_dict['loss_score_rot']:.4f} "
          f"/ tor {loss_dict['loss_score_tor']:.4f})", flush=True)
    print(f"  loss_se3   = {loss_dict['loss_se3']:.4f}", flush=True)
    print(f"  loss_phys  = {loss_dict['loss_phys']:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"grad |se3_head|      = {grad_norm(tribrain.se3_head):.4e}", flush=True)
    print(f"grad |film_heads|    = {sum(grad_norm(h) for h in tribrain.film_heads):.4e}", flush=True)
    print(f"grad |film_shared|   = {grad_norm(tribrain.film_shared):.4e}", flush=True)
    print(f"grad |transformer|   = {grad_norm(tribrain.transformer):.4e}", flush=True)
    print(f"grad |judge_head|    = {grad_norm(tribrain.judge_head):.4e} (frozen in phase1)", flush=True)
    print(f"grad |state_head|    = {grad_norm(tribrain.state_head):.4e} (frozen in phase1)", flush=True)
    print(f"grad |frozen expert| = {expert_grad:.4e} (must be 0)", flush=True)

    # Per-head SE3 grad norm (not just the whole module).
    print("-" * 78, flush=True)
    for name, p in tribrain.se3_head.named_parameters():
        g = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"  se3_head.{name}: grad={g:.4e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
