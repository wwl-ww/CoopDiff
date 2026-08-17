"""
Reverse-diffusion sampling for TriCoopDiff.

Composes the existing reverse SDE primitives in ``diffusion.py`` into a full
t=1→0 denoising loop that produces a predicted PROTAC pose (E3 reference frame)
and a predicted E3-POI pose.

Reference frame: E3 fixed at its featurized coordinates. PROTAC and POI are
sampled once at t=1 (via ``TernaryInitializer``) and denoised step by step,
rebuilding the two subgraphs + ternary graph inside ``model.forward`` at every
step from the current (noisy) coordinates.

This module is self-contained: it takes already-featurized B=1 data dicts
(``featurize_common.featurize_complex``) and the built model + diffusion, so it
does NOT import ``../DiffDock``.
"""

import numpy as np
import torch

from .diffusion import MultiRateDiffusion, TernaryInitializer


def _sanitize_score(score):
    """
    Replace NaN/inf entries in a score tensor with a small fraction of its mean
    |score|, mirroring the vendored DiffDock inference fallback (``sampling.py``
    ~L139-150). DiffDock's own loop does this because when the PROTAC drifts far
    from the receptor the cross edges vanish and the expert emits NaN scores;
    without the sanitize those NaNs leak into the pose via the reverse update.
    """
    if score is None or score.numel() == 0:
        return score
    if torch.isnan(score).any() or torch.isinf(score).any():
        eps = 0.01 * score.abs().nanmean()
        if torch.isnan(eps):
            eps = torch.tensor(0.01, device=score.device, dtype=score.dtype)
        score = torch.nan_to_num(
            score, nan=eps.item(), posinf=eps.item(), neginf=-eps.item()
        )
    return score


@torch.no_grad()
def sample_single(
    model,
    diffusion: MultiRateDiffusion,
    initializer: TernaryInitializer,
    e3_data: dict,
    poi_data: dict,
    ligand_data: dict,
    steps: int = None,
    device: str = "cuda",
    seed: int = None,
    return_trajectory: bool = False,
) -> dict:
    """
    Run one reverse-diffusion trajectory (single walker) from t=1 to t=0.

    Args:
        model: ``TriCoopDiff`` (already on ``device``; set to eval mode here).
        diffusion: ``MultiRateDiffusion`` (holds sigma schedules + reverse steps).
        initializer: ``TernaryInitializer``.
        e3_data/poi_data/ligand_data: B=1 featurized dicts (see featurize_common).
        steps: number of reverse steps (default: diffusion.inference_steps).
        device: 'cuda' or 'cpu'.
        seed: RNG seed for torch + numpy (None = unseeded).
        return_trajectory: also return per-step (ligand, poi) coords.

    Returns:
        dict with 'ligand_coords' [N_lig, 3], 'poi_coords' [N_poi, 3] (CPU numpy),
        plus 'trajectory' (list of (lig, poi) per step) when requested.
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if steps is None:
        steps = diffusion.inference_steps

    # Clean templates (drop the B=1 dim).
    ligand_template = ligand_data["coords"][0].detach()  # [N_lig, 3]
    poi_template = poi_data["coords"][0].detach()        # [N_poi, 3]
    e3_coords = e3_data["coords"][0].detach()            # [N_e3, 3]

    # E3 center = receptor COM. Native randomize_position uses receptor.pos.mean(0).
    e3_center = e3_coords.mean(dim=0, keepdim=True)            # [1, 3]

    # Initialize at t=1 (fully noised). Passing the receptor COM + torsion metadata
    # + tr_sigma_max makes initialize_protac replicate native randomize_position
    # (uniform torsions, rotate about ligand COM, translate to receptor COM,
    # Gaussian N(0, tr_sigma_max^2)) — sampling-fidelity hypothesis #3.
    ligand_pos = initializer.initialize_protac(
        e3_center, ligand_template,
        lig_edge_index=ligand_data.get("edge_index"),
        lig_edge_mask=ligand_data.get("edge_mask"),
        lig_mask_rotate=ligand_data.get("mask_rotate"),
        tr_sigma_max=diffusion.tr_sigma_max,
    )  # [1, N_lig, 3]
    poi_pos, rot_init = initializer.initialize_poi(                           # [1, N_poi, 3], [1,3,3]
        e3_center, poi_template, return_rotation=True,
    )

    # Cumulative SE3 rotation applied to the POI (template -> current frame).
    # r_e3_to_poi = R_cum^T rotates the PROTAC into the POI body frame so that
    # subgraph2 sees the correct PROTAC-POI relative geometry (see model.forward).
    R_cum = rot_init  # [1, 3, 3]

    schedule = diffusion.get_inference_schedule(steps)
    trajectory = [] if return_trajectory else None

    for i in range(steps):
        t = torch.tensor([schedule["tr_schedule"][i]], device=device, dtype=torch.float32)
        t_se3 = torch.tensor([schedule["se3_tr_schedule"][i]], device=device, dtype=torch.float32)
        dt = torch.tensor(
            [schedule["tr_schedule"][i] - schedule["tr_schedule"][i + 1]],
            device=device, dtype=torch.float32,
        )

        # Rebuild the data dicts with the current (noisy) coordinates. Edges and
        # features are constant; only coords change between steps.
        lig_d = {**ligand_data, "coords": ligand_pos}
        poi_d = {**poi_data, "coords": poi_pos}

        out = model(
            e3_data, poi_d, lig_d, t=t, t_se3=t_se3,
            r_e3_to_poi=R_cum.transpose(-1, -2),
        )
        s_q = {k: _sanitize_score(v) for k, v in out["s_q"].items()}
        s_se3 = _sanitize_score(out["s_se3"])  # [1, 6]

        add_noise = i < steps - 1  # no stochastic noise on the final step

        ligand_pos = diffusion.reverse_step_protac(
            ligand_pos, s_q["tr"], s_q["rot"], s_q["tor"],
            t, t, t, dt, dt, dt, add_noise=add_noise,
        )

        # Torsion reverse (PROTAC rotatable bonds) — only when the conformer has
        # rotatable bonds (edge_mask/mask_rotate present and tor score non-empty).
        if (
            ligand_data.get("edge_mask") is not None
            and ligand_data.get("mask_rotate") is not None
            and s_q["tor"].numel() > 0
        ):
            ligand_pos = diffusion.reverse_step_torsion(
                ligand_pos, s_q["tor"], t, dt,
                ligand_data["edge_index"], ligand_data["edge_mask"],
                ligand_data["mask_rotate"], add_noise=add_noise,
            )

        poi_pos, rot_matrix = diffusion.reverse_step_se3(
            poi_pos, s_se3, t_se3, t_se3, dt, dt, add_noise=add_noise,
            return_rotation=True,
        )
        R_cum = R_cum @ rot_matrix

        if return_trajectory:
            trajectory.append(
                (ligand_pos[0].detach().cpu().numpy(), poi_pos[0].detach().cpu().numpy())
            )

    result = {
        "ligand_coords": ligand_pos[0].detach().cpu().numpy(),
        "poi_coords": poi_pos[0].detach().cpu().numpy(),
    }
    if return_trajectory:
        result["trajectory"] = trajectory
    return result


def sample_ensemble(
    model,
    diffusion: MultiRateDiffusion,
    initializer: TernaryInitializer,
    e3_data: dict,
    poi_data: dict,
    ligand_data: dict,
    steps: int = None,
    device: str = "cuda",
    n_walkers: int = 32,
    seeds=None,
) -> list:
    """
    Run ``n_walkers`` independent reverse-diffusion trajectories.

    Args:
        (same as ``sample_single`` for model/data/diffusion/initializer)
        n_walkers: number of independent trajectories.
        seeds: optional list of per-walker RNG seeds (default: range(n_walkers)).

    Returns:
        list of dicts (one per walker), each with 'ligand_coords'/'poi_coords'.
    """
    if seeds is None:
        seeds = list(range(n_walkers))
    else:
        n_walkers = len(seeds)

    results = []
    for i in range(n_walkers):
        results.append(
            sample_single(
                model, diffusion, initializer, e3_data, poi_data, ligand_data,
                steps=steps, device=device, seed=seeds[i],
            )
        )
    return results
