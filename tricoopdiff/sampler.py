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


@torch.no_grad()
def _tensor_summary(tensor: torch.Tensor) -> dict:
    """Compact numeric summary for per-step parity diagnostics."""
    t = tensor.detach().float()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "norm": float(torch.linalg.vector_norm(t).item()),
        "max_abs": float(t.abs().max().item()) if t.numel() > 0 else 0.0,
    }


def _sanitize_score(score):
    """
    Replace NaN/inf entries in a score tensor with a small fraction of its mean
    |score|, mirroring the vendored DiffDock inference fallback (``sampling.py``
    ~L139-150). DiffDock's own loop does this because when the PROTAC drifts far
    from the receptor the cross edges vanish and the expert emits NaN scores;
    without the sanitize those NaNs leak into the pose via the reverse update.

    Returns:
        (sanitized_score, had_nonfinite)
    """
    if score is None or score.numel() == 0:
        return score, False
    had_nonfinite = bool(torch.isnan(score).any() or torch.isinf(score).any())
    if had_nonfinite:
        eps = 0.01 * score.abs().nanmean()
        if torch.isnan(eps):
            eps = torch.tensor(0.01, device=score.device, dtype=score.dtype)
        score = torch.nan_to_num(
            score, nan=eps.item(), posinf=eps.item(), neginf=-eps.item()
        )
    return score, had_nonfinite


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
    return_trace: bool = False,
    freeze_poi: bool = False,
    init_mode: str = "native",
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
        return_trace: also return per-step scalar diagnostics for parity debugging.
        freeze_poi: when True, keep the POI fixed at its featurized/native pose and
            disable the SE3 reverse updates. This defines the strict zero-FiLM
            E3-L parity baseline: the sampler reduces to native-style DiffDock on
            the PROTAC branch instead of running an extra ternary POI dynamics.
        init_mode: ``native`` or ``coupled5a`` initialization prior.

    Returns:
        dict with 'ligand_coords' [N_lig, 3], 'poi_coords' [N_poi, 3] (CPU numpy),
        plus 'trajectory' (list of (lig, poi) per step) when requested and
        'trace' (per-step diagnostic dicts) when requested.
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
        init_mode=init_mode,
    )  # [1, N_lig, 3]
    poi_pos, rot_init = initializer.initialize_poi(
        e3_center, poi_template, return_rotation=True, init_mode=init_mode,
    )

    if freeze_poi:
        poi_pos = poi_data["coords"].clone().detach()
        R_cum = torch.eye(3, device=device, dtype=poi_pos.dtype).unsqueeze(0)
    else:
        # Cumulative SE3 rotation applied to the POI (template -> current frame).
        # r_e3_to_poi = R_cum^T rotates the PROTAC into the POI body frame so that
        # subgraph2 sees the correct PROTAC-POI relative geometry (see model.forward).
        R_cum = rot_init  # [1, 3, 3]

    schedule = diffusion.get_inference_schedule(steps)
    trajectory = [] if return_trajectory else None
    trace = [] if return_trace else None

    init_ligand_com = ligand_pos.mean(dim=1)[0]
    init_poi_com = poi_pos.mean(dim=1)[0]
    init_record = {
        "stage": "init",
        "seed": seed,
        "ligand_com": init_ligand_com.detach().cpu().numpy().tolist(),
        "poi_com": init_poi_com.detach().cpu().numpy().tolist(),
        "ligand_template_com": ligand_template.mean(dim=0).detach().cpu().numpy().tolist(),
        "poi_template_com": poi_template.mean(dim=0).detach().cpu().numpy().tolist(),
        "e3_center": e3_center[0].detach().cpu().numpy().tolist(),
        "ligand_pocket_dist": float(torch.linalg.vector_norm(init_ligand_com - e3_center[0]).item()),
        "poi_e3_dist": float(torch.linalg.vector_norm(init_poi_com - e3_center[0]).item()),
        "n_rotatable_bonds": int(ligand_data.get("edge_mask").sum().item())
        if ligand_data.get("edge_mask") is not None else 0,
    }
    if return_trace:
        trace.append(init_record)

    for i in range(steps):
        t_tr = torch.tensor([schedule["tr_schedule"][i]], device=device, dtype=torch.float32)
        t_rot = torch.tensor([schedule["rot_schedule"][i]], device=device, dtype=torch.float32)
        t_tor = torch.tensor([schedule["tor_schedule"][i]], device=device, dtype=torch.float32)
        t_se3_tr = torch.tensor([schedule["se3_tr_schedule"][i]], device=device, dtype=torch.float32)
        t_se3_rot = torch.tensor([schedule["se3_rot_schedule"][i]], device=device, dtype=torch.float32)
        if i < steps - 1:
            dt_tr = torch.tensor(
                [schedule["tr_schedule"][i] - schedule["tr_schedule"][i + 1]],
                device=device, dtype=torch.float32,
            )
            dt_rot = torch.tensor(
                [schedule["rot_schedule"][i] - schedule["rot_schedule"][i + 1]],
                device=device, dtype=torch.float32,
            )
            dt_tor = torch.tensor(
                [schedule["tor_schedule"][i] - schedule["tor_schedule"][i + 1]],
                device=device, dtype=torch.float32,
            )
            dt_se3_tr = torch.tensor(
                [schedule["se3_tr_schedule"][i] - schedule["se3_tr_schedule"][i + 1]],
                device=device, dtype=torch.float32,
            )
            dt_se3_rot = torch.tensor(
                [schedule["se3_rot_schedule"][i] - schedule["se3_rot_schedule"][i + 1]],
                device=device, dtype=torch.float32,
            )
        else:
            # Match vendored DiffDock's sampling driver: the final deterministic
            # step uses dt = t_i (actual_steps=19 with a length-20 schedule), not
            # t_i - t_{i+1} against an explicit zero endpoint.
            dt_tr = t_tr.clone()
            dt_rot = t_rot.clone()
            dt_tor = t_tor.clone()
            dt_se3_tr = t_se3_tr.clone()
            dt_se3_rot = t_se3_rot.clone()

        tr_sigma, rot_sigma, tor_sigma = diffusion.t_to_sigma(t_tr, t_rot, t_tor)
        se3_tr_sigma, se3_rot_sigma = diffusion.se3_t_to_sigma(t_se3_tr, t_se3_rot)

        if bool(getattr(model.expert.model, "dynamic_max_cross", False)):
            receptor_crop_cutoff = tr_sigma * 3 + 20
        else:
            receptor_crop_cutoff = None

        # Rebuild the data dicts with the current (noisy) coordinates. Edges and
        # features are constant; only coords change between steps.
        lig_d = {**ligand_data, "coords": ligand_pos}
        poi_d = {**poi_data, "coords": poi_pos}

        if return_trace:
            setattr(model, '_debug_step0_probe', True)
            setattr(model, '_last_step0_probe', None)
        out = model(
            e3_data, poi_d, lig_d, t=t_tr, t_se3=t_se3_tr,
            r_e3_to_poi=R_cum.transpose(-1, -2),
            receptor_crop_cutoff=receptor_crop_cutoff,
        )
        if return_trace:
            step0_probe = getattr(model, '_last_step0_probe', None)
            setattr(model, '_debug_step0_probe', False)
        else:
            step0_probe = None
        s_q = {}
        sanitized_components = []
        for name, value in out["s_q"].items():
            s_q[name], had_nonfinite = _sanitize_score(value)
            if had_nonfinite:
                sanitized_components.append(name)
        s_se3, se3_had_nonfinite = _sanitize_score(out["s_se3"])
        if freeze_poi:
            s_se3 = torch.zeros_like(s_se3)
            se3_had_nonfinite = False

        add_noise = i < steps - 1  # no stochastic noise on the final step

        rigid_ligand_pos, rigid_target_pos = diffusion.reverse_step_protac(
            ligand_pos, s_q["tr"], s_q["rot"], s_q["tor"],
            t_tr, t_rot, t_tor, dt_tr, dt_rot, dt_tor, add_noise=add_noise,
        )
        ligand_pos = rigid_ligand_pos

        # Torsion reverse (PROTAC rotatable bonds) — only when the conformer has
        # rotatable bonds (edge_mask/mask_rotate present and tor score non-empty).
        if (
            ligand_data.get("edge_mask") is not None
            and ligand_data.get("mask_rotate") is not None
            and s_q["tor"].numel() > 0
        ):
            ligand_pos = diffusion.reverse_step_torsion(
                ligand_pos, s_q["tor"], t_tor, dt_tor,
                ligand_data["edge_index"], ligand_data["edge_mask"],
                ligand_data["mask_rotate"], add_noise=add_noise,
                rigid_target_pos=rigid_target_pos,
            )

        if not freeze_poi:
            poi_pos, rot_matrix = diffusion.reverse_step_se3(
                poi_pos, s_se3, t_se3_tr, t_se3_rot, dt_se3_tr, dt_se3_rot,
                add_noise=add_noise, return_rotation=True,
            )
            # `rot_matrix` is the applied POI-body rotation. Accumulate it on the
            # left so `R_cum` stays the template->current body-frame transform, and
            # the next forward pass can still recover the current POI frame via
            # `r_e3_to_poi = R_cumᵀ`.
            R_cum = rot_matrix @ R_cum

        if return_trace:
            lig_com = ligand_pos.mean(dim=1)[0]
            poi_com = poi_pos.mean(dim=1)[0]
            step_record = {
                "stage": "step",
                "step": i,
                "add_noise": add_noise,
                "sanitized_components": sanitized_components,
                "sanitized_se3": bool(se3_had_nonfinite),
                "freeze_poi": bool(freeze_poi),
                "t": {
                    "tr": float(t_tr.item()),
                    "rot": float(t_rot.item()),
                    "tor": float(t_tor.item()),
                    "se3_tr": float(t_se3_tr.item()),
                    "se3_rot": float(t_se3_rot.item()),
                },
                "dt": {
                    "tr": float(dt_tr.item()),
                    "rot": float(dt_rot.item()),
                    "tor": float(dt_tor.item()),
                    "se3_tr": float(dt_se3_tr.item()),
                    "se3_rot": float(dt_se3_rot.item()),
                },
                "sigma": {
                    "tr": float(tr_sigma.item()),
                    "rot": float(rot_sigma.item()),
                    "tor": float(tor_sigma.item()),
                    "se3_tr": float(se3_tr_sigma.item()),
                    "se3_rot": float(se3_rot_sigma.item()),
                },
                "receptor_crop_cutoff": float(receptor_crop_cutoff.item()),
                "ligand_com": lig_com.detach().cpu().numpy().tolist(),
                "poi_com": poi_com.detach().cpu().numpy().tolist(),
                "ligand_pocket_dist": float(torch.linalg.vector_norm(lig_com - e3_center[0]).item()),
                "poi_e3_dist": float(torch.linalg.vector_norm(poi_com - e3_center[0]).item()),
                "r_cum_det": float(torch.det(R_cum[0]).item()),
                "score_summary": {
                    "tr": _tensor_summary(s_q["tr"]),
                    "rot": _tensor_summary(s_q["rot"]),
                    "tor": _tensor_summary(s_q["tor"]),
                    "se3": _tensor_summary(s_se3),
                },
                "step0_probe": step0_probe,
            }
            trace.append(step_record)

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
    if return_trace:
        result["trace"] = trace
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
    freeze_poi: bool = False,
    init_mode: str = "native",
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
                steps=steps, device=device, seed=seeds[i], freeze_poi=freeze_poi,
                init_mode=init_mode,
            )
        )
    return results
