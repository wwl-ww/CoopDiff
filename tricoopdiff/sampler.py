"""
Reverse-diffusion sampling for TriCoopDiff.

Composes the deterministic reverse flow in ``sdpi_flow.py`` into a full
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

from .initialization import TernaryInitializer
from .sdpi_flow import SDPIFlow, FLOW_VERSION
from .consensus import SOURCE_NAMES


def actual_candidate_source(init_mode, initialization):
    """Return the source actually produced, including BiTSeed fallback."""
    if init_mode == "sdpi":
        source = "sdpi"
    elif init_mode == "fixed_candidate":
        source = initialization.get("candidate_source")
    elif init_mode == "expert_anchor":
        source = "sdpi" if initialization.get("fallback") else "bitseed"
    else:
        raise ValueError(f"unsupported initialization mode: {init_mode}")
    if source not in SOURCE_NAMES:
        raise ValueError("initializer did not report a valid actual candidate source")
    return source


@torch.no_grad()
def _tensor_summary(tensor: torch.Tensor) -> dict:
    """Compact numeric summary for per-step parity diagnostics."""
    t = tensor.detach().float()
    return {
        "mean": float(t.mean().item()) if t.numel() else 0.0,
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "norm": float(torch.linalg.vector_norm(t).item()),
        "max_abs": float(t.abs().max().item()) if t.numel() > 0 else 0.0,
    }


def _sanitize_score(score):
    """Detect a non-finite score without inventing a replacement value.

    The historical name is retained for local callers, but sanitization is no
    longer performed. A non-finite expert prediction invalidates the walker
    before coordinates are updated; ``nan_to_num`` would hide an OOD state and
    create a pose that is not sampled from the learned reverse process.

    Returns:
        (original_score, had_nonfinite)
    """
    if score is None or score.numel() == 0:
        return score, False
    had_nonfinite = bool(torch.isnan(score).any() or torch.isinf(score).any())
    return score, had_nonfinite


@torch.no_grad()
def sample_single(
    model,
    diffusion: SDPIFlow,
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
    init_mode: str = "sdpi",
    t_anchor: float = None,
    translation_velocity_scale: float = 1.0,
    rotation_velocity_scale: float = 1.0,
    velocity_transform=None,
    return_rank_features: bool = False,
    return_body_frame: bool = False,
    conditions=None,
) -> dict:
    """
    Run one reverse-flow trajectory (single walker) from t=1 to t=0.

    Args:
        model: ``TriCoopDiff`` (already on ``device``; set to eval mode here).
        diffusion: ``SDPIFlow`` (velocity preconditioning + reverse integration).
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
        init_mode: ``sdpi``, ``expert_anchor``, or a preselected native-free
            ``fixed_candidate`` proposal used by controlled refinement audits.
        t_anchor: must be 1.0 for the broad sdpi prior.
        velocity_transform: optional diagnostic callback applied to physical
            velocities immediately before integration. Production callers
            leave this as ``None``.
        return_rank_features: retain the final pre-update invariant consensus
            context for separately supervised pose ranking.
        conditions: optional immutable, node-aligned PROTAC condition features.

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
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    for name, value in (("translation_velocity_scale", translation_velocity_scale),
                        ("rotation_velocity_scale", rotation_velocity_scale)):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if velocity_transform is not None and not callable(velocity_transform):
        raise TypeError("velocity_transform must be callable or None")

    # Clean templates (drop the B=1 dim).
    ligand_template = ligand_data["coords"][0].detach()  # [N_lig, 3]
    poi_template = poi_data["coords"][0].detach()        # [N_poi, 3]
    e3_coords = e3_data["coords"][0].detach()            # [N_e3, 3]

    e3_center = e3_coords.mean(dim=0, keepdim=True)
    if init_mode not in {"sdpi", "expert_anchor", "fixed_candidate"}:
        raise ValueError(f"unsupported initialization mode: {init_mode}")
    if t_anchor is None:
        t_anchor = 1.0
    if init_mode != "fixed_candidate" and t_anchor != 1.0:
        raise ValueError("broad-prior sampling requires t_anchor=1")
    if init_mode == "fixed_candidate" and not 0.0 < float(t_anchor) <= 1.0:
        raise ValueError("fixed-candidate refinement requires t_anchor in (0, 1]")
    if not isinstance(diffusion, SDPIFlow):
        raise ValueError('sampling requires SDPIFlow, not the legacy DSM process')
    if initializer.no_torsion != bool(getattr(model.expert.model, 'no_torsion', False)):
        raise ValueError('initializer and model no_torsion must match')
    ligand_pos, poi_pos, r_e3_to_poi, initialization = initializer.initialize(
        e3_coords, poi_template, ligand_data, seed=seed,
    )
    if not torch.isfinite(ligand_pos).all() or not torch.isfinite(poi_pos).all():
        raise FloatingPointError("non-finite sdpi initialization")
    if freeze_poi:
        poi_pos = poi_data["coords"].clone().detach()
        r_e3_to_poi = torch.eye(3, device=device, dtype=poi_pos.dtype).unsqueeze(0)
    initialization["freeze_poi"] = bool(freeze_poi)
    initialization['generative_process'] = FLOW_VERSION
    candidate_source = actual_candidate_source(init_mode, initialization)
    initialization["candidate_source"] = candidate_source
    source_index = SOURCE_NAMES.index(candidate_source)
    initialization["expert"] = getattr(model.expert, "provenance", {})
    ca_distances = torch.cdist(e3_coords, poi_pos[0])
    initialization["initial_geometry"] = {
        "minimum_protein_ca_distance": float(ca_distances.min()),
        "protein_ca_pairs_below_3A": int((ca_distances < 3.0).sum()),
        "minimum_e3_ca_ligand_distance": float(torch.cdist(e3_coords, ligand_pos[0]).min()),
        "minimum_poi_ca_ligand_distance": float(torch.cdist(poi_pos[0], ligand_pos[0]).min()),
    }

    schedule = (
        diffusion.get_refinement_schedule(steps, t_start=t_anchor)
        if init_mode == "fixed_candidate"
        else diffusion.get_inference_schedule(steps, t_start=t_anchor)
    )
    trajectory = [] if return_trajectory else None
    trace = [] if return_trace else None
    nonfinite_score_steps = []
    final_rank_features = None

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
        "init_mode": init_mode,
        "t_anchor": float(t_anchor),
        "initialization": initialization,
    }
    if return_trace:
        trace.append(init_record)

    for i in range(steps):
        t_tr = torch.tensor([schedule["tr_schedule"][i]], device=device, dtype=torch.float32)
        t_rot = torch.tensor([schedule["rot_schedule"][i]], device=device, dtype=torch.float32)
        t_tor = torch.tensor([schedule["tor_schedule"][i]], device=device, dtype=torch.float32)
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
        else:
            # Match vendored DiffDock's sampling driver: the final deterministic
            # step uses dt = t_i (actual_steps=19 with a length-20 schedule), not
            # t_i - t_{i+1} against an explicit zero endpoint.
            dt_tr = t_tr.clone()
            dt_rot = t_rot.clone()
            dt_tor = t_tor.clone()

        tr_sigma, rot_sigma, tor_sigma = diffusion.t_to_sigma(t_tr, t_rot, t_tor)
        pol_tr_sigma, pol_rot_sigma, _ = diffusion.t_to_sigma(
            t_tr, t_rot, t_tor,
        )
        diffusion_state = diffusion.conditioning(t_tr, dt_tr, e3_data['coords'], poi_data['coords'])

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
            e3_data, poi_d, lig_d, t_e3l=t_tr, t_pol=t_tr,
            r_e3_to_poi=r_e3_to_poi,
            receptor_crop_cutoff=receptor_crop_cutoff,
            diffusion_state=diffusion_state,
            source=source_index,
            conditions=conditions,
        )
        if return_rank_features:
            features = out.get('rank_features')
            if features is None or features.ndim != 2 or features.shape[0] != 1:
                raise ValueError('model did not return rank_features [1,D]')
            final_rank_features = features[0].detach().cpu().numpy().copy()
        if return_trace:
            step0_probe = getattr(model, '_last_step0_probe', None)
            setattr(model, '_debug_step0_probe', False)
        else:
            step0_probe = None
        predicted = diffusion.normalized_predictions(out, t_tr)
        velocities = diffusion.velocities(predicted, e3_data['coords'], poi_data['coords'])
        for branch in ('e3_ligand', 'poi_ligand'):
            velocities[branch]['tr'] = (
                velocities[branch]['tr'] * translation_velocity_scale)
            velocities[branch]['rot'] = (
                velocities[branch]['rot'] * rotation_velocity_scale)
        if velocity_transform is not None:
            velocities = velocity_transform(
                velocities, ligand_pos, poi_pos, r_e3_to_poi,
                t_tr, dt_tr, ligand_data)
            if not isinstance(velocities, dict):
                raise TypeError("velocity_transform must return a velocity dictionary")
        interface_scores = velocities
        s_q = {}
        sanitized_components = []
        for name, value in (
            ("tr", interface_scores["e3_ligand"]["tr"]),
            ("rot", interface_scores["e3_ligand"]["rot"]),
            ("tor", velocities['torsion']),
        ):
            s_q[name], had_nonfinite = _sanitize_score(value)
            if had_nonfinite:
                sanitized_components.append(f"e3_ligand.{name}")
        s_pol = {}
        for name in ("tr", "rot"):
            s_pol[name], had_nonfinite = _sanitize_score(
                interface_scores["poi_ligand"][name]
            )
            if had_nonfinite:
                sanitized_components.append(f"poi_ligand.{name}")
        if freeze_poi:
            s_pol = {name: torch.zeros_like(value) for name, value in s_pol.items()}
        if sanitized_components:
            nonfinite_score_steps.append({
                "step": i,
                "components": sanitized_components,
            })
            if return_trace:
                trace.append({
                    "stage": "rejected",
                    "step": i,
                    "reason": "nonfinite_score",
                    "components": sanitized_components,
                })
            break

        add_noise = False  # deterministic reverse flow, not temperature SDE
        ligand_pos, poi_pos, r_e3_to_poi = diffusion.step(
            ligand_pos, poi_pos, r_e3_to_poi,
            {'e3_ligand': s_q, 'poi_ligand': s_pol, 'torsion': s_q['tor']},
            dt_tr, ligand_data, freeze_poi=freeze_poi,
        )

        if return_trace:
            lig_com = ligand_pos.mean(dim=1)[0]
            poi_com = poi_pos.mean(dim=1)[0]
            step_record = {
                "stage": "step",
                "step": i,
                "add_noise": add_noise,
                "sanitized_components": sanitized_components,
                "freeze_poi": bool(freeze_poi),
                "t": {
                    "tr": float(t_tr.item()),
                    "rot": float(t_rot.item()),
                    "tor": float(t_tor.item()),
                },
                "dt": {
                    "tr": float(dt_tr.item()),
                    "rot": float(dt_rot.item()),
                    "tor": float(dt_tor.item()),
                },
                "sigma": {
                    "tr": float(tr_sigma.item()),
                    "rot": float(rot_sigma.item()),
                    "tor": float(tor_sigma.item()),
                },
                "receptor_crop_cutoff": (
                    float(receptor_crop_cutoff.item())
                    if receptor_crop_cutoff is not None else None
                ),
                "ligand_com": lig_com.detach().cpu().numpy().tolist(),
                "poi_com": poi_com.detach().cpu().numpy().tolist(),
                "ligand_pocket_dist": float(torch.linalg.vector_norm(lig_com - e3_center[0]).item()),
                "poi_e3_dist": float(torch.linalg.vector_norm(poi_com - e3_center[0]).item()),
                # Flow records use physical velocity units, not DSM scores.
                "prediction_units": "angstrom/time, radians/time",
                # Keep the historical key for report compatibility; the value
                # is the determinant of the inverse/body-frame rotation.
                "r_cum_det": float(torch.det(r_e3_to_poi[0]).item()),
                "r_e3_to_poi_det": float(torch.det(r_e3_to_poi[0]).item()),
                "score_summary": {
                    "tr": _tensor_summary(s_q["tr"]),
                    "rot": _tensor_summary(s_q["rot"]),
                    "tor": _tensor_summary(s_q["tor"]),
                    "poi_ligand_tr": _tensor_summary(s_pol["tr"]),
                    "poi_ligand_rot": _tensor_summary(s_pol["rot"]),
                },
                "step0_probe": step0_probe,
                "consensus": {
                    key: value.detach().cpu().tolist()
                    for key, value in out.get('consensus_diagnostics', {}).items()
                },
            }
            trace.append(step_record)

        if return_trajectory:
            trajectory.append(
                (ligand_pos[0].detach().cpu().numpy(), poi_pos[0].detach().cpu().numpy())
            )

    result = {
        "ligand_coords": ligand_pos[0].detach().cpu().numpy(),
        "poi_coords": poi_pos[0].detach().cpu().numpy(),
        "initialization": initialization,
        "valid": not nonfinite_score_steps,
        "nonfinite_score_steps": nonfinite_score_steps,
        "termination_reason": (
            "nonfinite_score" if nonfinite_score_steps else "completed"
        ),
    }
    if return_trajectory:
        result["trajectory"] = trajectory
    if return_body_frame:
        result["body_frame"] = r_e3_to_poi[0].detach().cpu().numpy().copy()
    if return_trace:
        result["trace"] = trace
    if return_rank_features:
        if final_rank_features is None:
            raise RuntimeError('no rank features were produced')
        result['rank_features'] = final_rank_features
    return result


def sample_ensemble(
    model,
    diffusion: SDPIFlow,
    initializer: TernaryInitializer,
    e3_data: dict,
    poi_data: dict,
    ligand_data: dict,
    steps: int = None,
    device: str = "cuda",
    n_walkers: int = 32,
    seeds=None,
    freeze_poi: bool = False,
    init_mode: str = "sdpi",
    t_anchor: float = None,
    translation_velocity_scale: float = 1.0,
    rotation_velocity_scale: float = 1.0,
    return_rank_features: bool = False,
    conditions=None,
) -> list:
    """
    Run ``n_walkers`` independent reverse-flow trajectories.

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
    if n_walkers < 1:
        raise ValueError("at least one walker is required")

    results = []
    for i in range(n_walkers):
        print(f"[sample] walker {i + 1}/{n_walkers}, seed={seeds[i]}", flush=True)
        results.append(
            sample_single(
                model, diffusion, initializer, e3_data, poi_data, ligand_data,
                steps=steps, device=device, seed=seeds[i], freeze_poi=freeze_poi,
                init_mode=init_mode,
                t_anchor=t_anchor,
                translation_velocity_scale=translation_velocity_scale,
                rotation_velocity_scale=rotation_velocity_scale,
                return_rank_features=return_rank_features,
                conditions=conditions,
            )
        )
    return results
