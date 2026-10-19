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

from functools import wraps

import numpy as np
import torch

from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.sdpi_flow import SDPIFlow, FLOW_VERSION
from tricoopdiff.models.assembly import (
    DIRECT_RANK_FEATURE_STATE, LEGACY_RANK_FEATURE_STATE, SOURCE_NAMES,
)
from tricoopdiff.runtime import evaluation_context


def actual_candidate_source(init_mode, initialization):
    """Return the source actually produced, including BiTSeed fallback."""
    if init_mode == "sdpi":
        source = "sdpi"
    elif init_mode == "fixed_candidate":
        source = initialization.get("candidate_source")
    elif init_mode == "expert_anchor":
        source = "sdpi" if initialization.get("fallback") else "bitseed"
    elif init_mode == "protac_binary_anchor":
        source = "sdpi"
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


def _nonfinite_components(velocities):
    """Reject invalid motion without replacing values or rebuilding predictions."""
    components = [(f'{branch}.{name}', velocities[branch][name])
                  for branch in ('e3_ligand', 'poi_ligand') for name in ('tr', 'rot')]
    components.append(('e3_ligand.tor', velocities['torsion']))
    return [name for name, value in components if not torch.isfinite(value).all()]


def _sampling_context(function):
    """Inference must leave the caller's training mode and RNG unchanged."""
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        with evaluation_context(model):
            return function(model, *args, **kwargs)
    return wrapped


@_sampling_context
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
    protac_anchor_mode = init_mode == 'protac_binary_anchor'
    if init_mode not in {"sdpi", "expert_anchor", "fixed_candidate",
                         "protac_binary_anchor"}:
        raise ValueError(f"unsupported initialization mode: {init_mode}")
    if t_anchor is None:
        t_anchor = 1.0
    if init_mode != "fixed_candidate" and t_anchor != 1.0:
        raise ValueError("broad-prior sampling requires t_anchor=1")
    if init_mode == "fixed_candidate" and not 0.0 < float(t_anchor) <= 1.0:
        raise ValueError("fixed-candidate refinement requires t_anchor in (0, 1]")
    if protac_anchor_mode:
        if t_anchor != 1.0 or freeze_poi:
            raise ValueError('binary-anchor assembly requires full-time, moving-POI execution')
        if (not isinstance(conditions, dict) or
                conditions.get('contract') != 'external_binary_exact_bond_anchor_v1' or
                not getattr(getattr(model, 'trifuse', None), 'config', {}).get('protac_anchor_contract')):
            raise ValueError('binary-anchor sampling requires a PROTAC model and external binary condition')
        from tricoopdiff.initialization.protac import validate_anchor_graph_identity
        validate_anchor_graph_identity(conditions, ligand_data.get('mol'))
        if conditions.get('mapping_hypotheses'):
            from tricoopdiff.initialization.protac import select_binary_mapping_hypothesis
            if seed is None:
                seed = int(np.random.randint(0, 2**31 - 1))
            index = int(seed) % len(conditions['mapping_hypotheses'])
            conditions = select_binary_mapping_hypothesis(conditions, index)
    if not isinstance(diffusion, SDPIFlow):
        raise ValueError('sampling requires SDPIFlow, not the legacy DSM process')
    if initializer.no_torsion != bool(getattr(model.expert.model, 'no_torsion', False)):
        raise ValueError('initializer and model no_torsion must match')
    ligand_pos, poi_pos, r_e3_to_poi, initialization = initializer.initialize(
        e3_coords, poi_template, ligand_data, seed=seed,
    )
    if protac_anchor_mode:
        from tricoopdiff.initialization.protac import assemble_from_binary_anchors
        assembled = assemble_from_binary_anchors(
            ligand_pos[0], poi_template,
            torch.as_tensor(conditions['anchor1_xyz'], device=ligand_pos.device,
                            dtype=ligand_pos.dtype),
            torch.as_tensor(conditions['anchor2_xyz'], device=ligand_pos.device,
                            dtype=ligand_pos.dtype),
            conditions['anchor1_atoms'], conditions['anchor2_atoms'])
        ligand_pos = assembled['ligand'].unsqueeze(0)
        poi_pos = assembled['protein2'].unsqueeze(0)
        r_e3_to_poi = assembled['protein2_rotation'].unsqueeze(0)
        initialization.update(init_mode=init_mode,
            binary_anchor_contract=conditions['contract'],
            native_ternary_coordinates_used=False,
            anchor_fit_rmsd=assembled['anchor_rmsd'].detach().cpu().tolist(),
            mapping_hypothesis_index=conditions.get('mapping_hypothesis_index'))
    if not torch.isfinite(ligand_pos).all() or not torch.isfinite(poi_pos).all():
        raise FloatingPointError("non-finite sdpi initialization")
    if ligand_pos.shape != ligand_data['coords'].shape or poi_pos.shape != poi_data['coords'].shape:
        raise ValueError('initialization atom counts must match the indexed inputs')
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

    trajectory = [] if return_trajectory else None
    trace = [] if return_trace else None
    nonfinite_score_steps = []

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

    for i, (t_tr, dt_tr) in enumerate(diffusion.reverse_times(ligand_pos, steps, t_anchor)):
        t_rot = t_tor = t_tr
        dt_rot = dt_tor = dt_tr
        tr_sigma, rot_sigma, tor_sigma = diffusion.t_to_sigma(t_tr, t_rot, t_tor)
        if bool(getattr(model.expert.model, "dynamic_max_cross", False)):
            receptor_crop_cutoff = tr_sigma * 3 + 20
        else:
            receptor_crop_cutoff = None

        probe_enabled = getattr(model, '_debug_step0_probe', False)
        if return_trace:
            setattr(model, '_debug_step0_probe', True)
            setattr(model, '_last_step0_probe', None)
        try:
            out = diffusion.forward_state(
                model, e3_data, poi_data, ligand_data, ligand_pos, poi_pos,
                r_e3_to_poi, t_tr, dt_tr, source_index, conditions)
        finally:
            if return_trace:
                setattr(model, '_debug_step0_probe', probe_enabled)
        if return_trace:
            step0_probe = getattr(model, '_last_step0_probe', None)
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
            original_mode = velocities.get('rigid_output_mode', 'dual_interfaces_v1')
            original_torsion_mode = velocities.get('torsion_update_mode')
            velocities = velocity_transform(
                velocities, ligand_pos, poi_pos, r_e3_to_poi,
                t_tr, dt_tr, ligand_data)
            if not isinstance(velocities, dict):
                raise TypeError("velocity_transform must return a velocity dictionary")
            if velocities.get('rigid_output_mode', 'dual_interfaces_v1') != original_mode:
                raise ValueError('velocity_transform must preserve the rigid output mode')
            if velocities.get('torsion_update_mode') != original_torsion_mode:
                raise ValueError('velocity_transform must preserve the torsion update mode')
        model_config = getattr(getattr(model, 'trifuse', None), 'config', {})
        motion_contract = model_config.get('motion_contract', 'sdpi_flow_velocity_v1')
        if motion_contract == 'bounded_iterative_correction_v1':
            velocities = diffusion.bounded_correction(velocities)
            action_dt = torch.ones_like(dt_tr)
        elif motion_contract == 'sdpi_flow_velocity_v1':
            action_dt = dt_tr
        else:
            raise ValueError(f'unknown saved motion contract: {motion_contract!r}')
        s_q = {**velocities['e3_ligand'], 'tor': velocities['torsion']}
        s_pol = velocities['poi_ligand']
        sanitized_components = _nonfinite_components(velocities)
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
        if protac_anchor_mode:
            from tricoopdiff.initialization.protac import anchored_flow_step
            ligand_pos, poi_pos, r_e3_to_poi, anchor_errors = anchored_flow_step(
                ligand_pos, poi_pos, r_e3_to_poi, velocities, action_dt,
                ligand_data, poi_template.unsqueeze(0), conditions)
        else:
            ligand_pos, poi_pos, r_e3_to_poi = diffusion.step(
                ligand_pos, poi_pos, r_e3_to_poi,
                velocities,
                action_dt, ligand_data, freeze_poi=freeze_poi,
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
                "anchor_fit_rmsd": (anchor_errors.detach().cpu().tolist()
                                     if protac_anchor_mode else None),
                "rigid_output_mode": velocities.get('rigid_output_mode', 'dual_interfaces_v1'),
                "torsion_update_mode": velocities.get('torsion_update_mode', 'legacy_kabsch_v1'),
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
                    key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
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
    if return_rank_features and not nonfinite_score_steps:
        final = diffusion.forward_state(
            model, e3_data, poi_data, ligand_data, ligand_pos, poi_pos,
            r_e3_to_poi, ligand_pos.new_zeros(1), ligand_pos.new_zeros(1),
            source_index, conditions)
        features = final.get('rank_features')
        if (features is None or features.ndim != 2 or features.shape[0] != 1
                or not torch.isfinite(features).all()):
            raise ValueError('model did not return finite final rank_features [1,D]')
        final_rank_features = features[0].detach().cpu().numpy().copy()
        result['rank_features'] = final_rank_features
        trifuse = getattr(model, 'trifuse', None)
        state = (DIRECT_RANK_FEATURE_STATE if getattr(trifuse, 'direct_ternary_motion', False)
                 else LEGACY_RANK_FEATURE_STATE)
        result['rank_feature_state'] = state
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
