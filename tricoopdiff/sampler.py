"""
Reverse-diffusion sampling for TriCoopDiff.

Composes the deterministic reverse flow in ``dual_consensus.py`` into a full
t=1→0 denoising loop that produces a predicted PROTAC pose (E3 reference frame)
and a predicted E3-POI pose.

Reference frame: E3 fixed. Two ligand hypotheses and POI are initialized by
DualConsensusFlow at t=1 (the initializer argument supplies no_torsion only),
rebuilding the two subgraphs + ternary graph inside ``model.forward`` at every
step from the current (noisy) coordinates.

This module is self-contained: it takes already-featurized B=1 data dicts
(``featurize_common.featurize_complex``) and the built model + diffusion, so it
does NOT import ``../DiffDock``.
"""

import torch

from .initialization import TernaryInitializer
from .dual_consensus import DualConsensusFlow as SDPIFlow, FLOW_VERSION


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
) -> dict:
    from .dual_consensus import dual_sample
    return dual_sample(model, diffusion, initializer, e3_data, poi_data, ligand_data,
                       steps, device, seed, return_trajectory, return_trace,
                       freeze_poi, init_mode, t_anchor)

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
            )
        )
    return results
