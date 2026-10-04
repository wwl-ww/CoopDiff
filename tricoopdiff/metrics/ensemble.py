"""Conformational-ensemble and calibrated cooperativity analysis.

Population-derived free energies in this module describe the model's empirical
sampling distribution. They are not absolute binding free energies unless the
sampling protocol has independently been shown to reproduce equilibrium.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


R_KCAL_MOL_K = 0.00198720425864083


def fixed_frame_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    """Return RMSD without alignment for poses in the same receptor frame."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError(f"coordinate shape mismatch: {first.shape} vs {second.shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("coordinates must be finite")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def pairwise_pose_rmsd(poses: Sequence[np.ndarray]) -> np.ndarray:
    """Build a symmetric fixed-frame RMSD matrix for equal-shape poses."""
    poses = [np.asarray(pose, dtype=float) for pose in poses]
    matrix = np.zeros((len(poses), len(poses)), dtype=float)
    for first in range(len(poses)):
        for second in range(first + 1, len(poses)):
            value = fixed_frame_rmsd(poses[first], poses[second])
            matrix[first, second] = value
            matrix[second, first] = value
    return matrix


def cluster_poses(poses: Sequence[np.ndarray], rmsd_cutoff: float = 2.0) -> np.ndarray:
    """Average-linkage cluster poses using fixed-receptor-frame RMSD."""
    return _cluster_distances(pairwise_pose_rmsd(poses), rmsd_cutoff)


def _cluster_distances(matrix: np.ndarray, rmsd_cutoff: float) -> np.ndarray:
    """Cluster a precomputed matrix so reports calculate pair distances once."""
    if not np.isfinite(rmsd_cutoff) or rmsd_cutoff <= 0:
        raise ValueError("rmsd_cutoff must be positive")
    count = len(matrix)
    if count == 0:
        return np.asarray([], dtype=int)
    if count == 1:
        return np.asarray([1], dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(matrix, checks=False)
    return fcluster(
        linkage(condensed, method="average"),
        t=rmsd_cutoff, criterion="distance",
    ).astype(int)


def population_free_energies(
    labels: Sequence[int], temperature_k: float = 298.15,
) -> List[Dict[str, float]]:
    """Convert observed cluster populations to relative empirical free energy."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must contain at least one one-dimensional assignment")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    delta_g = -R_KCAL_MOL_K * temperature_k * np.log(
        probabilities / probabilities.max()
    )
    order = np.argsort(-counts, kind="stable")
    return [
        {
            "cluster": int(unique[index]),
            "count": int(counts[index]),
            "probability": float(probabilities[index]),
            "relative_free_energy_kcal_mol": float(delta_g[index]),
        }
        for index in order
    ]


def cooperativity_from_kd(kd_binary: float, kd_ternary: float) -> float:
    """Return experimental cooperativity alpha = Kd_binary / Kd_ternary."""
    if kd_binary <= 0 or kd_ternary <= 0:
        raise ValueError("dissociation constants must be positive")
    return float(kd_binary / kd_ternary)


def cooperativity_from_ddg(
    ddg_ternary_minus_binary_kcal_mol: float,
    temperature_k: float = 298.15,
) -> float:
    """Convert calibrated ΔΔG=(ΔG_ternary−ΔG_binary) to alpha."""
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    exponent = -float(ddg_ternary_minus_binary_kcal_mol) / (
        R_KCAL_MOL_K * temperature_k
    )
    return float(np.exp(np.clip(exponent, -700.0, 700.0)))


def cooperativity_ranking(
    predicted_ddg: Sequence[float], experimental_alpha: Sequence[float],
    temperature_k: float = 298.15, top_fraction: float = 0.2,
) -> Dict[str, float]:
    """Evaluate calibrated ΔΔG predictions against measured alpha values."""
    predicted_ddg = np.asarray(predicted_ddg, dtype=float)
    experimental_alpha = np.asarray(experimental_alpha, dtype=float)
    if predicted_ddg.shape != experimental_alpha.shape or predicted_ddg.ndim != 1:
        raise ValueError("predicted_ddg and experimental_alpha must be equal 1-D arrays")
    if predicted_ddg.size < 2 or not 0 < top_fraction <= 1:
        raise ValueError("ranking needs at least two samples and a valid top_fraction")
    if (experimental_alpha <= 0).any() or not np.isfinite(
            np.concatenate([predicted_ddg, experimental_alpha])).all():
        raise ValueError("ranking inputs must be finite and alpha must be positive")
    if np.ptp(predicted_ddg) == 0 or np.ptp(experimental_alpha) == 0:
        raise ValueError("Spearman ranking requires variability in both inputs")
    from scipy.stats import spearmanr

    predicted_alpha = np.asarray([
        cooperativity_from_ddg(value, temperature_k) for value in predicted_ddg
    ])
    rho, pvalue = spearmanr(predicted_alpha, experimental_alpha)
    top_count = max(1, int(np.ceil(predicted_ddg.size * top_fraction)))
    predicted_top = set(np.argsort(-predicted_alpha)[:top_count].tolist())
    observed_top = set(np.argsort(-experimental_alpha)[:top_count].tolist())
    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
        "top_fraction": float(top_fraction),
        "top_n": int(top_count),
        "top_n_hit_rate": float(len(predicted_top & observed_top) / top_count),
    }


def energy_population_association(
    labels: Sequence[int], energies_kcal_mol: Sequence[float],
) -> Optional[Dict[str, float]]:
    """Compare cluster population with external per-pose physical energies."""
    labels = np.asarray(labels, dtype=int)
    energies = np.asarray(energies_kcal_mol, dtype=float)
    if labels.shape != energies.shape or labels.ndim != 1:
        raise ValueError("labels and energies must be equal 1-D arrays")
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return None
    cluster_energy = np.asarray([
        np.median(energies[labels == label]) for label in unique
    ])
    if not np.isfinite(cluster_energy).all():
        raise ValueError("energies must be finite")
    from scipy.stats import rankdata, spearmanr

    if np.ptp(cluster_energy) == 0 or np.ptp(counts) == 0:
        rho, pvalue = None, None
    else:
        rho_value, pvalue_value = spearmanr(-cluster_energy, counts)
        rho, pvalue = float(rho_value), float(pvalue_value)
    dominant = int(np.argmax(counts))
    energy_ranks = rankdata(cluster_energy, method="min")
    return {
        "n_clusters": int(unique.size),
        "favorability_population_spearman_rho": rho,
        "spearman_pvalue": pvalue,
        "dominant_cluster_energy_rank": int(energy_ranks[dominant]),
    }


def analyze_ensemble(
    poses: Sequence[np.ndarray], rmsd_cutoff: float = 2.0,
    temperature_k: float = 298.15,
    energies_kcal_mol: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return clustering, diversity, populations, and optional energy agreement."""
    matrix = pairwise_pose_rmsd(poses)
    labels = _cluster_distances(matrix, rmsd_cutoff)
    upper = (matrix[np.triu_indices(len(poses), k=1)]
             if len(poses) else np.asarray([]))
    populations = population_free_energies(labels, temperature_k) if len(labels) else []
    report = {
        "n_poses": len(poses),
        "rmsd_cutoff_angstrom": float(rmsd_cutoff),
        "distance_definition": "fixed-E3-frame RMSD over concatenated PROTAC+POI coordinates",
        "mean_pairwise_rmsd": float(upper.mean()) if upper.size else 0.0,
        "std_pairwise_rmsd": float(upper.std()) if upper.size else 0.0,
        "n_clusters": len(populations),
        "dominant_fraction": populations[0]["probability"] if populations else 0.0,
        "cluster_labels": labels.tolist(),
        "clusters": populations,
        "free_energy_interpretation": (
            "-RT ln(p_cluster/p_max) from diffusion walker frequencies; "
            "empirical model-sampling potential, not absolute binding free energy"
        ),
    }
    if energies_kcal_mol is not None:
        report["energy_population"] = energy_population_association(
            labels, energies_kcal_mol,
        )
    return report
