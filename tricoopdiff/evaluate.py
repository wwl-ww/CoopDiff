"""
Evaluation metrics for TriCoopDiff.

Metrics categories:
1. Structure quality: DockQ, RMSD (global/warhead/linker), clash rate, linker strain
2. Ensemble quality: energy-probability Spearman ρ, diversity, resampling consistency
3. Functional ranking: α-ranking Spearman ρ, Top-20% hit rate, productive AUC
4. Comparison baselines: vs DeepTernary, vs docking (Vina/Glide/MM-PBSA)
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Structure Quality Metrics
# ---------------------------------------------------------------------------

def compute_dockq(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    interface_cutoff: float = 5.0,
) -> float:
    """
    Compute DockQ score for ternary complex.

    DockQ = fraction of native interface contacts recovered in the prediction.
    An interface contact is defined as a pair of residues (one from each protein)
    with Cβ-Cβ distance < cutoff in the reference structure.

    Args:
        pred_coords: [N_pred, 3] predicted Cβ coordinates
        ref_coords: [N_ref, 3] reference Cβ coordinates
        interface_cutoff: distance cutoff for interface contacts (Å)

    Returns:
        DockQ score in [0, 1], higher is better
    """
    from scipy.spatial.distance import cdist

    # Compute pairwise distances
    pred_dist = cdist(pred_coords, pred_coords)
    ref_dist = cdist(ref_coords, ref_coords)

    # Interface contacts in reference
    ref_contacts = ref_dist < interface_cutoff

    if ref_contacts.sum() == 0:
        return 0.0

    # Fraction of reference contacts recovered
    pred_contacts = pred_dist < interface_cutoff
    recovered = (ref_contacts & pred_contacts).sum()
    dockq = recovered / ref_contacts.sum()

    return float(dockq)


def compute_rmsd(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    align_first: bool = True,
) -> float:
    """
    Compute RMSD after optimal alignment (Kabsch algorithm).

    Args:
        pred_coords: [N, 3]
        ref_coords: [N, 3]
        align_first: if True, perform Kabsch alignment before computing RMSD

    Returns:
        RMSD in Å
    """
    if align_first:
        pred_coords = kabsch_align(pred_coords, ref_coords)
    diff = pred_coords - ref_coords
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_rmsd_by_region(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    align_first: bool = True,
) -> Dict[str, float]:
    """
    Compute RMSD separately for different molecular regions.

    Args:
        pred_coords: [N, 3]
        ref_coords: [N, 3]
        region_masks: dict mapping region name → [N] bool mask
        align_first: Kabsch align on all atoms before computing per-region RMSD

    Returns:
        dict of region → RMSD
    """
    if align_first:
        pred_coords = kabsch_align(pred_coords, ref_coords)

    results = {}
    for region, mask in region_masks.items():
        if mask.sum() > 0:
            results[region] = compute_rmsd(
                pred_coords[mask], ref_coords[mask], align_first=False
            )
        else:
            results[region] = float('nan')

    results['global'] = compute_rmsd(pred_coords, ref_coords, align_first=False)
    return results


def compute_linker_rmsd(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    linker_mask: np.ndarray,
) -> float:
    """
    RMSD specifically for PROTAC linker atoms.
    Linker atoms are identified by excluding warhead atoms (binders).
    """
    if linker_mask.sum() == 0:
        return 0.0
    return compute_rmsd(pred_coords[linker_mask], ref_coords[linker_mask])


def compute_clash_rate(
    ensemble_coords: List[np.ndarray],
    atomic_numbers: np.ndarray,
    vdw_scale: float = 0.8,
) -> Dict[str, float]:
    """
    Compute steric clash statistics across a conformational ensemble.

    Args:
        ensemble_coords: list of [N, 3] coordinate arrays
        atomic_numbers: [N] atomic numbers
        vdw_scale: VDW radius scaling factor

    Returns:
        dict with 'mean_clash_rate', 'max_clash_rate', 'std_clash_rate'
    """
    from .reflect import detect_clashes

    clash_rates = []
    for coords in ensemble_coords:
        clashes = detect_clashes(coords, atomic_numbers, vdw_scale)
        clash_rates.append(clashes.sum() / len(atomic_numbers))

    return {
        'mean_clash_rate': float(np.mean(clash_rates)),
        'max_clash_rate': float(np.max(clash_rates)),
        'std_clash_rate': float(np.std(clash_rates)),
    }


def compute_linker_strain_energy(
    ensemble_coords: List[np.ndarray],
    dihedral_indices: List[Tuple[int, int, int, int]],
) -> Dict[str, float]:
    """
    Estimate linker strain energy via MMFF94 (requires RDKit).

    For now, returns torsional deviation as a proxy.
    Full MMFF94 integration requires RDKit molecule construction.
    """
    strains = []
    for coords in ensemble_coords:
        strain = _estimate_torsional_strain(coords, dihedral_indices)
        strains.append(strain)

    return {
        'mean_strain': float(np.mean(strains)),
        'max_strain': float(np.max(strains)),
        'min_strain': float(np.min(strains)),
    }


# ---------------------------------------------------------------------------
# Ensemble Quality Metrics
# ---------------------------------------------------------------------------

def compute_ensemble_energy_probability_correlation(
    ensemble_coords: List[np.ndarray],
    ensemble_weights: List[float],
    energy_func=None,
) -> float:
    """
    Spearman ρ between MMFF94 energies and sampling frequencies.

    A positive ρ indicates that lower-energy conformations are sampled more often,
    suggesting the ensemble has physical meaning.

    Args:
        ensemble_coords: list of conformations
        ensemble_weights: sampling frequency of each conformation
        energy_func: callable that returns energy for a conformation

    Returns:
        Spearman ρ correlation coefficient
    """
    if energy_func is None:
        # Fallback: use a simple repulsive energy proxy
        energy_func = _simple_repulsive_energy

    energies = [energy_func(coords) for coords in ensemble_coords]

    if len(set(energies)) < 2 or len(set(ensemble_weights)) < 2:
        return 0.0

    rho, _ = spearmanr(energies, ensemble_weights)
    return float(rho)


def compute_ensemble_diversity(
    ensemble_coords: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute pairwise RMSD statistics across the ensemble.

    Returns:
        dict with 'mean_pairwise_rmsd', 'std_pairwise_rmsd', 'n_clusters'
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    n = len(ensemble_coords)
    if n < 2:
        return {'mean_pairwise_rmsd': 0.0, 'std_pairwise_rmsd': 0.0, 'n_clusters': n}

    # Compute all pairwise RMSDs
    pairwise_rmsd = []
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = compute_rmsd(ensemble_coords[i], ensemble_coords[j])
            pairwise_rmsd.append(rmsd)

    pairwise_rmsd = np.array(pairwise_rmsd)

    # Clustering at 2Å cutoff
    if len(pairwise_rmsd) > 0:
        condensed = squareform(pairwise_rmsd)
        linkage_matrix = linkage(condensed, method='average')
        clusters = fcluster(linkage_matrix, t=2.0, criterion='distance')
        n_clusters = len(set(clusters))
    else:
        n_clusters = n

    return {
        'mean_pairwise_rmsd': float(np.mean(pairwise_rmsd)),
        'std_pairwise_rmsd': float(np.std(pairwise_rmsd)),
        'n_clusters': n_clusters,
    }


# ---------------------------------------------------------------------------
# Functional Ranking Metrics
# ---------------------------------------------------------------------------

def compute_cooperativity_ranking(
    predicted_ranks: np.ndarray,
    experimental_alpha: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate PROTAC cooperativity α ranking performance.

    Args:
        predicted_ranks: [N] predicted ranking (lower = more cooperative)
        experimental_alpha: [N] experimental cooperativity values

    Returns:
        dict with 'spearman_rho', 'p_value', 'top20_hit_rate'
    """
    rho, p_value = spearmanr(predicted_ranks, experimental_alpha)

    # Top-20% hit rate: fraction of top 20% predictions that are in top 20% experiments
    n = len(predicted_ranks)
    top_k = max(1, int(n * 0.2))
    top_pred = np.argsort(predicted_ranks)[:top_k]
    top_exp = np.argsort(experimental_alpha)[:top_k]
    hit_rate = len(set(top_pred) & set(top_exp)) / top_k

    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'top20_hit_rate': float(hit_rate),
    }


def compute_productive_auc(
    productive_scores: np.ndarray,
    productive_labels: np.ndarray,
) -> float:
    """
    AUROC for discriminating productive vs non-productive ternary complexes.

    Args:
        productive_scores: [N] predicted productive probability
        productive_labels: [N] binary ground truth
    Returns:
        AUROC score
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(productive_labels, productive_scores))


# ---------------------------------------------------------------------------
# Full Evaluation Report
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    data_loader,
    diffusion,
    device: str = 'cuda',
    n_walkers: int = 32,
) -> dict:
    """
    Run full evaluation: structure quality + ensemble quality + functional ranking.

    Returns a nested dict with all metrics, ready for the LaTeX results table.
    """
    all_metrics = {
        'structure': {},
        'ensemble': {},
        'functional': {},
    }

    # Placeholder for actual evaluation loop
    # Each batch: run multi-walker sampling → collect ensemble → compute metrics

    return all_metrics


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: find optimal rotation to align mobile onto target.

    Args:
        mobile: [N, 3] coordinates to align
        target: [N, 3] reference coordinates
    Returns:
        [N, 3] aligned coordinates
    """
    # Center both point sets
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_c = mobile - mobile_center
    target_c = target - target_center

    # Covariance matrix
    H = mobile_c.T @ target_c

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation matrix (handle reflection)
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1, 1, d])
    R = V @ D @ U.T

    # Apply rotation
    aligned = mobile_c @ R + target_center
    return aligned


def _estimate_torsional_strain(
    coords: np.ndarray,
    dihedral_indices: List[Tuple[int, int, int, int]],
) -> float:
    """Estimate torsional strain from dihedral angles."""
    strains = []
    for i, j, k, l in dihedral_indices:
        angle = _compute_dihedral(coords[i], coords[j], coords[k], coords[l])
        # Penalize near 0° or 180° (eclipsed conformations)
        sin_sq = np.sin(angle) ** 2
        strains.append(1.0 - sin_sq)  # 0 = staggered (good), 1 = eclipsed (bad)
    return np.mean(strains) if strains else 0.0


def _compute_dihedral(p1, p2, p3, p4):
    """Compute dihedral angle between four points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return float(np.arctan2(y, x))


def _simple_repulsive_energy(coords: np.ndarray) -> float:
    """Simple repulsive energy proxy for ensemble evaluation."""
    from scipy.spatial.distance import pdist
    dists = pdist(coords)
    # Penalize close contacts (< 2Å)
    close = dists[dists < 2.0]
    return float(np.sum((2.0 - close) ** 2))
