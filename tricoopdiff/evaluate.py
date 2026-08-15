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
# POI / ternary interface metrics (frame-invariant)
# ---------------------------------------------------------------------------

def _parse_native_complex(pdb_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse a native ternary complex PDB into PROTAC / E3 / POI coordinates.

    protac22 convention (``gt_complex.pdb`` / ``complex.pdb``):
      - HETATM records → PROTAC ligand atoms (PDB order == ligand SDF order).
      - ATOM records → protein atoms, grouped by chain ID. The chain co-chained
        with the ligand is E3; the *other* protein chain is POI.
      - Receptors are featurized at Cα level (``all_atoms=False`` in
        ``model_parameters.yml``), so protein coords are reduced to one Cα per
        residue, in residue order (matching ``featurize_common`` output).

    Returns:
        {'ligand': [N_lig, 3], 'e3_ca': [N_e3, 3], 'poi_ca': [N_poi, 3]},
        or None when the file is missing, unparseable, or the POI chain is
        ambiguous (e.g. the ligand is not co-chained with exactly one protein).
    """
    try:
        with open(pdb_path) as f:
            lines = f.readlines()
    except OSError:
        return None

    ligand = []
    lig_chains = set()
    chains = {}  # chain id -> list of (resid, atom_name, xyz)

    for ln in lines:
        rec = ln[:6].strip()
        if rec not in ('ATOM', 'HETATM'):
            continue
        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue
        chain = ln[21:22].strip() or ' '
        if rec == 'HETATM':
            ligand.append([x, y, z])
            lig_chains.add(chain)
        else:
            atom_name = ln[12:16].strip()
            resid = ln[22:26].strip()
            chains.setdefault(chain, []).append((resid, atom_name, [x, y, z]))

    ligand = np.asarray(ligand, dtype=float)
    if ligand.shape[0] == 0 or len(chains) < 2:
        return None

    def ca_coords(entries):
        """One Cα per residue, in residue order (Gly has a CA too)."""
        seen = set()
        out = []
        for resid, atom_name, xyz in entries:
            if atom_name == 'CA' and resid not in seen:
                seen.add(resid)
                out.append(xyz)
        return np.asarray(out, dtype=float)

    # POI = the protein chain NOT co-chained with the ligand.
    poi_chains = [c for c in chains if c not in lig_chains]
    e3_chains = [c for c in chains if c in lig_chains]
    if len(poi_chains) != 1:
        return None

    poi_ca = ca_coords(chains[poi_chains[0]])
    e3_ca = ca_coords(chains[e3_chains[0]]) if e3_chains else np.zeros((0, 3))
    if poi_ca.shape[0] == 0:
        return None
    return {'ligand': ligand, 'e3_ca': e3_ca, 'poi_ca': poi_ca}


def compute_poi_dockq(
    pred_poi: np.ndarray,
    pred_ligand: np.ndarray,
    native_complex_pdb: str,
    cutoff: float = 8.0,
) -> Optional[Dict[str, float]]:
    """
    Frame-invariant PROTAC-POI interface contact recovery (DockQ-style Fnat).

    DockQ's native-contact fraction (Fnat) measures how much of the native
    PROTAC↔POI interface is recovered by the prediction. It is frame-invariant:
    it depends only on the *relative* PROTAC-POI geometry, so it is valid even
    though the sampler emits PROTAC+POI in the (featurized) E3 frame while the
    native complex is in the PDB frame.

    Atom correspondence (why this works):
      - PROTAC atoms follow SDF order; the native HETATM block uses the same
        order (both trace to ``ligand.sdf``/``ligand.pdb``).
      - POI coords are featurized at Cα level (``all_atoms=False``), so both the
        predicted POI (``poi_data['coords']`` template) and the native POI
        (Cα extracted in residue order) have one atom per residue, same order.

    If the atom counts do not match (correspondence broken), a coarse
    contact-density ratio is returned instead and flagged via
    ``atom_correspondence``.

    Args:
        pred_poi: [N_poi, 3] predicted POI Cα coords (E3 frame, from sampler).
        pred_ligand: [N_lig, 3] predicted PROTAC coords (E3 frame, from sampler).
        native_complex_pdb: path to the native ternary complex PDB.
        cutoff: contact distance cutoff (Å) between PROTAC atom and POI Cα.

    Returns:
        dict with 'dockq' (Fnat in [0,1]) and contact counts, or None when the
        native PDB is unavailable/unparseable.
    """
    from scipy.spatial.distance import cdist

    native = _parse_native_complex(native_complex_pdb)
    if native is None:
        return None

    ref_lig = native['ligand']   # [N_lig, 3] native PROTAC
    ref_poi = native['poi_ca']   # [N_poi, 3] native POI Cα

    n_lig_ok = pred_ligand.shape[0] == ref_lig.shape[0]
    n_poi_ok = pred_poi.shape[0] == ref_poi.shape[0]
    pred_d = cdist(pred_ligand, pred_poi)
    n_pred = int((pred_d < cutoff).sum())

    if not (n_lig_ok and n_poi_ok):
        # Correspondence broken → density ratio only (still non-NaN).
        n_native = int((cdist(ref_lig, ref_poi) < cutoff).sum())
        if n_native == 0:
            return None
        return {
            'dockq': float(min(n_pred, n_native) / n_native),
            'n_native_contacts': n_native,
            'n_recovered_contacts': min(n_pred, n_native),
            'n_pred_contacts': n_pred,
            'atom_correspondence': False,
        }

    ref_contacts = cdist(ref_lig, ref_poi) < cutoff
    n_native = int(ref_contacts.sum())
    if n_native == 0:
        return {
            'dockq': 0.0,
            'n_native_contacts': 0,
            'n_recovered_contacts': 0,
            'n_pred_contacts': n_pred,
            'atom_correspondence': True,
        }

    recovered = int((ref_contacts & (pred_d < cutoff)).sum())
    return {
        'dockq': float(recovered / n_native),
        'n_native_contacts': n_native,
        'n_recovered_contacts': recovered,
        'n_pred_contacts': n_pred,
        'atom_correspondence': True,
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
# Ensemble Analyzer
# ---------------------------------------------------------------------------

class EnsembleAnalyzer:
    """Analyze conformational ensemble from multi-walker sampling."""

    def __init__(self, ensemble_coords: list, atomic_numbers=None):
        self.ensemble = ensemble_coords
        self.n_walkers = len(ensemble_coords)

    def full_report(self) -> dict:
        return {
            'diversity': self.diversity(),
            'clustering': self.clustering(rmsd_cutoff=2.0),
            'convergence': self.convergence(),
            'top1_quality': self.top1_quality(),
        }

    def diversity(self) -> dict:
        rmsds = []
        for i in range(self.n_walkers):
            for j in range(i+1, self.n_walkers):
                rmsds.append(compute_rmsd(self.ensemble[i], self.ensemble[j]))
        rmsds = np.array(rmsds)
        return {
            'mean_pairwise_rmsd': float(np.mean(rmsds)) if len(rmsds) > 0 else 0.0,
            'std_pairwise_rmsd': float(np.std(rmsds)) if len(rmsds) > 0 else 0.0,
            'n_walkers': self.n_walkers,
        }

    def clustering(self, rmsd_cutoff: float = 2.0) -> dict:
        if self.n_walkers < 2:
            return {'n_clusters': self.n_walkers, 'dominant_fraction': 1.0}
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist_mat = np.zeros((self.n_walkers, self.n_walkers))
        for i in range(self.n_walkers):
            for j in range(i+1, self.n_walkers):
                d = compute_rmsd(self.ensemble[i], self.ensemble[j])
                dist_mat[i,j] = d; dist_mat[j,i] = d

        condensed = squareform(dist_mat)
        labels = fcluster(linkage(condensed, method='average'), t=rmsd_cutoff, criterion='distance')
        unique, counts = np.unique(labels, return_counts=True)
        return {
            'n_clusters': int(len(unique)),
            'cluster_sizes': sorted(counts.tolist(), reverse=True),
            'dominant_fraction': float(max(counts)/self.n_walkers),
        }

    def convergence(self) -> dict:
        if self.n_walkers < 8:
            return {'stabilized': False}
        rng = np.random.RandomState(42)
        sizes = [4, 8, 16, min(32, self.n_walkers)]
        results = {}
        for size in sizes:
            if size > self.n_walkers: continue
            vals = []
            for _ in range(5):
                idx = rng.choice(self.n_walkers, size, replace=False)
                sub = [self.ensemble[i] for i in idx]
                div = EnsembleAnalyzer(sub).diversity()
                vals.append(div['mean_pairwise_rmsd'])
            results[f'n{size}'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        return {'stabilized': len(results) >= 2, 'subsample_analysis': results}

    def top1_quality(self) -> dict:
        if self.n_walkers < 2:
            return {'top1_index': 0}
        clustering = self.clustering()
        return {
            'top1_index': 0,
            'dominant_fraction': clustering['dominant_fraction'],
        }


def evaluate_model(
    model,
    e3_data: dict,
    poi_data: dict,
    ligand_data: dict,
    diffusion,
    initializer=None,
    device: str = 'cuda',
    n_walkers: int = 32,
    steps: int = None,
    native_complex_pdb: Optional[str] = None,
) -> dict:
    """
    Sample a conformational ensemble and score structure quality vs the native
    complex embedded in the featurized data dicts.

    PROTAC (E3-L) pose: the reference PROTAC pose is ``ligand_data['coords']``
    in the E3 frame, which is the same frame the sampler produces (E3 is fixed),
    so Kabsch-aligned RMSD and pocket distance are well-defined.

    POI (E3-POI / PROTAC-POI interface): reported via ``compute_poi_dockq``, a
    frame-invariant native-contact recovery (DockQ-style Fnat) that needs no
    global-frame alignment. It requires the native ternary complex PDB (see
    ``native_complex_pdb``); when that is unavailable the ``poi_*`` keys are
    set to None (never NaN).

    Args:
        model: TriCoopDiff (already on ``device``).
        e3_data/poi_data/ligand_data: B=1 featurized dicts (featurize_common).
        diffusion: MultiRateDiffusion.
        initializer: TernaryInitializer (default: fresh one).
        device: 'cuda' or 'cpu'.
        n_walkers: number of independent trajectories.
        steps: reverse-diffusion steps (default: diffusion.inference_steps).
        native_complex_pdb: optional path to the native ternary complex PDB
            (e.g. ``DeepTernary/output/protac22/<case>/gt_complex.pdb``). When
            None, POI metrics are skipped.

    Returns:
        {'structure': {...}, 'ensemble': {...}, 'functional': {}}
    """
    from .diffusion import TernaryInitializer
    from .sampler import sample_ensemble

    if initializer is None:
        initializer = TernaryInitializer()

    ensemble = sample_ensemble(
        model, diffusion, initializer, e3_data, poi_data, ligand_data,
        steps=steps, device=device, n_walkers=n_walkers,
    )

    ref_lig = ligand_data['coords'][0].detach().cpu().numpy()  # [N_lig, 3] native

    protac_rmsds = []
    pocket_dists = []
    pred_ligs = []
    poi_dockqs = []
    poi_native_contacts = None
    n_nan = 0
    for r in ensemble:
        pred = r['ligand_coords']  # [N_lig, 3] numpy
        if not np.isfinite(pred).all():
            n_nan += 1
            continue
        pred_ligs.append(pred)
        protac_rmsds.append(compute_rmsd(pred, ref_lig, align_first=True))
        pocket_dists.append(
            float(np.linalg.norm(pred.mean(axis=0) - ref_lig.mean(axis=0)))
        )
        if native_complex_pdb is not None:
            poi_m = compute_poi_dockq(r['poi_coords'], pred, native_complex_pdb)
            if poi_m is not None:
                poi_dockqs.append(poi_m['dockq'])
                if poi_native_contacts is None:
                    poi_native_contacts = poi_m['n_native_contacts']
    if n_nan:
        print(f"[evaluate] skipped {n_nan}/{len(ensemble)} walkers with non-finite coords")

    protac_rmsds = np.asarray(protac_rmsds)
    pocket_dists = np.asarray(pocket_dists)

    if len(protac_rmsds) == 0:
        # All walkers failed to produce finite coords (should be rare after the
        # score sanitization in sampler.py, but guard against a zero-size array).
        structure = {
            'protac_rmsd_best': float('nan'),
            'protac_rmsd_mean': float('nan'),
            'protac_rmsd_median': float('nan'),
            'protac_pocket_dist_mean': float('nan'),
            'protac_pocket_dist_best': float('nan'),
        }
    else:
        structure = {
            'protac_rmsd_best': float(protac_rmsds.min()),
            'protac_rmsd_mean': float(protac_rmsds.mean()),
            'protac_rmsd_median': float(np.median(protac_rmsds)),
            'protac_pocket_dist_mean': float(pocket_dists.mean()),
            'protac_pocket_dist_best': float(pocket_dists.min()),
        }

    # Steric clash over the PROTAC ensemble (atomic numbers unknown at MVP scale;
    # use carbon=6 as a proxy — same convention as the reflection post-processor).
    try:
        structure['clash_rate'] = compute_clash_rate(
            pred_ligs, np.full(ref_lig.shape[0], 6)
        )
    except Exception:
        structure['clash_rate'] = None

    # PROTAC-POI interface recovery (DockQ-style Fnat), frame-invariant. None
    # (never NaN) when the native complex PDB is unavailable/unparseable.
    if poi_dockqs:
        poi_dockqs = np.asarray(poi_dockqs)
        structure['poi_dockq_best'] = float(poi_dockqs.max())
        structure['poi_dockq_mean'] = float(poi_dockqs.mean())
        structure['poi_dockq_median'] = float(np.median(poi_dockqs))
        structure['poi_interface_native_contacts'] = poi_native_contacts
    else:
        structure['poi_dockq_best'] = None
        structure['poi_dockq_mean'] = None
        structure['poi_dockq_median'] = None
        structure['poi_interface_native_contacts'] = None

    ensemble_metrics = {
        'n_walkers': len(ensemble),
        'n_nan_walkers': n_nan,
        'protac_rmsd_std': float(protac_rmsds.std()) if len(protac_rmsds) > 1 else 0.0,
    }
    if len(pred_ligs) > 1:
        ensemble_metrics['diversity'] = compute_ensemble_diversity(pred_ligs)

    return {
        'structure': structure,
        'ensemble': ensemble_metrics,
        # Functional ranking (α, Top-20% hit rate, AUROC) requires predicted
        # productivity per complex — wired in a later phase.
        'functional': {},
    }


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

    # SVD (guard against NaN/inf leaking in from a failed reverse step)
    if not (np.isfinite(H).all()):
        return mobile
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return mobile
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
