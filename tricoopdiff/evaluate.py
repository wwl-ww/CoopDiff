"""Sample and score ternary poses in the fixed E3 reference frame.

Protein-only full-atom DockQ is the benchmark metric. Approximate contact and
ligand metrics are diagnostics, not substitutes for that protocol. Ensemble
diversity uses fixed-frame poses; functional prediction is not implemented.
"""

import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Structure Quality Metrics
# ---------------------------------------------------------------------------

# NOTE (BUG-1): `compute_dockq` was removed here. It computed *intra*-molecule
# contact-map similarity (`cdist(coords, coords)` on a single coordinate set),
# which is NOT DockQ (DockQ = Fnat across two proteins + iRMS/Lrms). It was
# also dead code — `evaluate_model` uses `compute_e3_interface_dockq` /
# `compute_poi_dockq` instead, and no other module imported or called it.

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


def compute_clash_rate(
    ensemble_coords: List[np.ndarray],
    atomic_numbers: np.ndarray,
    vdw_scale: float = 0.8,
    bonds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute steric clash statistics across a conformational ensemble.

    Args:
        ensemble_coords: list of [N, 3] coordinate arrays
        atomic_numbers: [N] atomic numbers
        vdw_scale: VDW radius scaling factor
        bonds: optional bond graph ([2, E] int array or (i, j) iterable) so 1-2
            / 1-3 / 1-4 pairs are excluded (normal covalent geometry). Without
            it only 1-2 pairs are skipped via a covalent-radius heuristic, which
            leaves ring 1-3 atoms flagged and inflates the rate.

    Returns:
        dict with 'mean_clash_rate', 'max_clash_rate', 'std_clash_rate'
    """
    from tricoopdiff.geometry.reflect import detect_clashes

    clash_rates = []
    for coords in ensemble_coords:
        clashes = detect_clashes(coords, atomic_numbers, vdw_scale, bonds)
        clash_rates.append(clashes.sum() / len(atomic_numbers))

    return {
        'mean_clash_rate': float(np.mean(clash_rates)),
        'max_clash_rate': float(np.max(clash_rates)),
        'std_clash_rate': float(np.std(clash_rates)),
    }


# ---------------------------------------------------------------------------
# POI / ternary interface metrics (frame-invariant)
# ---------------------------------------------------------------------------

def compute_deepternary_dockq(
    native_ligand: np.ndarray,
    native_e3: np.ndarray,
    native_poi: np.ndarray,
    pred_ligand: np.ndarray,
    pred_e3: np.ndarray,
    pred_poi: np.ndarray,
    cutoff: float = 5.0,
) -> Optional[Dict[str, float]]:
    """Match DeepTernary's ``simple_dockq.cal_approx_dockq`` convention.

    E3 is the receptor; PROTAC and POI are concatenated as the mobile partner.
    Fnat is recovery of native interface-node identities (not contact pairs),
    Lrms is measured in the fixed-E3 frame, and iRMS uses a Kabsch fit of the
    native-interface nodes. This is a diagnostic approximation, not the protein-only full-atom
    DockQ used as the primary benchmark metric.
    """
    from scipy.spatial.distance import cdist

    arrays = (native_ligand, native_e3, native_poi, pred_ligand, pred_e3, pred_poi)
    if any(array.ndim != 2 or array.shape[1] != 3 for array in arrays):
        return None
    if (
        native_ligand.shape != pred_ligand.shape
        or native_e3.shape != pred_e3.shape
        or native_poi.shape != pred_poi.shape
    ):
        return None

    native_mobile = np.concatenate([native_ligand, native_poi], axis=0)
    pred_mobile = np.concatenate([pred_ligand, pred_poi], axis=0)

    def pocket_indices(mobile, receptor):
        distances = cdist(mobile, receptor)
        mobile_ids = np.flatnonzero(distances.min(axis=1) <= cutoff)
        receptor_ids = np.flatnonzero(distances.min(axis=0) <= cutoff)
        return mobile_ids, receptor_ids

    native_mobile_ids, native_receptor_ids = pocket_indices(native_mobile, native_e3)
    pred_mobile_ids, pred_receptor_ids = pocket_indices(pred_mobile, pred_e3)
    native_nodes = {
        *(f"{index}_lig" for index in native_mobile_ids.tolist()),
        *(f"{index}_rec" for index in native_receptor_ids.tolist()),
    }
    if not native_nodes:
        return None
    pred_nodes = {
        *(f"{index}_lig" for index in pred_mobile_ids.tolist()),
        *(f"{index}_rec" for index in pred_receptor_ids.tolist()),
    }
    fnat = len(native_nodes & pred_nodes) / len(native_nodes)

    lrms = float(np.sqrt(np.mean(np.sum((native_mobile - pred_mobile) ** 2, axis=1))))
    native_interface = np.concatenate(
        [native_mobile[native_mobile_ids], native_e3[native_receptor_ids]], axis=0
    )
    pred_interface = np.concatenate(
        [pred_mobile[native_mobile_ids], pred_e3[native_receptor_ids]], axis=0
    )
    aligned_interface = kabsch_align(pred_interface, native_interface)
    irms = float(
        np.sqrt(np.mean(np.sum((native_interface - aligned_interface) ** 2, axis=1)))
    )
    dockq = (
        fnat
        + 1.0 / (1.0 + (irms / 1.5) ** 2)
        + 1.0 / (1.0 + (lrms / 8.5) ** 2)
    ) / 3.0
    return {
        "dockq": float(dockq),
        "fnat": float(fnat),
        "lrms": lrms,
        "irms": irms,
        "n_native_interface_nodes": len(native_nodes),
        "n_recovered_interface_nodes": len(native_nodes & pred_nodes),
        "cutoff": float(cutoff),
    }

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
            # Match ProDy's default blank/A alternate-location selection.
            if ln[16:17] not in (' ', 'A'):
                continue
            atom_name = ln[12:16].strip()
            resid = ln[22:27]  # insertion-coded residues are separate nodes
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
        Contact recovery ('dockq') only with matching atom counts. Otherwise
        return 'contact_density_ratio' as a separate, non-correspondence
        diagnostic; it is not Fnat. Return None for an unavailable native.
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
        # A count ratio cannot measure recovery of specific native contacts.
        n_native = int((cdist(ref_lig, ref_poi) < cutoff).sum())
        if n_native == 0:
            return None
        return {
            'contact_density_ratio': float(n_pred / n_native),
            'n_native_contacts': n_native,
            'n_recovered_contacts': None,
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


def compute_poi_com_dist(
    pred_poi: np.ndarray,
    native_complex_pdb: str,
    featurized_e3_ca: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    """
    Frame-aware POI placement (E3-POI COM distance) vs the native complex.

    ``featurize_common`` centers the E3 at its COM (so the featurized E3 frame
    origin ≈ E3 COM), and the sampler emits POI relative to that same COM.
    When current E3 coordinates are supplied, native E3 is rigid-fitted into
    that frame before measuring POI placement. The legacy two-argument call
    assumes an E3-centered frame with the native orientation.

    Note: this intentionally does NOT report a Kabsch-aligned POI RMSD. The POI
    template is the native POI as a rigid body, so a Kabsch superposition always
    recovers ~0 Å regardless of where the POI was actually placed — that number
    carries no placement signal. The E3-POI COM distance (and ``compute_poi_dockq``)
    are the meaningful POI metrics.

    Returns:
        {'poi_com_dist': Å (E3-POI COM placement error),
         'native_e3_poi_com_dist': Å (reference E3-POI separation)},
        or None when the native PDB is unavailable/unparseable or the POI atom
        count does not match.
    """
    native = _parse_native_complex(native_complex_pdb)
    if native is None or native['e3_ca'].shape[0] == 0:
        return None
    native_poi_e3 = native['poi_ca'] - native['e3_ca'].mean(axis=0)  # [N_poi, 3]
    if featurized_e3_ca is not None:
        if featurized_e3_ca.shape != native['e3_ca'].shape:
            return None
        from tricoopdiff.metrics.dockq import _fit_rigid

        rotation, translation = _fit_rigid(native['e3_ca'], featurized_e3_ca)
        native_poi_e3 = native['poi_ca'] @ rotation + translation
    if pred_poi.shape[0] != native_poi_e3.shape[0]:
        return None
    com_dist = float(np.linalg.norm(pred_poi.mean(axis=0) - native_poi_e3.mean(axis=0)))
    native_com_dist = float(np.linalg.norm(
        native['poi_ca'].mean(axis=0) - native['e3_ca'].mean(axis=0)
    ))
    return {
        'poi_com_dist': com_dist,
        'native_e3_poi_com_dist': native_com_dist,
    }


# ---------------------------------------------------------------------------
# E3-PROTAC interface metrics (fixed-receptor frame)
# ---------------------------------------------------------------------------

def compute_e3_interface_dockq(
    e3_ca: np.ndarray,
    pred_ligand: np.ndarray,
    native_ligand: np.ndarray,
    cutoff: float = 8.0,
) -> Optional[Dict[str, float]]:
    """
    E3-PROTAC interface quality (DockQ composite) in the fixed-receptor frame.

    The sampler emits the PROTAC in the E3 frame with E3 fixed at its featurized
    coordinates (``e3_data['coords']``), and the native PROTAC reference
    (``ligand_data['coords']``) is featurized in the *same* E3-L subgraph — so no
    superposition is needed. This is the E3-side analogue of the pure-DiffDock
    baseline's ``dockq`` / ``fnat`` / ``Lrms``
    (DeepTernary ``DockQ/dockq_util.py::cal_dockq``).

    Definitions (standard DockQ, d1=1.5 Å iRMS, d2=8.5 Å Lrms):
      - Fnat  = fraction of native E3-PROTAC contacts (E3 Cα ↔ PROTAC atom within
                ``cutoff``) recovered by the prediction.
      - Lrms  = ligand RMSD in the fixed-receptor frame (no alignment).
      - iRMS  = RMSD over the PROTAC atoms that make a native contact.
      - DockQ = (Fnat + 1/(1+(iRMS/1.5)^2) + 1/(1+(Lrms/8.5)^2)) / 3

    The receptor is featurized at Cα level (``all_atoms=False``), so ``cutoff``
    is set to 8.0 Å — the Cα→ligand gap is larger than the all-atom→ligand gap
    the baseline uses at 5.0 Å (at 5.0 Å this Cα-only metric has only ~12 native
    contacts for a typical case, too sparse to be meaningful). This matches
    ``compute_poi_dockq``'s PROTAC-POI Cα cutoff for consistency. Values are
    therefore comparable to the baseline in spirit but not identical numerically
    (the baseline's ``cal_dockq`` uses full-atom protein). Returns None when
    there are no native contacts to recover (or atom counts mismatch).
    """
    from scipy.spatial import KDTree

    if pred_ligand.shape != native_ligand.shape:
        return None

    tree = KDTree(e3_ca)
    native_pairs = set()
    for i, p in enumerate(native_ligand):
        for j in tree.query_ball_point(p, cutoff):
            native_pairs.add((i, j))
    n_native = len(native_pairs)
    if n_native == 0:
        return None

    pred_pairs = set()
    for i, p in enumerate(pred_ligand):
        for j in tree.query_ball_point(p, cutoff):
            pred_pairs.add((i, j))
    recovered = len(pred_pairs & native_pairs)
    fnat = recovered / n_native

    diff = pred_ligand - native_ligand
    lrms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    iface_idx = np.asarray(sorted({i for i, _ in native_pairs}), dtype=int)
    if iface_idx.size:
        idiff = pred_ligand[iface_idx] - native_ligand[iface_idx]
        irms = float(np.sqrt(np.mean(np.sum(idiff ** 2, axis=1))))
    else:
        irms = lrms

    dockq = (
        fnat
        + 1.0 / (1.0 + (irms / 1.5) ** 2)
        + 1.0 / (1.0 + (lrms / 8.5) ** 2)
    ) / 3.0

    return {
        'dockq': float(dockq),
        'fnat': float(fnat),
        'lrms': lrms,
        'irms': irms,
        'n_native_contacts': n_native,
        'n_recovered_contacts': recovered,
        'n_pred_contacts': len(pred_pairs),
    }


# ---------------------------------------------------------------------------
# Ensemble Quality Metrics
# ---------------------------------------------------------------------------

def compute_ensemble_diversity(ensemble_coords: List[np.ndarray]) -> Dict[str, float]:
    """Pose diversity in the fixed E3 frame, not conformer-aligned diversity."""
    from tricoopdiff.metrics.ensemble import analyze_ensemble

    report = analyze_ensemble(ensemble_coords, rmsd_cutoff=2.)
    return {
        **{key: report[key] for key in ('mean_pairwise_rmsd', 'std_pairwise_rmsd', 'n_clusters')},
        'coordinate_frame': 'fixed_e3_v1',
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
    debug_trace: bool = False,
    debug_seed: int = 0,
    freeze_poi: bool = False,
    init_mode: str = "sdpi",
    t_anchor: float = None,
    deepternary_dockq_paths: Optional[Dict[str, str]] = None,
    seed: int = 0,
    translation_velocity_scale: float = 1.0,
    rotation_velocity_scale: float = 1.0,
    return_rank_features: bool = False,
    conditions=None,
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
        diffusion: SDPIFlow.
        initializer: TernaryInitializer (default: fresh one).
        device: 'cuda' or 'cpu'.
        n_walkers: number of independent trajectories.
        steps: reverse-diffusion steps (default: diffusion.inference_steps).
        native_complex_pdb: optional path to the native ternary complex PDB
            (e.g. ``DeepTernary/output/protac22/<case>/gt_complex.pdb``). When
            None, POI metrics are skipped.
        debug_trace: when True, also run a single-walker trajectory-level trace
            for parity debugging (seeded by ``debug_seed``).
        debug_seed: RNG seed for the single debug trace.

    Returns:
        {'structure': {...}, 'ensemble': {...}, 'functional': {}, 'debug_trace': ...}
    """
    from tricoopdiff.initialization import TernaryInitializer
    from tricoopdiff.sampler import sample_ensemble, sample_single

    if initializer is None:
        initializer = TernaryInitializer(no_torsion=bool(getattr(model.expert.model, 'no_torsion', False)))

    ensemble = sample_ensemble(
        model, diffusion, initializer, e3_data, poi_data, ligand_data,
        steps=steps, device=device, n_walkers=n_walkers,
        seeds=range(seed, seed + n_walkers),
        freeze_poi=freeze_poi,
        init_mode=init_mode,
        t_anchor=t_anchor,
        translation_velocity_scale=translation_velocity_scale,
        rotation_velocity_scale=rotation_velocity_scale,
        return_rank_features=return_rank_features,
        conditions=conditions,
    )

    ref_lig = ligand_data['coords'][0].detach().cpu().numpy()  # [N_lig, 3] native
    e3_ca = e3_data['coords'][0].detach().cpu().numpy()        # [N_e3, 3] fixed E3 Cα
    ref_poi = poi_data['coords'][0].detach().cpu().numpy()     # [N_poi, 3] native POI Cα
    lig_bonds = ligand_data['edge_index'].detach().cpu().numpy()  # [2, E] covalent graph

    # Per-walker records (one per finite trajectory, aligned indices) so the
    # "best" columns can report a single coherent pose instead of each metric's
    # independent extremum (see the best-sample selection below).
    walkers = []
    e3_native_contacts = None
    poi_native_contacts = None
    poi_native_com_dist = None
    n_nan = 0
    invalid_seeds = []
    for walker_index, r in enumerate(ensemble):
        pred = r['ligand_coords']  # [N_lig, 3] numpy
        pred_poi = r['poi_coords']
        if (
            not r.get('valid', True)
            or not np.isfinite(pred).all()
            or not np.isfinite(pred_poi).all()
        ):
            n_nan += 1
            invalid_seeds.append(seed + walker_index)
            continue
        rec = {
            'seed': seed + walker_index,
            'numerically_valid': r.get('numerically_valid', True),
            'physical_valid': r.get('physical_valid'),
            'physical_check_scope': r.get('physical_check_scope', 'not_checked'),
            'ligand_internal_clash_pairs': r.get('ligand_internal_clash_pairs'),
            'pred': pred,
            'pred_poi': pred_poi,
            'protac_rmsd': compute_rmsd(pred, ref_lig, align_first=True),
            'pocket_dist': float(np.linalg.norm(pred.mean(axis=0) - ref_lig.mean(axis=0))),
        }
        if return_rank_features:
            features = r.get('rank_features')
            if features is None or np.asarray(features).ndim != 1:
                raise ValueError('walker is missing one-dimensional rank_features')
            rec['rank_features'] = np.asarray(features).tolist()
        deepternary_m = compute_deepternary_dockq(
            ref_lig, e3_ca, ref_poi,
            pred, e3_ca, pred_poi,
        )
        if deepternary_m is not None:
            rec['approx_dockq'] = deepternary_m['dockq']
        if deepternary_dockq_paths is not None and native_complex_pdb is not None:
            try:
                from tricoopdiff.metrics.dockq import compute_fullatom_dockq
                exact_m = compute_fullatom_dockq(
                    pred_ligand=pred,
                    pred_poi_ca=pred_poi,
                    featurized_e3_ca=e3_ca,
                    native_complex_pdb=native_complex_pdb,
                    **deepternary_dockq_paths,
                )
                if not all(np.isfinite(exact_m[key]) for key in ('dockq', 'fnat', 'lrms', 'irms')):
                    raise FloatingPointError('non-finite full-atom DockQ metrics')
                rec['dockq'] = exact_m['dockq']
                rec['dockq_fnat'] = exact_m['fnat']
                rec['dockq_lrms'] = exact_m['lrms']
                rec['dockq_irms'] = exact_m['irms']
            except Exception as error:
                rec['exact_dockq_error'] = f'{type(error).__name__}: {error}'
                print(f"[evaluate] DeepTernary full-atom DockQ failed: {error}")
        e3_m = compute_e3_interface_dockq(e3_ca, pred, ref_lig)
        if e3_m is not None:
            rec['e3_dockq'] = e3_m['dockq']
            rec['e3_fnat'] = e3_m['fnat']
            rec['e3_lrms'] = e3_m['lrms']
            if e3_native_contacts is None:
                e3_native_contacts = e3_m['n_native_contacts']
        if native_complex_pdb is not None:
            poi_m = compute_poi_dockq(pred_poi, pred, native_complex_pdb)
            if poi_m is not None:
                rec['poi_atom_correspondence'] = poi_m['atom_correspondence']
                if poi_m['atom_correspondence']:
                    rec['poi_dockq'] = poi_m['dockq']
                else:
                    rec['poi_contact_density_ratio'] = poi_m['contact_density_ratio']
                if poi_native_contacts is None:
                    poi_native_contacts = poi_m['n_native_contacts']
            poi_c = compute_poi_com_dist(pred_poi, native_complex_pdb, e3_ca)
            if poi_c is not None:
                rec['poi_com_dist'] = poi_c['poi_com_dist']
                poi_native_com_dist = poi_c['native_e3_poi_com_dist']
        walkers.append(rec)
    if n_nan:
        print(
            f"[evaluate] skipped {n_nan}/{len(ensemble)} walkers with non-finite "
            "scores or coordinates"
        )

    pred_ligs = [w['pred'] for w in walkers]
    protac_rmsds = np.asarray([w['protac_rmsd'] for w in walkers])
    pocket_dists = np.asarray([w['pocket_dist'] for w in walkers])
    # Best sample follows DeepTernary's ternary DockQ convention. Every "_best"
    # field below refers to this same coherent walker.
    exact_indices = [index for index, walker in enumerate(walkers) if 'dockq' in walker]
    if exact_indices:
        best_idx = max(exact_indices, key=lambda index: walkers[index]['dockq'])
        selection_metric = 'dockq'
    elif walkers:
        best_idx = int(np.argmax([w.get('approx_dockq', -1.0) for w in walkers]))
        selection_metric = 'approx_dockq'
    else:
        best_idx = 0
        selection_metric = None

    # Initialize every metric to None; the finite-walker branch below fills in
    # what it can. "best" always means "the single dockq-best walker's value"
    # (see best_idx above) so the best column is one coherent pose, while
    # mean/median are ensemble statistics over all finite walkers.
    structure = {
        'dockq_best': None, 'dockq_mean': None, 'dockq_median': None,
        'dockq_fnat_best': None, 'dockq_fnat_mean': None,
        'dockq_lrms_best': None, 'dockq_lrms_mean': None,
        'dockq_irms_best': None, 'dockq_irms_mean': None,
        'protac_rmsd_best': None, 'protac_rmsd_mean': None, 'protac_rmsd_median': None,
        'protac_pocket_dist_mean': None, 'protac_pocket_dist_best': None,
        'clash_rate': None,
        'poi_dockq_best': None, 'poi_dockq_mean': None, 'poi_dockq_median': None,
        'poi_interface_native_contacts': None,
        'poi_com_dist_best': None, 'poi_com_dist_mean': None,
        'native_e3_poi_com_dist': None,
        'e3_interface_dockq_best': None, 'e3_interface_dockq_mean': None,
        'e3_interface_dockq_median': None,
        'e3_interface_fnat_best': None, 'e3_interface_fnat_mean': None,
        'e3_interface_lrms_best': None, 'e3_interface_lrms_mean': None,
        'e3_interface_native_contacts': None,
    }

    if walkers:
        best = walkers[best_idx]
        dockqs = np.asarray([w['dockq'] for w in walkers if 'dockq' in w])
        dockq_fnats = np.asarray([w['dockq_fnat'] for w in walkers if 'dockq_fnat' in w])
        dockq_lrms = np.asarray([w['dockq_lrms'] for w in walkers if 'dockq_lrms' in w])
        dockq_irms = np.asarray([w['dockq_irms'] for w in walkers if 'dockq_irms' in w])
        structure['dockq_best'] = float(best['dockq']) if 'dockq' in best else None
        structure['dockq_mean'] = float(dockqs.mean()) if len(dockqs) else None
        structure['dockq_median'] = float(np.median(dockqs)) if len(dockqs) else None
        structure['dockq_fnat_best'] = float(best['dockq_fnat']) if 'dockq_fnat' in best else None
        structure['dockq_fnat_mean'] = float(dockq_fnats.mean()) if len(dockq_fnats) else None
        structure['dockq_lrms_best'] = float(best['dockq_lrms']) if 'dockq_lrms' in best else None
        structure['dockq_lrms_mean'] = float(dockq_lrms.mean()) if len(dockq_lrms) else None
        structure['dockq_irms_best'] = float(best['dockq_irms']) if 'dockq_irms' in best else None
        structure['dockq_irms_mean'] = float(dockq_irms.mean()) if len(dockq_irms) else None
        structure['protac_rmsd_best'] = float(best['protac_rmsd'])
        structure['protac_rmsd_mean'] = float(protac_rmsds.mean())
        structure['protac_rmsd_median'] = float(np.median(protac_rmsds))
        structure['protac_pocket_dist_mean'] = float(pocket_dists.mean())
        structure['protac_pocket_dist_best'] = float(best['pocket_dist'])

    # Steric clash over the PROTAC ensemble (atomic numbers unknown at MVP scale;
    # use carbon=6 as a proxy — same convention as the reflection post-processor).
    # The ligand bond graph is passed so 1-2/1-3/1-4 covalent geometry is excluded.
    try:
        structure['clash_rate'] = compute_clash_rate(
            pred_ligs, np.full(ref_lig.shape[0], 6), bonds=lig_bonds
        )
    except Exception:
        structure['clash_rate'] = None

    if walkers:
        best = walkers[best_idx]

        # PROTAC-POI interface recovery (DockQ-style Fnat), frame-invariant.
        poi_dockqs = np.asarray([w['poi_dockq'] for w in walkers if 'poi_dockq' in w])
        structure['poi_dockq_best'] = float(best['poi_dockq']) if 'poi_dockq' in best else None
        structure['poi_dockq_mean'] = float(poi_dockqs.mean()) if len(poi_dockqs) else None
        structure['poi_dockq_median'] = float(np.median(poi_dockqs)) if len(poi_dockqs) else None
        structure['poi_interface_native_contacts'] = poi_native_contacts

        # POI placement (frame-aware): E3-POI COM distance (no Kabsch RMSD — the
        # POI template is the native rigid body, so a Kabsch superposition would
        # trivially give ~0 and carry no placement signal).
        poi_com_dists = np.asarray([w['poi_com_dist'] for w in walkers if 'poi_com_dist' in w])
        structure['poi_com_dist_best'] = float(best['poi_com_dist']) if 'poi_com_dist' in best else None
        structure['poi_com_dist_mean'] = float(poi_com_dists.mean()) if len(poi_com_dists) else None
        structure['native_e3_poi_com_dist'] = poi_native_com_dist

        # E3-PROTAC interface quality (fixed-receptor frame), always available
        # from the featurized native (no native PDB needed).
        e3_dockqs = np.asarray([w['e3_dockq'] for w in walkers if 'e3_dockq' in w])
        e3_fnats = np.asarray([w['e3_fnat'] for w in walkers if 'e3_fnat' in w])
        e3_lrms = np.asarray([w['e3_lrms'] for w in walkers if 'e3_lrms' in w])
        structure['e3_interface_dockq_best'] = float(best['e3_dockq']) if 'e3_dockq' in best else None
        structure['e3_interface_dockq_mean'] = float(e3_dockqs.mean()) if len(e3_dockqs) else None
        structure['e3_interface_dockq_median'] = float(np.median(e3_dockqs)) if len(e3_dockqs) else None
        structure['e3_interface_fnat_best'] = float(best['e3_fnat']) if 'e3_fnat' in best else None
        structure['e3_interface_fnat_mean'] = float(e3_fnats.mean()) if len(e3_fnats) else None
        structure['e3_interface_lrms_best'] = float(best['e3_lrms']) if 'e3_lrms' in best else None
        structure['e3_interface_lrms_mean'] = float(e3_lrms.mean()) if len(e3_lrms) else None
        structure['e3_interface_native_contacts'] = e3_native_contacts

    ensemble_metrics = {
        'n_walkers': len(ensemble),
        'n_valid_walkers': len(walkers),
        'n_nan_walkers': n_nan,
        'n_exact_dockq_walkers': len(exact_indices),
        'invalid_seeds': invalid_seeds,
        'selection_metric': selection_metric,
        'protac_rmsd_std': float(protac_rmsds.std()) if len(protac_rmsds) > 1 else 0.0,
    }
    if len(pred_ligs) > 1:
        ensemble_metrics['diversity'] = compute_ensemble_diversity(pred_ligs)
    if walkers:
        from tricoopdiff.metrics.ensemble import analyze_ensemble
        ternary_poses = [
            np.concatenate([walker['pred'], walker['pred_poi']], axis=0)
            for walker in walkers
        ]
        ensemble_metrics['conformational_analysis'] = analyze_ensemble(
            ternary_poses, rmsd_cutoff=2.0,
        )

    debug_trace_report = None
    if debug_trace:
        debug_trace_report = sample_single(
            model, diffusion, initializer, e3_data, poi_data, ligand_data,
            steps=steps, device=device, seed=debug_seed, return_trace=True,
            freeze_poi=freeze_poi,
            init_mode=init_mode,
            t_anchor=t_anchor,
            translation_velocity_scale=translation_velocity_scale,
            rotation_velocity_scale=rotation_velocity_scale,
            conditions=conditions,
        )

    report = {
        'walkers': [
            {key: value for key, value in walker.items() if key not in ('pred', 'pred_poi')}
            for walker in walkers
        ],
        'initialization': [r['initialization'] for r in ensemble],
        'structure': structure,
        'ensemble': ensemble_metrics,
        # Functional ranking (α, Top-20% hit rate, AUROC) requires predicted
        # productivity per complex — wired in a later phase.
        'functional': {},
    }
    if debug_trace_report is not None:
        report['debug_trace'] = {
            'seed': debug_seed,
            'trace': debug_trace_report.get('trace', []),
            'final_ligand_coords': debug_trace_report['ligand_coords'].tolist(),
            'final_poi_coords': debug_trace_report['poi_coords'].tolist(),
        }
    return report


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
    # Rotation matrix (handle reflection). For covariance H = mobile_c.T @
    # target_c the optimal rotation is R = U @ diag([1,1,d]) @ Vt. The previous
    # form V @ D @ U.T is that rotation's *transpose*, which silently inflated
    # every Kabsch-aligned RMSD (verified: a known rigid transform gave ~33 Å
    # instead of ~0).
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt

    # Apply rotation
    aligned = mobile_c @ R + target_center
    return aligned
