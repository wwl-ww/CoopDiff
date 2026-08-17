"""
TriReflect: Physical post-processing for ternary complex structures.

Detects and optionally fixes physical rule violations in generated structures:
1. Steric clashes (atom pairs closer than van der Waals radii)
2. Bond length anomalies (> 1.5× standard bond length)
3. Linker torsional strain (near-eclipsed dihedrals, within 30° of 0° or ±120°)
4. Local MMFF94 energy minimization (max 3 rounds)

The "reflect" metaphor: the diffusion model generates, TriReflect reflects
back physically impossible structures for correction.
"""

import numpy as np
import torch
from scipy.spatial import KDTree
from typing import Optional, Tuple, List


# Standard van der Waals radii (Å) for common elements
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}

# Covalent radii (Å) for common elements. Used to skip covalently bonded (1-2)
# atom pairs in clash detection — a bonded pair sits near the sum of covalent
# radii, well inside any VDW clash threshold, and is not a steric clash.
COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
}

# Tolerance factor for "is this pair covalently bonded". A bonded pair has a
# distance ≈ sum of covalent radii (within ~20%), so anything below 1.2× that
# sum is treated as bonded and excluded from clash detection.
BOND_DISTANCE_TOLERANCE = 1.2

# Standard bond lengths (Å) for common bond types
STANDARD_BOND_LENGTHS = {
    ('C', 'C', 1): 1.54,  # C-C single
    ('C', 'C', 2): 1.34,  # C=C double
    ('C', 'C', 3): 1.20,  # C≡C triple
    ('C', 'N', 1): 1.47,  # C-N single
    ('C', 'N', 2): 1.28,  # C=N double
    ('C', 'O', 1): 1.43,  # C-O single
    ('C', 'O', 2): 1.21,  # C=O double
    ('C', 'S', 1): 1.82,  # C-S single
    ('N', 'N', 1): 1.45,  # N-N single
    ('N', 'O', 1): 1.40,  # N-O single
    ('O', 'O', 1): 1.48,  # O-O single
}

# Default bond length tolerance factor
BOND_LENGTH_TOLERANCE = 1.5

# Linker strain threshold (degrees): a dihedral within this angular distance of
# an *eclipsed* conformation (0° or ±120°) is flagged as strained. Staggered
# conformations (gauche ±60°, anti 180°) are the low-energy minima, not strain.
LINKER_STRAIN_MIN = 30.0


def detect_clashes(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    vdw_scale: float = 0.8,
    bonds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect steric clashes using KD-tree.

    Args:
        coords: [N, 3] atom coordinates
        atomic_numbers: [N] atomic numbers
        vdw_scale: scaling factor for VDW radii (0.8 = allow slight overlap)
        bonds: optional bond graph — either a [2, E] int array of atom indices
            or an iterable of (i, j) pairs. When given, atom pairs within three
            bonds (1-2 / 1-3 / 1-4) are excluded, since those separations are
            normal covalent geometry (ring geminals, vicinals) not clashes.
            When None, only 1-2 pairs are excluded via a covalent-radius
            heuristic (less accurate, e.g. ring 1-3 atoms still flagged).

    Returns:
        [N] bool array, True for atoms involved in clashes
    """
    N = coords.shape[0]
    tree = KDTree(coords)
    radii = np.array([VDW_RADII.get(z, 1.70) for z in atomic_numbers])
    cov_radii = np.array([COVALENT_RADII.get(z, 0.76) for z in atomic_numbers])

    # Find all pairs within max possible VDW contact
    max_radius = np.max(radii) * vdw_scale * 2
    pairs = tree.query_pairs(r=max_radius, output_type='ndarray')

    if len(pairs) == 0:
        return np.zeros(N, dtype=bool)

    # Bond separation matrix [N, N] (shortest path ≤ 3, else -1), or None to
    # fall back to the covalent-radius 1-2 heuristic below.
    skip_sep = _bond_separation_lookup(N, bonds) if bonds is not None else None

    clash_mask = np.zeros(N, dtype=bool)

    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if skip_sep is not None:
            if skip_sep[i, j] != -1 and skip_sep[i, j] <= 3:
                continue  # 1-2 / 1-3 / 1-4: normal covalent geometry
        elif dist < (cov_radii[i] + cov_radii[j]) * BOND_DISTANCE_TOLERANCE:
            continue  # no bond graph: covalent-radius heuristic for 1-2 only
        threshold = (radii[i] + radii[j]) * vdw_scale
        if dist < threshold:
            clash_mask[i] = True
            clash_mask[j] = True

    return clash_mask


def _bond_separation_lookup(n_atoms: int, bonds) -> np.ndarray:
    """
    Shortest-path bond separation for all pairs, truncated at depth 3.

    Returns an [N, N] int8 array where entry [i, j] is the number of bonds on
    the shortest path between i and j (0 = self, 1 = bonded, 2 = geminal, 3 =
    vicinal), or -1 if the separation exceeds 3 (treated as "not bonded nearby").
    """
    adj = [set() for _ in range(n_atoms)]
    if getattr(bonds, "ndim", 1) == 2:
        # [2, E] index array (e.g. torch_geometric edge_index converted to numpy)
        for i, j in bonds.T:
            i, j = int(i), int(j)
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                adj[i].add(j)
                adj[j].add(i)
    else:
        for i, j in bonds:
            i, j = int(i), int(j)
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                adj[i].add(j)
                adj[j].add(i)

    sep = np.full((n_atoms, n_atoms), -1, dtype=np.int8)
    for src in range(n_atoms):
        sep[src, src] = 0
        frontier = [src]
        for depth in range(1, 4):
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if sep[src, b] == -1:
                        sep[src, b] = depth
                        nxt.append(b)
            frontier = nxt
            if not frontier:
                break
    return sep


def detect_bond_anomalies(
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    bond_types: List[int],
    atomic_numbers: np.ndarray,
    tolerance: float = BOND_LENGTH_TOLERANCE,
) -> np.ndarray:
    """
    Detect anomalous bond lengths.

    Args:
        coords: [N, 3] atom coordinates
        bonds: list of (i, j) index pairs defining bonds
        bond_types: list of bond orders (1=single, 2=double, 3=triple)
        atomic_numbers: [N] atomic numbers
        tolerance: multiplier on standard length to flag as anomaly

    Returns:
        [len(bonds)] bool array, True for anomalous bonds
    """
    anomalies = np.zeros(len(bonds), dtype=bool)

    for idx, ((i, j), order) in enumerate(zip(bonds, bond_types)):
        dist = np.linalg.norm(coords[i] - coords[j])
        elem_i = _atomic_number_to_symbol(atomic_numbers[i])
        elem_j = _atomic_number_to_symbol(atomic_numbers[j])

        # Look up standard length (order-independent fallback)
        std_len = STANDARD_BOND_LENGTHS.get(
            (elem_i, elem_j, order),
            STANDARD_BOND_LENGTHS.get((elem_j, elem_i, order), None),
        )
        if std_len is None:
            # Fallback: use sum of covalent radii
            std_len = 1.5  # approximate

        if dist > std_len * tolerance or dist < std_len / tolerance:
            anomalies[idx] = True

    return anomalies


def detect_linker_strain(
    dihedral_angles: np.ndarray,
    strain_min: float = LINKER_STRAIN_MIN,
) -> np.ndarray:
    """
    Detect strained torsional angles in the PROTAC linker.

    A dihedral is flagged as strained only when it is near an *eclipsed*
    conformation (0° or ±120°), which is the torsional barrier. Staggered
    conformations — gauche (±60°) and anti (180°) — are the low-energy minima
    and are NOT strained; anti in particular was previously mis-flagged.

    Args:
        dihedral_angles: [K] dihedral angles in radians
        strain_min: angular distance (degrees) from the nearest eclipsed
            conformation below which a dihedral is flagged as strained

    Returns:
        [K] bool array, True for strained bonds
    """
    abs_deg = np.abs(np.degrees(dihedral_angles))
    # Dihedral is in (-π, π] so |φ| ∈ [0°, 180°]. Eclipsed conformations sit at
    # 0° and 120° (240° ≡ -120° ≡ |120°|). Distance to the nearest one:
    dist_to_eclipsed = np.minimum(abs_deg, np.abs(abs_deg - 120.0))
    return dist_to_eclipsed < strain_min


def local_minimization(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    clash_atoms: np.ndarray,
    max_rounds: int = 3,
    step_size: float = 0.01,
    bonds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simple local energy minimization to resolve clashes.
    Performs gradient descent on a repulsive VDW potential.

    This is a lightweight alternative to full MMFF94 minimization.
    For production use, integrate with RDKit's MMFF94 implementation.

    Args:
        coords: [N, 3] atom coordinates
        atomic_numbers: [N] atomic numbers
        clash_atoms: [N] bool mask of clashing atoms
        max_rounds: maximum minimization rounds
        step_size: gradient step size
        bonds: optional bond graph forwarded to `detect_clashes` so 1-2/1-3/1-4
            pairs are excluded (see `detect_clashes` for accepted formats)

    Returns:
        [N, 3] adjusted coordinates
    """
    coords = coords.copy()

    for _ in range(max_rounds):
        if not clash_atoms.any():
            break

        clash_idx = np.where(clash_atoms)[0]
        radii = np.array([VDW_RADII.get(z, 1.70) for z in atomic_numbers])

        for idx in clash_idx:
            # Compute repulsive forces from all other atoms
            delta = coords[idx] - coords  # [N, 3]
            dist = np.linalg.norm(delta, axis=1)  # [N]
            dist = np.maximum(dist, 1e-3)  # avoid division by zero

            vdw_sum = radii[idx] + radii  # [N]
            overlap = vdw_sum - dist  # [N]
            overlap = np.maximum(overlap, 0)  # only repulsive

            # Repulsive force: F = -∇(overlap²) = 2 * overlap * direction
            forces = 2 * overlap[:, np.newaxis] * (delta / dist[:, np.newaxis])
            force = forces.sum(axis=0)

            # Gradient step
            coords[idx] += step_size * force

        # Re-check clashes
        clash_atoms = detect_clashes(coords, atomic_numbers, bonds=bonds)

    return coords


def tricoop_reflect(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_types: Optional[List[int]] = None,
    dihedral_angles: Optional[np.ndarray] = None,
    do_minimization: bool = True,
) -> dict:
    """
    Full TriReflect pipeline: detect + report all physical violations,
    optionally run local minimization to fix clashes.

    Args:
        coords: [N, 3] atom coordinates (all three molecules concatenated)
        atomic_numbers: [N] atomic numbers
        bonds: optional list of bond pairs
        bond_types: optional list of bond orders
        dihedral_angles: optional [K] PROTAC linker dihedral angles
        do_minimization: whether to run local minimization

    Returns:
        dict:
            'coords': [N, 3] (possibly adjusted) coordinates
            'clashes': [N] bool clash mask
            'n_clashes': int number of clashing atoms
            'bond_anomalies': [E] bool or None
            'n_bond_anomalies': int
            'linker_strain': [K] bool or None
            'n_strained_bonds': int
            'is_valid': bool True if no hard violations remain
    """
    # 1. Clash detection
    clash_mask = detect_clashes(coords, atomic_numbers, bonds=bonds)
    n_clashes = clash_mask.sum()

    # 2. Bond anomaly detection
    bond_anomalies = None
    n_bond_anomalies = 0
    if bonds is not None and bond_types is not None:
        bond_anomalies = detect_bond_anomalies(coords, bonds, bond_types, atomic_numbers)
        n_bond_anomalies = bond_anomalies.sum()

    # 3. Linker strain detection
    linker_strain = None
    n_strained = 0
    if dihedral_angles is not None:
        linker_strain = detect_linker_strain(dihedral_angles)
        n_strained = linker_strain.sum()

    # 4. (Optional) Local minimization
    if do_minimization and n_clashes > 0:
        coords = local_minimization(coords, atomic_numbers, clash_mask, bonds=bonds)
        # Re-check after minimization
        clash_mask = detect_clashes(coords, atomic_numbers, bonds=bonds)
        n_clashes = clash_mask.sum()

    # 5. Validity flag: no hard violations
    is_valid = (n_clashes == 0 and n_bond_anomalies == 0)

    return {
        'coords': coords,
        'clashes': clash_mask,
        'n_clashes': n_clashes,
        'bond_anomalies': bond_anomalies,
        'n_bond_anomalies': n_bond_anomalies,
        'linker_strain': linker_strain,
        'n_strained_bonds': n_strained,
        'is_valid': is_valid,
    }


def _atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    mapping = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
    }
    return mapping.get(z, 'C')
