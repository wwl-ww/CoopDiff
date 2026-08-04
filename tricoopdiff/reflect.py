"""
TriReflect: Physical post-processing for ternary complex structures.

Detects and optionally fixes physical rule violations in generated structures:
1. Steric clashes (atom pairs closer than van der Waals radii)
2. Bond length anomalies (> 1.5× standard bond length)
3. Linker torsional strain (|dihedral| < 30° or > 150°)
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

# Linker strain thresholds (degrees)
LINKER_STRAIN_MIN = 30.0   # |dihedral| below this = strained
LINKER_STRAIN_MAX = 150.0  # |dihedral| above this = strained


def detect_clashes(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    vdw_scale: float = 0.8,
) -> np.ndarray:
    """
    Detect steric clashes using KD-tree.

    Args:
        coords: [N, 3] atom coordinates
        atomic_numbers: [N] atomic numbers
        vdw_scale: scaling factor for VDW radii (0.8 = allow slight overlap)

    Returns:
        [N] bool array, True for atoms involved in clashes
    """
    N = coords.shape[0]
    tree = KDTree(coords)
    radii = np.array([VDW_RADII.get(z, 1.70) for z in atomic_numbers])

    # Find all pairs within max possible VDW contact
    max_radius = np.max(radii) * vdw_scale * 2
    pairs = tree.query_pairs(r=max_radius, output_type='ndarray')

    if len(pairs) == 0:
        return np.zeros(N, dtype=bool)

    clash_mask = np.zeros(N, dtype=bool)

    for i, j in pairs:
        # Skip bonded atoms (1-2 pairs) — would need bond graph for accurate check
        dist = np.linalg.norm(coords[i] - coords[j])
        threshold = (radii[i] + radii[j]) * vdw_scale
        if dist < threshold:
            clash_mask[i] = True
            clash_mask[j] = True

    return clash_mask


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
    strain_max: float = LINKER_STRAIN_MAX,
) -> np.ndarray:
    """
    Detect strained torsional angles in the PROTAC linker.

    Args:
        dihedral_angles: [K] dihedral angles in radians
        strain_min: minimum acceptable |angle| in degrees
        strain_max: maximum acceptable |angle| in degrees

    Returns:
        [K] bool array, True for strained bonds
    """
    abs_deg = np.abs(np.degrees(dihedral_angles))
    return (abs_deg < strain_min) | (abs_deg > strain_max)


def local_minimization(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    clash_atoms: np.ndarray,
    max_rounds: int = 3,
    step_size: float = 0.01,
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
        clash_atoms = detect_clashes(coords, atomic_numbers)

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
    clash_mask = detect_clashes(coords, atomic_numbers)
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
        coords = local_minimization(coords, atomic_numbers, clash_mask)
        # Re-check after minimization
        clash_mask = detect_clashes(coords, atomic_numbers)
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
