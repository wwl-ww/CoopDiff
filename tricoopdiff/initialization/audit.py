"""Analytic diagnostics, not a certificate of train/inference equivalence.

All coordinates use the indexed heavy-atom order. Conditional KL values must
not be interpreted as the KL of the data-marginal terminal distribution.
"""

import math

import numpy as np


def conditional_translation_kl(native_center, receptor_center, train_sigma,
                               prior_sigma):
    """KL(N(native_center, s²I) || N(receptor_center, r²I)), in nats."""
    native = np.asarray(native_center, dtype=np.float64)
    receptor = np.asarray(receptor_center, dtype=np.float64)
    if native.shape != (3,) or receptor.shape != (3,):
        raise ValueError("centers must have shape [3]")
    if not np.isfinite(native).all() or not np.isfinite(receptor).all():
        raise ValueError("centers must be finite")
    if not all(math.isfinite(s) and s > 0 for s in (train_sigma, prior_sigma)):
        raise ValueError("standard deviations must be finite and positive")
    ratio = float(train_sigma / prior_sigma)
    displacement = float(np.sum(((native - receptor) / prior_sigma) ** 2))
    return 0.5 * (3 * (ratio ** 2 - 1 - 2 * math.log(ratio)) + displacement)


def invariant_atom_pairs(mol, torsion_bonds):
    """Distances unchanged by rigid motion and the allowed bond rotations.

    Include pairs within each rigid fragment and every bonded/1--3 pair.
    These are necessary invariants, not a complete characterization (e.g.
    signed chirality is not tested). Validate directed masks before calling.
    """
    n_atoms = mol.GetNumAtoms()
    adjacency = [set() for _ in range(n_atoms)]
    edges = set()
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[u].add(v)
        adjacency[v].add(u)
        edges.add(tuple(sorted((u, v))))
    cuts = {tuple(sorted(map(int, bond))) for bond in torsion_bonds}
    if not cuts <= edges:
        raise ValueError("torsion axes must be chemical bonds")
    pairs = set(edges)
    for neighbors in adjacency:
        neighbors = sorted(neighbors)
        pairs.update((u, v) for i, u in enumerate(neighbors) for v in neighbors[i + 1:])
    remaining = set(range(n_atoms))
    while remaining:
        root = min(remaining)
        component, pending = {root}, [root]
        while pending:
            u = pending.pop()
            for v in adjacency[u]:
                if tuple(sorted((u, v))) not in cuts and v not in component:
                    component.add(v)
                    pending.append(v)
        remaining -= component
        ordered = sorted(component)
        pairs.update((u, v) for i, u in enumerate(ordered) for v in ordered[i + 1:])
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def invariant_geometry_gap(native_coords, initial_coords, pairs, atol=1e-5):
    """Necessary reachability test and a conservative atom RMSD lower bound.

    For every invariant pair, |d_native - d_initial| <= ||e_i|| + ||e_j||
    <= sqrt(2N) * RMSD. This holds after ANY allowed torsions and rigid fit.
    A zero gap does not prove reachability; a positive gap proves exact indexed
    reconstruction impossible under these degrees of freedom.
    """
    native = np.asarray(native_coords, dtype=np.float64)
    initial = np.asarray(initial_coords, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 3 or len(native) == 0:
        raise ValueError("coordinates must have nonempty shape [N, 3]")
    if initial.shape != native.shape:
        raise ValueError("native and initial atom maps must have equal shapes")
    if not np.isfinite(native).all() or not np.isfinite(initial).all():
        raise ValueError("coordinates must be finite")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("pairs must be an integer [K, 2] array")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= len(native)):
        raise ValueError("pair index out of range")
    i, j = pairs.T
    differences = np.abs(np.linalg.norm(native[i] - native[j], axis=-1)
                         - np.linalg.norm(initial[i] - initial[j], axis=-1))
    maximum = float(differences.max()) if len(differences) else 0.0
    return {
        'n_invariant_pairs': len(pairs),
        'n_pairs_above_tolerance': int((differences > atol).sum()),
        'distance_max_abs_error_angstrom': maximum,
        'distance_rms_error_angstrom': float(np.sqrt(np.mean(differences ** 2))) if len(pairs) else 0.0,
        'indexed_atom_rmsd_lower_bound_angstrom': maximum / math.sqrt(2 * len(native)),
        'exact_reachability_ruled_out_at_tolerance': maximum > atol,
        'tolerance_angstrom': atol,
    }
