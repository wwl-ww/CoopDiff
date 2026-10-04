"""Necessary endpoint-consistency diagnostics have independently checked bounds."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.initialization.audit import (
    conditional_translation_kl, invariant_atom_pairs, invariant_geometry_gap,
)


def test_conditional_gaussian_kl():
    assert conditional_translation_kl([0, 0, 0], [0, 0, 0], 2, 2) == 0
    assert math.isclose(conditional_translation_kl([2, 0, 0], [0, 0, 0], 2, 2), 0.5)
    expected = 0.5 * 3 * (4 - 1 - math.log(4))
    assert math.isclose(conditional_translation_kl([0, 0, 0], [0, 0, 0], 4, 2), expected)
    # Numeric expectation of log q - log p checks direction and dimension.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200000, 3)) * 2 + [1, 2, 3]
    empirical = np.mean(-3 * math.log(2 / 3)
                        - np.sum((x - [1, 2, 3]) ** 2, axis=1) / 8
                        + np.sum(x ** 2, axis=1) / 18)
    analytic = conditional_translation_kl([1, 2, 3], [0, 0, 0], 2, 3)
    assert abs(empirical - analytic) < 0.01


def test_invariant_pairs_and_rmsd_bound():
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CCCC')
    pairs = invariant_atom_pairs(mol, [(1, 2)])
    assert set(map(tuple, pairs)) == {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
    native = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [1., 1, 1]])
    moved = native.copy()
    moved[0, 0] -= 0.4
    gap = invariant_geometry_gap(native, moved, pairs)
    rmsd = np.sqrt(np.mean(np.sum((native - moved) ** 2, axis=1)))
    assert gap['exact_reachability_ruled_out_at_tolerance']
    assert 0 < gap['indexed_atom_rmsd_lower_bound_angstrom'] <= rmsd
    unchanged = invariant_geometry_gap(native, native + [10, -3, 2], pairs)
    assert not unchanged['exact_reachability_ruled_out_at_tolerance']
    # With no allowed torsions, all six pair distances are invariant.
    assert len(invariant_atom_pairs(mol, [])) == 6


def test_real_torsion_updates_preserve_audited_invariants():
    import tricoopdiff.diffdock_compat  # noqa: F401
    import torch
    from rdkit import Chem
    from test_initialization import ligand_input
    from tricoopdiff.initialization import validate_ligand_graph
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    data = ligand_input('CCCOCC')
    mol = Chem.Mol(data['mol'])
    bonds, masks = validate_ligand_graph(mol, data['edge_index'], data['edge_mask'], data['mask_rotate'])
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    pairs = invariant_atom_pairs(mol, bonds.tolist())
    for seed in range(5):
        updates = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(bonds))
        moved = modify_conformer_torsion_angles(coords, bonds, masks, updates, as_numpy=False)
        # Vendored as_numpy=False returns float32, even from float64 input.
        gap = invariant_geometry_gap(coords.numpy(), moved.numpy(), pairs, atol=1e-6)
        assert not gap['exact_reachability_ruled_out_at_tolerance'], gap


def test_audit_rejects_nonfinite_and_bad_shapes():
    checks = [
        lambda: conditional_translation_kl([0, 0, 0], [0, 0, 0], 0, 1),
        lambda: conditional_translation_kl([0, 0, 0], [0, 0, 0], 1, float('nan')),
        lambda: conditional_translation_kl([float('nan'), 0, 0], [0, 0, 0], 1, 1),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((3, 3)), np.array([[0, 1]])),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((2, 3)), np.array([[0, 2]])),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((2, 3)), np.array([[0., 1.]])),
    ]
    for check in checks:
        try:
            check()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid audit inputs must fail')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}')
