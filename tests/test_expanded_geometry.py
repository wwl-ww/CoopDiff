"""Heavy-atom ranking features remain invariant and expose clashes."""

import numpy as np

from tricoopdiff.expanded_geometry import (EXPANDED_GEOMETRY_NAMES,
                                           apply_fitted_rigid,
                                           expanded_geometry_features)


def rotation():
    q, _ = np.linalg.qr(np.asarray([[.3, .4, .5], [.2, -.8, .1], [.7, .2, -.6]]))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_expanded_features_are_rigid_invariant_and_fixed_width():
    rng = np.random.RandomState(7)
    e3 = rng.normal(size=(17, 3))
    poi = rng.normal(size=(13, 3)) + np.asarray([8., 0., 0.])
    ligand = rng.normal(size=(9, 3)) + np.asarray([4., 0., 0.])
    original = expanded_geometry_features(e3, poi, ligand)
    rigid = lambda value: value @ rotation().T + np.asarray([3., -5., 2.])
    moved = expanded_geometry_features(rigid(e3), rigid(poi), rigid(ligand))
    assert original.shape == (len(EXPANDED_GEOMETRY_NAMES),) == (37,)
    assert np.allclose(original, moved, atol=1e-6, rtol=1e-6)


def test_expanded_features_report_ligand_and_interface_clashes():
    e3 = np.asarray([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    poi = e3 + np.asarray([.5, 0., 0.])
    ligand = np.asarray([[.1, .1, 0.], [20., 20., 20.]])
    values = dict(zip(EXPANDED_GEOMETRY_NAMES,
                      expanded_geometry_features(e3, poi, ligand)))
    assert values['e3_ligand_frac_lt2'] == .5
    assert values['poi_ligand_frac_lt2'] == .5
    assert values['e3_to_poi_frac_lt2'] == 1.
    assert values['poi_to_e3_frac_lt2'] == 1.


def test_fitted_rigid_moves_heavy_atoms_with_ca_frame():
    source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    heavy = np.concatenate((source, np.asarray([[.3, .4, .5]])))
    transform = lambda value: value @ rotation() + np.asarray([2., -4., 7.])
    moved, rmsd = apply_fitted_rigid(source, transform(source), heavy)
    assert rmsd < 1e-10
    assert np.allclose(moved, transform(heavy), atol=1e-10, rtol=0)


if __name__ == '__main__':
    test_expanded_features_are_rigid_invariant_and_fixed_width()
    test_expanded_features_report_ligand_and_interface_clashes()
    test_fitted_rigid_moves_heavy_atoms_with_ca_frame()
