"""Native-free heavy-atom geometry features for candidate pose ranking."""

import numpy as np
from scipy.spatial import cKDTree


_LIGAND_STATS = ('min', 'q10', 'q25', 'median', 'q75',
                 'frac_lt2', 'frac_lt4', 'frac_lt6')
_PROTEIN_COVERAGE = ('coverage_lt2', 'coverage_lt4', 'coverage_lt6', 'coverage_lt8')
_INTERFACE_STATS = ('min', 'q01', 'q05', 'frac_lt2', 'frac_lt4', 'frac_lt6')

EXPANDED_GEOMETRY_NAMES = tuple(
    f'{side}_{name}'
    for side in ('e3_ligand', 'poi_ligand')
    for name in (*_LIGAND_STATS, *_PROTEIN_COVERAGE)
) + tuple(
    f'{direction}_{name}'
    for direction in ('e3_to_poi', 'poi_to_e3')
    for name in _INTERFACE_STATS
) + ('protein_heavy_centroid_distance',)


def apply_fitted_rigid(source_ca, target_ca, points):
    """Fit source C-alpha coordinates to a candidate and move matching heavy atoms."""
    source = _points(source_ca, 'source_ca')
    target = _points(target_ca, 'target_ca')
    moving = _points(points, 'points')
    if source.shape != target.shape or len(source) < 3:
        raise ValueError('source and target C-alpha arrays must match with N >= 3')
    source_center, target_center = source.mean(0), target.mean(0)
    x, y = source - source_center, target - target_center
    u, singular, vh = np.linalg.svd(x.T @ y)
    if singular[1] <= np.finfo(np.float64).eps * singular[0]:
        raise ValueError('C-alpha rigid fit is underdetermined')
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vh)
    rotation = u @ np.diag(sign) @ vh
    translation = target_center - source_center @ rotation
    fitted = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return moving @ rotation + translation, rmsd


def _points(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError(f'{name} must have shape [N,3] with N > 0')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite coordinates')
    return value


def _fractions(distances, radii):
    return [float(np.mean(distances < radius)) for radius in radii]


def expanded_geometry_features(e3_heavy, poi_heavy, ligand_heavy):
    """Return invariant distance summaries without native or quality information."""
    e3 = _points(e3_heavy, 'e3_heavy')
    poi = _points(poi_heavy, 'poi_heavy')
    ligand = _points(ligand_heavy, 'ligand_heavy')
    e3_tree, poi_tree, ligand_tree = cKDTree(e3), cKDTree(poi), cKDTree(ligand)
    values = []
    for protein, tree in ((e3, e3_tree), (poi, poi_tree)):
        ligand_nearest = tree.query(ligand, k=1, workers=1)[0]
        values.extend(float(np.quantile(ligand_nearest, q))
                      for q in (0., .1, .25, .5, .75))
        values.extend(_fractions(ligand_nearest, (2., 4., 6.)))
        protein_nearest = ligand_tree.query(protein, k=1, workers=1)[0]
        values.extend(_fractions(protein_nearest, (2., 4., 6., 8.)))
    for source, target_tree in ((e3, poi_tree), (poi, e3_tree)):
        nearest = target_tree.query(source, k=1, workers=1)[0]
        values.extend(float(np.quantile(nearest, q)) for q in (0., .01, .05))
        values.extend(_fractions(nearest, (2., 4., 6.)))
    values.append(float(np.linalg.norm(e3.mean(0) - poi.mean(0))))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(EXPANDED_GEOMETRY_NAMES),) or not np.isfinite(result).all():
        raise ValueError('invalid expanded geometry feature vector')
    if (result < 0).any():
        raise ValueError('distance features must be nonnegative')
    return result
