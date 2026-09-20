"""DeepTernary candidate selector features stay native-free and invariant."""

from pathlib import Path
import tempfile

import numpy as np

from tricoopdiff.deepternary_ranking import (
    FEATURE_NAMES, FEATURE_VERSION, deepternary_candidate_features,
)


def _record(kind, serial, atom, chain, xyz):
    x, y, z = xyz
    residue = 'LIG' if kind == 'HETATM' else 'ALA'
    return (f'{kind:<6}{serial:5d} {atom:^4} {residue:>3} {chain}{serial:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  ')


def _write(path, transform=lambda value: value):
    groups = [('ATOM', 'CA', 'A', [[0., 0., 0.], [2., 0., 0.]]),
              ('ATOM', 'CA', 'B', [[6., 0., 0.], [8., 1., 0.]]),
              ('HETATM', 'C1', 'A', [[1., 1., 0.], [5., 1., 0.]])]
    lines, serial = [], 1
    for kind, atom, chain, coordinates in groups:
        for coordinate in coordinates:
            lines.append(_record(kind, serial, atom, chain, transform(np.asarray(coordinate))))
            serial += 1
    path.write_text('\n'.join(lines) + '\n')


def test_deepternary_candidate_features_are_rigid_invariant_and_versioned():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); original = root / 'original.pdb'; moved = root / 'moved.pdb'
        _write(original)
        rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        _write(moved, lambda value: value @ rotation + 7.)
        first = deepternary_candidate_features(original, 2.5, .1)
        second = deepternary_candidate_features(moved, 2.5, .1)
    assert FEATURE_VERSION == 'deepternary_candidate_geometry_v1'
    assert len(FEATURE_NAMES) == first.shape[0] == 16
    assert np.allclose(first, second, atol=1e-5)


if __name__ == '__main__':
    test_deepternary_candidate_features_are_rigid_invariant_and_versioned()
    print('DeepTernary ranking feature test passed')
